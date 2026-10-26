/*
 * Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

// Tensor-core kernels of the certified search on Hopper (sm_90a): TMA loads, WGMMA products with
// fp32 accumulation, and an epilogue that never writes the product matrix.
//
// Each thread block keeps one tile of kTileM query rows and walks every G-th tile of kTileN
// database rows (persistent over the database), with a TMA pipeline that prefetches across tiles.
// Per tile, the reduction over the row length runs in one sequential loop inside the block (no
// split-K), so that the accumulation error model of the rounding certificate (gamma = 16 L u) holds
// for every tile. Two blocks share one SM, so that the epilogue of one block overlaps the
// tensor-core work of the other.

#include <cuvs/core/cuda_fp16.hpp>  // first: __half with default visibility, as in cuVS

#include "kernels.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <stdexcept>
#include <string>

#include <cute/tensor.hpp>

#include "cutlass/arch/barrier.h"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

namespace cuvs::neighbors::certex::detail {

using namespace cute;

namespace {

constexpr int kBK   = 64;  // halves per K tile: one 128-byte swizzle atom
constexpr int kPipe = 3;   // TMA stages

void check(cudaError_t e, const char* what)
{
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("certex: ") + what + ": " + cudaGetErrorString(e));
  }
}

template <class SmemLayoutA, class SmemLayoutB>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<half_t, cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<half_t, cosize_v<SmemLayoutB>> B;
  uint64_t full[kPipe];
  uint64_t empty[kPipe];
  float theta[kTileM];
  uint32_t kept;
};

using TileShape = decltype(make_shape(Int<kTileM>{}, Int<kTileN>{}, Int<kBK>{}));
using SmemA     = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                                     make_shape(Int<kTileM>{}, Int<kBK>{}, Int<kPipe>{})));
using SmemB     = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                                     make_shape(Int<kTileN>{}, Int<kBK>{}, Int<kPipe>{})));
using Mma = decltype(make_tiled_mma(SM90_64x128x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{},
                                    Layout<Shape<_2, _1, _1>>{}));
constexpr int kThreads = 256;  // two warpgroups

struct ScreenOut {
  const float* theta;
  int32_t* q;
  int32_t* x;
  float* v;
  uint32_t* count;  // one per thread block
  uint32_t cap;     // per thread block
  Filter filter;
};

struct AnchorOut {
  float* best_v;
  int32_t* best_i;
  int groups;
  Filter filter;
  int stride;  // sample row s is database row s * stride
};

template <bool kAnchor, class TmaA, class TmaB>
__global__ __launch_bounds__(kThreads, 2) void tile_kernel(int M,
                                                           int N,
                                                           int K,
                                                           CUTLASS_GRID_CONSTANT TmaA const tma_a,
                                                           CUTLASS_GRID_CONSTANT TmaB const tma_b,
                                                           ScreenOut so,
                                                           AnchorOut ao)
{
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)  // sm_90a only: TMA and WGMMA
  extern __shared__ char shared_memory[];
  using Storage = SharedStorage<SmemA, SmemB>;
  Storage& smem = *reinterpret_cast<Storage*>(shared_memory);

  const int m_blk    = blockIdx.y;
  const int n_tiles  = (N + kTileN - 1) / kTileN;
  const int k_tiles  = K / kBK;
  const int G        = gridDim.x;
  const int my_tiles = blockIdx.x < n_tiles ? (n_tiles - 1 - blockIdx.x) / G + 1 : 0;
  const int iters    = my_tiles * k_tiles;

  Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));
  Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));
  Tensor gA =
    local_tile(mA, make_shape(Int<kTileM>{}, Int<kBK>{}), make_coord(m_blk, _));  // (BM,BK,KT)
  Tensor gB =
    local_tile(mB, make_shape(Int<kTileN>{}, Int<kBK>{}), make_coord(_, _));  // (BN,BK,NT,KT)
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemB{});
  auto [tAgA, tAsA] =
    tma_partition(tma_a, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sA), group_modes<0, 2>(gA));
  auto [tBgB, tBsB] =
    tma_partition(tma_b, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sB), group_modes<0, 2>(gB));
  constexpr int tx_bytes =
    sizeof(make_tensor_like(tensor<0>(tAsA))) + sizeof(make_tensor_like(tensor<0>(tBsB)));

  using FullBar       = cutlass::arch::ClusterTransactionBarrier;
  using EmptyBar      = cutlass::arch::ClusterBarrier;
  const int warp_idx  = cutlass::canonical_warp_idx_sync();
  const bool producer = warp_idx == 0 && cute::elect_one_sync();
  if (producer) {
    for (int p = 0; p < kPipe; ++p) {
      FullBar::init(&smem.full[p], 1);
      EmptyBar::init(&smem.empty[p], kThreads);
    }
  }
  if constexpr (!kAnchor) {
    for (int i = threadIdx.x; i < kTileM; i += kThreads) {
      int m         = m_blk * kTileM + i;
      smem.theta[i] = m < M ? so.theta[m] : FLT_MAX;
    }
  }
  if (threadIdx.x == 0) { smem.kept = 0; }
  const int64_t seg = int64_t(blockIdx.y * gridDim.x + blockIdx.x) * so.cap;
  cutlass::arch::fence_barrier_init();
  __syncthreads();

  auto issue = [&](int it, int stage) {
    int ti = it / k_tiles, kb = it % k_tiles;
    int t = blockIdx.x + ti * G;
    FullBar::arrive_and_expect_tx(&smem.full[stage], tx_bytes);
    copy(tma_a.with(smem.full[stage]), tAgA(_, kb), tAsA(_, stage));
    copy(tma_b.with(smem.full[stage]), tBgB(_, t, kb), tBsB(_, stage));
  };
  if (producer) {
    for (int it = 0; it < iters && it < kPipe; ++it) {
      issue(it, it);
    }
  }

  Mma mma;
  ThrMMA thr  = mma.get_thread_slice(threadIdx.x);
  Tensor tCsA = thr.partition_A(sA);
  Tensor tCsB = thr.partition_B(sB);
  Tensor cC   = make_identity_tensor(make_shape(Int<kTileM>{}, Int<kTileN>{}));
  Tensor tCcC = thr.partition_C(cC);
  Tensor tCrC = thr.make_fragment_C(tCcC);
  Tensor tCrA = thr.make_fragment_A(tCsA);
  Tensor tCrB = thr.make_fragment_B(tCsB);

  // the two accumulator rows of this thread (a 64 x N warpgroup tile gives every thread two rows)
  const int r0 = get<0>(tCcC(0));
  const int r1 = get<0>(tCcC(2));  // rows alternate in pairs of elements (see the epilogue)

  // rstate walks the stages that the MMA reads; wstate walks the refills, which wait for the
  // release of a stage
  auto rstate = cutlass::PipelineState<kPipe>();
  auto wstate = cutlass::PipelineState<kPipe>();

  // kb and ti count the K tile and the database tile of iteration it (no division per iteration)
  int kb = 0, ti = 0;
  for (int it = 0; it < iters; ++it) {
    const int stage = rstate.index();
    FullBar::wait(&smem.full[stage], rstate.phase());
    warpgroup_fence_operand(tCrC);
    warpgroup_arrive();
    if constexpr (kAnchor) {
      // the anchor pass keeps the cleared product and the division by k_tiles: with scale-out zero
      // and counters it takes twice as long (BIGANN-200M m = 59: 36.3 ms against 18.2 ms a batch)
      if (it % k_tiles == 0) { clear(tCrC); }
      gemm(mma, tCrA(_, _, _, stage), tCrB(_, _, _, stage), tCrC);
    } else {
      // the first product of a database tile overwrites the accumulators (scale-out zero), which
      // gives the same sums as clearing them first, without a select per accumulator
      CUTE_UNROLL
      for (int k = 0; k < size<2>(tCrA); ++k) {
        mma.accumulate_ = (kb == 0 && k == 0) ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
        gemm(mma, tCrA(_, _, k, stage), tCrB(_, _, k, stage), tCrC);
      }
    }
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);
    EmptyBar::arrive(&smem.empty[stage]);
    ++rstate;
    if (producer && it + kPipe < iters) {
      const int ws = wstate.index();
      EmptyBar::wait(&smem.empty[ws], wstate.phase());
      issue(it + kPipe, ws);
      ++wstate;
    }
    bool last;
    int t;  // the database tile of this iteration
    if constexpr (kAnchor) {
      last = it % k_tiles == k_tiles - 1;
      t    = blockIdx.x + (it / k_tiles) * G;
    } else {
      last = kb == k_tiles - 1;
      t    = blockIdx.x + ti * G;
      if (++kb == k_tiles) {
        kb = 0;
        ++ti;
      }
    }
    if (!last) { continue; }

    // epilogue of database tile t
    const int64_t n0 = int64_t(t) * kTileN;
    if constexpr (!kAnchor) {
      const float th0 = smem.theta[r0], th1 = smem.theta[r1];
      static_assert(decltype(size(tCrC))::value <= 64, "one bit per accumulator element");
      // In the accumulator of a 64 x N warpgroup tile, element i lies in row r0 when (i >> 1) is
      // even and in row r1 = r0 + 8 otherwise. The fast pass takes the maximum of each row, so that
      // a thread without a kept pair pays two comparisons beyond the maxima; only a thread with a
      // kept pair builds the mask of its elements.
      float m0 = -FLT_MAX, m1 = -FLT_MAX;
      CUTE_UNROLL
      for (int i = 0; i < size(tCrC); ++i) {
        if ((i >> 1) & 1) {
          m1 = fmaxf(m1, tCrC(i));
        } else {
          m0 = fmaxf(m0, tCrC(i));
        }
      }
      uint64_t mask = 0;
      if (m0 >= th0 || m1 >= th1) {
        // the slow path tests a group of four elements of one row only when the group's maximum
        // passes, with 32-bit masks (8 elements form two groups: rows r0 and r1)
        uint32_t lo = 0, hi = 0;
        CUTE_UNROLL
        for (int c = 0; c < size(tCrC) / 8; ++c) {
          const int i    = 8 * c;
          const float g0 = fmaxf(fmaxf(tCrC(i), tCrC(i + 1)), fmaxf(tCrC(i + 4), tCrC(i + 5)));
          const float g1 = fmaxf(fmaxf(tCrC(i + 2), tCrC(i + 3)), fmaxf(tCrC(i + 6), tCrC(i + 7)));
          uint32_t bits  = 0;
          if (g0 >= th0) {
            bits |= uint32_t(tCrC(i) >= th0) | uint32_t(tCrC(i + 1) >= th0) << 1 |
                    uint32_t(tCrC(i + 4) >= th0) << 4 | uint32_t(tCrC(i + 5) >= th0) << 5;
          }
          if (g1 >= th1) {
            bits |= uint32_t(tCrC(i + 2) >= th1) << 2 | uint32_t(tCrC(i + 3) >= th1) << 3 |
                    uint32_t(tCrC(i + 6) >= th1) << 6 | uint32_t(tCrC(i + 7) >= th1) << 7;
          }
          if (c < 4) {
            lo |= bits << (8 * c);
          } else {
            hi |= bits << (8 * (c - 4));
          }
        }
        mask = uint64_t(hi) << 32 | lo;
      }
      if (mask && n0 + kTileN > N) {
        CUTE_UNROLL
        for (int i = 0; i < size(tCrC); ++i) {
          if (n0 + get<1>(tCcC(i)) >= N) { mask &= ~(uint64_t(1) << i); }
        }
      }
      if (mask && so.filter.kind != Filter::kNone) {  // the filter only meets kept pairs
        CUTE_UNROLL
        for (int i = 0; i < size(tCrC); ++i) {
          if (((mask >> i) & 1) &&
              !so.filter.pass(m_blk * kTileM + get<0>(tCcC(i)), n0 + get<1>(tCcC(i)))) {
            mask &= ~(uint64_t(1) << i);
          }
        }
      }
      if (mask) {
        // a shared-memory atomic per thread with kept pairs reserves its positions in the block's
        // segment
        uint32_t pos = atomicAdd(&smem.kept, uint32_t(__popcll(mask)));
        CUTE_UNROLL
        for (int c = 0; c < size(tCrC) / 8; ++c) {
          if ((mask >> (8 * c)) & 0xffu) {
            CUTE_UNROLL
            for (int j = 0; j < 8; ++j) {
              const int i = 8 * c + j;
              if ((mask >> i) & 1) {
                if (pos < so.cap) {
                  so.q[seg + pos] = m_blk * kTileM + get<0>(tCcC(i));
                  so.x[seg + pos] = int32_t(n0 + get<1>(tCcC(i)));
                  so.v[seg + pos] = tCrC(i);
                }
                ++pos;
              }
            }
          }
        }
      }
    } else {
      float bv[2] = {-FLT_MAX, -FLT_MAX};
      int bi[2]   = {0, 0};
      CUTE_UNROLL
      for (int i = 0; i < size(tCrC); ++i) {
        const int s   = get<0>(tCcC(i)) == r0 ? 0 : 1;
        const int c   = get<1>(tCcC(i));
        const bool ok = n0 + c < N && (ao.filter.kind == Filter::kNone ||
                                       ao.filter.pass(m_blk * kTileM + get<0>(tCcC(i)),
                                                      (n0 + c) * int64_t(ao.stride)));
        const float v = ok ? tCrC(i) : -FLT_MAX;
        if (v > bv[s] || (v == bv[s] && c < bi[s])) {
          bv[s] = v;
          bi[s] = c;
        }
      }
      CUTE_UNROLL
      for (int s = 0; s < 2; ++s) {
        CUTE_UNROLL
        for (int off = 1; off <= 2;
             off <<= 1) {  // the four threads of a quad hold the columns of a row
          const float ov = __shfl_xor_sync(0xffffffffu, bv[s], off);
          const int oi   = __shfl_xor_sync(0xffffffffu, bi[s], off);
          if (ov > bv[s] || (ov == bv[s] && oi < bi[s])) {
            bv[s] = ov;
            bi[s] = oi;
          }
        }
      }
      if ((threadIdx.x & 3) == 0) {
        const int rows[2] = {r0, r1};
        CUTE_UNROLL
        for (int s = 0; s < 2; ++s) {
          const int m = m_blk * kTileM + rows[s];
          if (m < M) {
            ao.best_v[int64_t(m) * ao.groups + t] = bv[s];
            ao.best_i[int64_t(m) * ao.groups + t] = int32_t(n0 + bi[s]);
          }
        }
      }
    }
  }
  if constexpr (!kAnchor) {
    __syncthreads();
    if (threadIdx.x == 0) { so.count[blockIdx.y * gridDim.x + blockIdx.x] = smem.kept; }
  }
#endif
}

/**
 * Routed certified pass: the screening kernel of tile_kernel over a list of tiles, each pairing a
 * tile of the queries of one label with a tile of the rows of that label (rows sorted by label).
 * The TMA loads start at any row; rows of a loaded tile outside its label are masked in the
 * epilogue, so that a pair of different labels is never kept.
 */
template <class TmaA, class TmaB>
__global__ __launch_bounds__(kThreads, 2) void routed_kernel(int M,
                                                             int N,
                                                             int K,
                                                             CUTLASS_GRID_CONSTANT TmaA const tma_a,
                                                             CUTLASS_GRID_CONSTANT TmaB const tma_b,
                                                             const route_group* groups,
                                                             int n_groups,
                                                             int n_tiles,
                                                             ScreenOut so)
{
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)  // sm_90a only: TMA and WGMMA
  extern __shared__ char shared_memory[];
  using Storage = SharedStorage<SmemA, SmemB>;
  Storage& smem = *reinterpret_cast<Storage*>(shared_memory);

  const int k_tiles  = K / kBK;
  const int G        = gridDim.x;
  const int my_tiles = blockIdx.x < n_tiles ? (n_tiles - 1 - blockIdx.x) / G + 1 : 0;
  const int iters    = my_tiles * k_tiles;

  Tensor mA              = tma_a.get_tma_tensor(make_shape(M, K));
  Tensor mB              = tma_b.get_tma_tensor(make_shape(N, K));
  Tensor sA              = make_tensor(make_smem_ptr(smem.A.begin()), SmemA{});
  Tensor sB              = make_tensor(make_smem_ptr(smem.B.begin()), SmemB{});
  constexpr int tx_bytes = int(sizeof(half_t)) * (kTileM + kTileN) * kBK;

  using FullBar       = cutlass::arch::ClusterTransactionBarrier;
  using EmptyBar      = cutlass::arch::ClusterBarrier;
  const int warp_idx  = cutlass::canonical_warp_idx_sync();
  const bool producer = warp_idx == 0 && cute::elect_one_sync();
  if (producer) {
    for (int p = 0; p < kPipe; ++p) {
      FullBar::init(&smem.full[p], 1);
      EmptyBar::init(&smem.empty[p], kThreads);
    }
  }
  if (threadIdx.x == 0) { smem.kept = 0; }
  const int64_t seg = int64_t(blockIdx.x) * so.cap;
  cutlass::arch::fence_barrier_init();
  __syncthreads();

  auto issue = [&](int it, int stage) {
    const int kb       = it % k_tiles;
    const RouteTile rt = route_tile(groups, n_groups, blockIdx.x + (it / k_tiles) * G);
    Tensor gA          = local_tile(domain_offset(make_coord(rt.q0, 0), mA),
                           make_shape(Int<kTileM>{}, Int<kBK>{}),
                           make_coord(0, _));  // (BM, BK, KT)
    Tensor gB          = local_tile(domain_offset(make_coord(rt.x0, 0), mB),
                           make_shape(Int<kTileN>{}, Int<kBK>{}),
                           make_coord(0, _));  // (BN, BK, KT)
    auto [tAgA, tAsA] =
      tma_partition(tma_a, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sA), group_modes<0, 2>(gA));
    auto [tBgB, tBsB] =
      tma_partition(tma_b, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sB), group_modes<0, 2>(gB));
    FullBar::arrive_and_expect_tx(&smem.full[stage], tx_bytes);
    copy(tma_a.with(smem.full[stage]), tAgA(_, kb), tAsA(_, stage));
    copy(tma_b.with(smem.full[stage]), tBgB(_, kb), tBsB(_, stage));
  };
  if (producer) {
    for (int it = 0; it < iters && it < kPipe; ++it) {
      issue(it, it);
    }
  }

  Mma mma;
  ThrMMA thr   = mma.get_thread_slice(threadIdx.x);
  Tensor tCsA  = thr.partition_A(sA);
  Tensor tCsB  = thr.partition_B(sB);
  Tensor cC    = make_identity_tensor(make_shape(Int<kTileM>{}, Int<kTileN>{}));
  Tensor tCcC  = thr.partition_C(cC);
  Tensor tCrC  = thr.make_fragment_C(tCcC);
  Tensor tCrA  = thr.make_fragment_A(tCsA);
  Tensor tCrB  = thr.make_fragment_B(tCsB);
  const int r0 = get<0>(tCcC(0));
  const int r1 = get<0>(tCcC(2));

  auto rstate = cutlass::PipelineState<kPipe>();
  auto wstate = cutlass::PipelineState<kPipe>();

  int kb = 0, ti = 0;  // K tile and work item of iteration it, as in tile_kernel
  for (int it = 0; it < iters; ++it) {
    const int stage = rstate.index();
    FullBar::wait(&smem.full[stage], rstate.phase());
    warpgroup_fence_operand(tCrC);
    warpgroup_arrive();
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tCrA); ++k) {
      mma.accumulate_ = (kb == 0 && k == 0) ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
      gemm(mma, tCrA(_, _, k, stage), tCrB(_, _, k, stage), tCrC);
    }
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);
    EmptyBar::arrive(&smem.empty[stage]);
    ++rstate;
    if (producer && it + kPipe < iters) {
      const int ws = wstate.index();
      EmptyBar::wait(&smem.empty[ws], wstate.phase());
      issue(it + kPipe, ws);
      ++wstate;
    }
    const bool last = kb == k_tiles - 1;
    const int item  = blockIdx.x + ti * G;
    if (++kb == k_tiles) {
      kb = 0;
      ++ti;
    }
    if (!last) { continue; }

    // epilogue: rows outside the tile's queries never pass (threshold FLT_MAX), columns outside
    // the tile's rows are masked
    const RouteTile rt = route_tile(groups, n_groups, item);
    const float th0    = rt.q0 + r0 < rt.q1 ? so.theta[rt.q0 + r0] : FLT_MAX;
    const float th1    = rt.q0 + r1 < rt.q1 ? so.theta[rt.q0 + r1] : FLT_MAX;
    float m0 = -FLT_MAX, m1 = -FLT_MAX;
    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
      if ((i >> 1) & 1) {
        m1 = fmaxf(m1, tCrC(i));
      } else {
        m0 = fmaxf(m0, tCrC(i));
      }
    }
    uint64_t mask = 0;
    if (m0 >= th0 || m1 >= th1) {
      CUTE_UNROLL
      for (int i = 0; i < size(tCrC); ++i) {
        mask |= uint64_t(tCrC(i) >= (((i >> 1) & 1) ? th1 : th0) && rt.x0 + get<1>(tCcC(i)) < rt.x1)
                << i;
      }
    }
    if (mask) {
      uint32_t pos = atomicAdd(&smem.kept, uint32_t(__popcll(mask)));
      CUTE_UNROLL
      for (int i = 0; i < size(tCrC); ++i) {
        if ((mask >> i) & 1) {
          if (pos < so.cap) {
            so.q[seg + pos] = rt.q0 + get<0>(tCcC(i));
            so.x[seg + pos] = rt.x0 + get<1>(tCcC(i));
            so.v[seg + pos] = tCrC(i);
          }
          ++pos;
        }
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) { so.count[blockIdx.x] = smem.kept; }
#endif
}

int grid_x(int M, int64_t N)
{
  const int m_tiles = (M + kTileM - 1) / kTileM;
  const int n_tiles = int((N + kTileN - 1) / kTileN);
  // at most two blocks per SM in all: one more block than that would start a second wave
  int G = 2 * sm_count() / m_tiles;
  return G < 1 ? 1 : (G > n_tiles ? n_tiles : G);
}

template <bool kAnchor>
int launch(const half* q_rows,
           int B,
           const half* x_rows,
           int64_t N,
           int row_len,
           ScreenOut so,
           AnchorOut ao,
           cudaStream_t stream)
{
  if (row_len % kBK != 0) {
    throw std::invalid_argument("certex: row length must be a multiple of 64");
  }
  if (N >= (int64_t(1) << 31)) {
    throw std::invalid_argument("certex: at most 2^31 - 1 rows per shard");
  }
  if (B == 0 || N == 0) { return 0; }
  const int M = B, Nn = int(N), K = row_len;
  auto* A   = reinterpret_cast<half_t const*>(q_rows);
  auto* Bp  = reinterpret_cast<half_t const*>(x_rows);
  Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, Int<1>{}));
  Tensor mB = make_tensor(make_gmem_ptr(Bp), make_shape(Nn, K), make_stride(K, Int<1>{}));
  auto tma_a =
    make_tma_atom(SM90_TMA_LOAD{}, mA, SmemA{}(_, _, 0), make_shape(Int<kTileM>{}, Int<kBK>{}));
  auto tma_b =
    make_tma_atom(SM90_TMA_LOAD{}, mB, SmemB{}(_, _, 0), make_shape(Int<kTileN>{}, Int<kBK>{}));

  const int m_tiles = (M + kTileM - 1) / kTileM;
  const int G       = grid_x(M, N);
  const int smem    = int(sizeof(SharedStorage<SmemA, SmemB>));
  auto* kernel      = &tile_kernel<kAnchor, decltype(tma_a), decltype(tma_b)>;
  check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem),
        "smem attribute");
  kernel<<<dim3(G, m_tiles), kThreads, smem, stream>>>(M, Nn, K, tma_a, tma_b, so, ao);
  check(cudaGetLastError(), "tile kernel launch");
  return G * m_tiles;
}

auto routed_grid(int n_tiles) -> int { return std::max(1, std::min(2 * sm_count(), n_tiles)); }

}  // namespace

namespace {

std::atomic<screen_kernel> kernel_override{screen_kernel::automatic};

/** Compute capability major of the current device, cached per device. */
auto device_major() -> int
{
  static std::array<std::atomic<int>, 64> cache{};  // 0: not yet queried
  int dev = 0;
  cudaGetDevice(&dev);
  auto& slot = cache[dev & 63];
  int v      = slot.load();
  if (v == 0) {
    cudaDeviceGetAttribute(&v, cudaDevAttrComputeCapabilityMajor, dev);
    slot.store(v);
  }
  return v;
}

}  // namespace

void require_supported_device()
{
  if (device_major() < 8) {
    throw std::invalid_argument("certex: needs a GPU of compute capability 8.0 or later");
  }
}

namespace {

/**
 * The kernel that runs: an explicit choice, the override, or the best one for the device. The
 * TMA and WGMMA kernels exist for compute capability 9.0 (sm_90a) only; every other GPU of
 * compute capability 8.0 or later, Blackwell included, runs the mma.sync kernels.
 */
auto resolve(screen_kernel kernel, int row_len, int64_t n_rows = 0) -> screen_kernel
{
  require_supported_device();
  if (kernel == screen_kernel::automatic) { kernel = kernel_override.load(); }
  const bool hopper = device_major() == 9;
  if (kernel == screen_kernel::automatic) {
    if (!hopper) { return screen_kernel::sm80; }
    return pingpong_preferred(row_len, n_rows) ? screen_kernel::pingpong
                                               : screen_kernel::cooperative;
  }
  if (kernel != screen_kernel::sm80 && !hopper) {
    throw std::invalid_argument("certex: the TMA and WGMMA kernels need compute capability 9.0");
  }
  if (kernel == screen_kernel::pingpong && !pingpong_supported(row_len)) {
    throw std::invalid_argument("certex: no warp-specialized pass for this row length");
  }
  return kernel;
}

}  // namespace

void set_kernel_override(screen_kernel kernel) { kernel_override.store(kernel); }

int routed_segments(int n_tiles, screen_kernel kernel)
{
  if (resolve(kernel, 64) == screen_kernel::sm80) { return sm80_routed_segments(n_tiles); }
  return n_tiles == 0 ? 0 : routed_grid(n_tiles);
}

int screen_routed(const half* q_rows,
                  int B,
                  const half* x_rows,
                  int64_t N,
                  int row_len,
                  const float* theta,
                  const route_group* groups,
                  int n_groups,
                  int n_tiles,
                  int32_t* seg_q,
                  int32_t* seg_x,
                  float* seg_v,
                  uint32_t* seg_count,
                  uint32_t seg_cap,
                  cudaStream_t stream,
                  screen_kernel kernel)
{
  if (resolve(kernel, row_len) == screen_kernel::sm80) {
    return screen_routed_sm80(q_rows,
                              B,
                              x_rows,
                              N,
                              row_len,
                              theta,
                              groups,
                              n_groups,
                              n_tiles,
                              seg_q,
                              seg_x,
                              seg_v,
                              seg_count,
                              seg_cap,
                              stream);
  }
  if (row_len % kBK != 0) {
    throw std::invalid_argument("certex: row length must be a multiple of 64");
  }
  if (N >= (int64_t(1) << 31)) {
    throw std::invalid_argument("certex: at most 2^31 - 1 rows per shard");
  }
  if (B == 0 || N == 0 || n_tiles == 0) { return 0; }
  const int M = B, Nn = int(N), K = row_len;
  auto* A   = reinterpret_cast<half_t const*>(q_rows);
  auto* Bp  = reinterpret_cast<half_t const*>(x_rows);
  Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, Int<1>{}));
  Tensor mB = make_tensor(make_gmem_ptr(Bp), make_shape(Nn, K), make_stride(K, Int<1>{}));
  auto tma_a =
    make_tma_atom(SM90_TMA_LOAD{}, mA, SmemA{}(_, _, 0), make_shape(Int<kTileM>{}, Int<kBK>{}));
  auto tma_b =
    make_tma_atom(SM90_TMA_LOAD{}, mB, SmemB{}(_, _, 0), make_shape(Int<kTileN>{}, Int<kBK>{}));
  const int G    = routed_grid(n_tiles);
  const int smem = int(sizeof(SharedStorage<SmemA, SmemB>));
  auto* fn       = &routed_kernel<decltype(tma_a), decltype(tma_b)>;
  check(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem),
        "smem attribute");
  fn<<<G, kThreads, smem, stream>>>(M,
                                    Nn,
                                    K,
                                    tma_a,
                                    tma_b,
                                    groups,
                                    n_groups,
                                    n_tiles,
                                    ScreenOut{theta, seg_q, seg_x, seg_v, seg_count, seg_cap, {}});
  check(cudaGetLastError(), "routed kernel launch");
  return G;
}

int sm_count()
{
  static int n = [] {
    int dev = 0, v = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev);
    return v;
  }();
  return n;
}

int cooperative_segments(int B, int64_t N)
{
  if (B == 0 || N == 0) { return 0; }
  return grid_x(B, N) * ((B + kTileM - 1) / kTileM);
}

int screen_segments(int B, int64_t N, int row_len, screen_kernel kernel)
{
  switch (resolve(kernel, row_len, N)) {
    case screen_kernel::pingpong: return pingpong_segments(B, N);
    case screen_kernel::sm80: return sm80_segments(B, N);
    default: return cooperative_segments(B, N);
  }
}

int screen(const half* q_rows,
           int B,
           const half* x_rows,
           int64_t N,
           int row_len,
           const float* theta,
           int32_t* seg_q,
           int32_t* seg_x,
           float* seg_v,
           uint32_t* seg_count,
           uint32_t seg_cap,
           Filter filter,
           cudaStream_t stream,
           screen_kernel kernel)
{
  const screen_kernel k = resolve(kernel, row_len, N);
  if (k == screen_kernel::sm80) {
    return screen_sm80(q_rows,
                       B,
                       x_rows,
                       N,
                       row_len,
                       theta,
                       seg_q,
                       seg_x,
                       seg_v,
                       seg_count,
                       seg_cap,
                       filter,
                       stream);
  }
  if (k == screen_kernel::pingpong) {
    return screen_pingpong(q_rows,
                           B,
                           x_rows,
                           N,
                           row_len,
                           theta,
                           seg_q,
                           seg_x,
                           seg_v,
                           seg_count,
                           seg_cap,
                           filter,
                           stream);
  }
  return launch<false>(q_rows,
                       B,
                       x_rows,
                       N,
                       row_len,
                       ScreenOut{theta, seg_q, seg_x, seg_v, seg_count, seg_cap, filter},
                       AnchorOut{},
                       stream);
}

void anchor_groups(const half* q_rows,
                   int B,
                   const half* s_rows,
                   int S,
                   int row_len,
                   float* best_v,
                   int32_t* best_i,
                   Filter filter,
                   int stride,
                   cudaStream_t stream,
                   screen_kernel kernel)
{
  if (resolve(kernel, row_len) == screen_kernel::sm80) {
    anchor_groups_sm80(q_rows, B, s_rows, S, row_len, best_v, best_i, filter, stride, stream);
    return;
  }
  launch<true>(q_rows,
               B,
               s_rows,
               S,
               row_len,
               ScreenOut{},
               AnchorOut{best_v, best_i, (S + kTileN - 1) / kTileN, filter, stride},
               stream);
}

}  // namespace cuvs::neighbors::certex::detail
