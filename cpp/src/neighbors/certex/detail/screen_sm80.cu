/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// Tensor-core kernels of the certified search for Ampere (sm_80) and later: cp.async pipelines,
// ldmatrix, and mma.sync products with fp32 accumulation. Hopper uses the TMA and WGMMA kernels of
// screen_sm90.cu and screen_pingpong_sm90.cu; these kernels serve every other GPU of compute
// capability 8.0 or higher (and run on Hopper too, which tests use).
//
// Each thread block keeps one tile of 128 query rows and walks every G-th tile of 128 database rows
// (persistent over the database); the cp.async pipeline prefetches across tiles. The reduction over
// the row length runs in one sequential loop inside the block (no split-K), so that the
// accumulation error model of the rounding certificate holds. The routed mode walks a list of
// tiles instead (route_group in kernels.hpp), each pairing queries and rows of one label.

#include <cuvs/core/cuda_fp16.hpp>  // first: __half with default visibility, as in cuVS

#include "kernels.hpp"

#include <algorithm>
#include <cfloat>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <cute/tensor.hpp>

#include "cutlass/cutlass.h"

namespace cuvs::neighbors::certex::detail {

using namespace cute;

namespace {

constexpr int kM       = 128;  // query rows per block
constexpr int kN       = 128;  // database rows per tile
constexpr int kK       = 64;   // halves per K tile
constexpr int kStages  = 3;
constexpr int kThreads = 256;  // 8 warps, 4 along M and 2 along N

void check(cudaError_t e, const char* what)
{
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("certex: ") + what + ": " + cudaGetErrorString(e));
  }
}

using SwizzleAtom = decltype(composition(
  Swizzle<3, 3, 3>{}, Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, _64>>>{}));
// The query tile is resident (KT tiles of kK halves, the whole row, loaded once per block) for rows
// of up to 256 halves; longer rows and the routed pass, whose query tile changes, stage it with
// the database tile (KT = 0).
template <int KT>
using SmemA = decltype(tile_to_shape(
  SwizzleAtom{}, make_shape(Int<kM>{}, Int<kK>{}, Int<(KT > 0 ? KT : kStages)>{})));
using SmemB =
  decltype(tile_to_shape(SwizzleAtom{}, make_shape(Int<kN>{}, Int<kK>{}, Int<kStages>{})));
// 8 warps as 4 x 2 over (M, N); every warp covers 16 x 16 products per k-step, as ldmatrix x4
// loads for both operands
using Mma = decltype(make_tiled_mma(
  SM80_16x8x16_F32F16F16F32_TN{}, Layout<Shape<_4, _2, _1>>{}, Tile<_64, _32, _16>{}));
using S2R = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;

template <int KT>
struct Storage {
  alignas(128) cute::ArrayEngine<half_t, cosize_v<SmemA<KT>>> A;
  alignas(128) cute::ArrayEngine<half_t, cosize_v<SmemB>> B;
  float part_v[kM][2];  // anchor mode: the best of each half of a tile per row
  int part_i[kM][2];
  uint32_t kept;
};

struct Args {
  const half_t* q;
  const half_t* x;
  int M, N, K;
  // screening
  const float* theta;
  int32_t* seg_q;
  int32_t* seg_x;
  float* seg_v;
  uint32_t* seg_count;
  uint32_t seg_cap;
  // anchors
  float* best_v;
  int32_t* best_i;
  int groups;
  int stride;
  Filter filter;
  // routed mode
  const route_group* route;
  int n_groups;
  int n_tiles;
};

/** Load one K tile of 128 rows (rows beyond `rows` repeat the last row; their results are masked).
 */
template <class SmemTensor>
__device__ __forceinline__ void load_tile(
  SmemTensor& s, int stage, const half_t* base, int row0, int rows, int K, int kb)
{
  // 128 rows x 8 chunks of 16 bytes; 256 threads take 4 chunks each
  CUTE_UNROLL
  for (int t = 0; t < 4; ++t) {
    const int idx = threadIdx.x + t * kThreads;
    const int r = idx / 8, c = idx % 8;
    const int g       = min(row0 + r, rows - 1);
    const half_t* src = base + int64_t(g) * K + kb * kK + c * 8;
    half_t* dst       = &s(r, c * 8, stage);
    SM80_CP_ASYNC_CACHEALWAYS<uint128_t>::copy(*reinterpret_cast<const uint128_t*>(src),
                                               *reinterpret_cast<uint128_t*>(dst));
  }
}

template <bool kAnchor, bool kRouted, int KT>
__global__ __launch_bounds__(kThreads, 2) void sm80_kernel(Args a)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  extern __shared__ char shared_memory[];
  Storage<KT>& smem = *reinterpret_cast<Storage<KT>*>(shared_memory);
  Tensor sA         = make_tensor(make_smem_ptr(smem.A.begin()), SmemA<KT>{});
  Tensor sB         = make_tensor(make_smem_ptr(smem.B.begin()), SmemB{});

  const int m_blk   = blockIdx.y;
  const int G       = gridDim.x;
  const int n_tiles = kRouted ? a.n_tiles : (a.N + kN - 1) / kN;
  const int k_tiles = a.K / kK;
  const int mine    = blockIdx.x < n_tiles ? (n_tiles - 1 - blockIdx.x) / G + 1 : 0;
  const int iters   = mine * k_tiles;

  if (threadIdx.x == 0) { smem.kept = 0; }
  const int64_t seg = int64_t(blockIdx.y * gridDim.x + blockIdx.x) * a.seg_cap;

  auto issue = [&](int it) {
    if (it < iters) {
      const int t = blockIdx.x + (it / k_tiles) * G, kb = it % k_tiles, st = it % kStages;
      if constexpr (kRouted) {
        const RouteTile rt = route_tile(a.route, a.n_groups, t);
        load_tile(sA, st, a.q, rt.q0, a.M, a.K, kb);
        load_tile(sB, st, a.x, rt.x0, a.N, a.K, kb);
      } else {
        if constexpr (KT == 0) { load_tile(sA, st, a.q, m_blk * kM, a.M, a.K, kb); }
        load_tile(sB, st, a.x, t * kN, a.N, a.K, kb);
      }
    }
    cp_async_fence();
  };
  if constexpr (KT > 0) {  // the resident query tile joins the first group of loads
    if (iters > 0) {
      CUTE_UNROLL
      for (int kb = 0; kb < KT; ++kb) {
        load_tile(sA, kb, a.q, m_blk * kM, a.M, a.K, kb);
      }
    }
  }
  for (int it = 0; it < kStages - 1; ++it) {
    issue(it);
  }

  Mma mma;
  ThrMMA thr  = mma.get_slice(threadIdx.x);
  Tensor tCrA = thr.partition_fragment_A(sA(_, _, 0));  // (MMA, MMA_M, MMA_K)
  Tensor tCrB = thr.partition_fragment_B(sB(_, _, 0));  // (MMA, MMA_N, MMA_K)
  Tensor cC   = make_identity_tensor(make_shape(Int<kM>{}, Int<kN>{}));
  Tensor tCcC = thr.partition_C(cC);  // (MMA, MMA_M, MMA_N)
  Tensor tCrC = thr.make_fragment_C(tCcC);
  // the four accumulator rows of this thread: element (v, m, n) lies in row rows[m][(v >> 1) & 1]
  int rows[2][2];
  CUTE_UNROLL
  for (int m = 0; m < 2; ++m) {
    rows[m][0] = get<0>(tCcC(0, m, 0));
    rows[m][1] = get<0>(tCcC(2, m, 0));
  }
  float th[2][2];  // thresholds of the four rows (screening)
  if constexpr (!kAnchor && !kRouted) {
    CUTE_UNROLL
    for (int m = 0; m < 2; ++m) {
      CUTE_UNROLL
      for (int r = 0; r < 2; ++r) {
        const int q = m_blk * kM + rows[m][r];
        th[m][r]    = q < a.M ? a.theta[q] : FLT_MAX;
      }
    }
  }
  static_assert(decltype(size<1>(tCrC))::value == 2, "two MMA rows per thread");
  // whether any product of a row reaches its threshold; a thread without one pays the maxima
  auto any_kept = [&]() {
    bool hit = false;
    CUTE_UNROLL
    for (int m = 0; m < 2; ++m) {
      float m0 = -FLT_MAX, m1 = -FLT_MAX;
      CUTE_UNROLL
      for (int n = 0; n < size<2>(tCrC); ++n) {
        m0 = fmaxf(m0, fmaxf(tCrC(0, m, n), tCrC(1, m, n)));
        m1 = fmaxf(m1, fmaxf(tCrC(2, m, n), tCrC(3, m, n)));
      }
      hit |= m0 >= th[m][0] || m1 >= th[m][1];
    }
    return hit;
  };
  static_assert(decltype(size(tCrC))::value == 64, "one bit per accumulator element");
  // bit i of the mask: element i (linear index of tCrC, i = v + 4 (m + 2 n)) passes its threshold
  auto threshold_mask = [&]() {
    uint64_t mask = 0;
    CUTE_UNROLL
    for (int i = 0; i < 64; ++i) {
      mask |= uint64_t(tCrC(i) >= th[(i >> 2) & 1][(i >> 1) & 1]) << i;
    }
    return mask;
  };
  // one shared-memory atomic per thread reserves the positions of its pairs in the segment
  auto write_mask = [&](uint64_t mask, int q0, int64_t x0) {
    uint32_t pos = atomicAdd(&smem.kept, uint32_t(__popcll(mask)));
    CUTE_UNROLL
    for (int c = 0; c < 8; ++c) {
      if ((mask >> (8 * c)) & 0xffu) {
        CUTE_UNROLL
        for (int j = 0; j < 8; ++j) {
          const int i = 8 * c + j;
          if ((mask >> i) & 1) {
            if (pos < a.seg_cap) {
              a.seg_q[seg + pos] = q0 + get<0>(tCcC(i));
              a.seg_x[seg + pos] = int32_t(x0 + get<1>(tCcC(i)));
              a.seg_v[seg + pos] = tCrC(i);
            }
            ++pos;
          }
        }
      }
    }
  };
  auto s2r_a       = make_tiled_copy_A(S2R{}, mma);
  auto s2r_b       = make_tiled_copy_B(S2R{}, mma);
  auto thr_a       = s2r_a.get_slice(threadIdx.x);
  auto thr_b       = s2r_b.get_slice(threadIdx.x);
  Tensor tXsA      = thr_a.partition_S(sA);  // (CPY, MMA_M, MMA_K, STAGE)
  Tensor tXrA      = thr_a.retile_D(tCrA);
  Tensor tXsB      = thr_b.partition_S(sB);
  Tensor tXrB      = thr_b.retile_D(tCrB);
  constexpr int KB = decltype(size<2>(tCrA))::value;  // k16 blocks per K tile

  for (int it = 0; it < iters; ++it) {
    const int kb = it % k_tiles, st = it % kStages;
    cp_async_wait<kStages - 2>();
    __syncthreads();
    issue(it + kStages - 1);  // the slot consumed in the previous iteration
    if (kb == 0) { clear(tCrC); }
    CUTE_UNROLL
    for (int k = 0; k < KB; ++k) {
      copy(s2r_a, tXsA(_, _, k, KT > 0 ? kb : st), tXrA(_, _, k));
      copy(s2r_b, tXsB(_, _, k, st), tXrB(_, _, k));
      gemm(mma, tCrA(_, _, k), tCrB(_, _, k), tCrC);
    }
    if (kb != k_tiles - 1) { continue; }

    const int t = blockIdx.x + (it / k_tiles) * G;
    if constexpr (kRouted) {
      // rows outside the tile's queries never pass, columns outside the tile's rows are masked
      const RouteTile rt = route_tile(a.route, a.n_groups, t);
      CUTE_UNROLL
      for (int m = 0; m < 2; ++m) {
        CUTE_UNROLL
        for (int r = 0; r < 2; ++r) {
          const int q = rt.q0 + rows[m][r];
          th[m][r]    = q < rt.q1 ? a.theta[q] : FLT_MAX;
        }
      }
      if (any_kept()) {
        uint64_t mask = threshold_mask();
        if (mask && rt.x0 + kN > rt.x1) {
          CUTE_UNROLL
          for (int i = 0; i < 64; ++i) {
            if (rt.x0 + get<1>(tCcC(i)) >= rt.x1) { mask &= ~(uint64_t(1) << i); }
          }
        }
        if (mask) { write_mask(mask, rt.q0, rt.x0); }
      }
      continue;
    }
    const int64_t n0 = int64_t(t) * kN;
    if constexpr (!kAnchor) {
      // the row maxima first; the filter only meets pairs that pass their threshold
      if (any_kept()) {
        uint64_t mask = threshold_mask();
        if (mask && n0 + kN > a.N) {
          CUTE_UNROLL
          for (int i = 0; i < 64; ++i) {
            if (n0 + get<1>(tCcC(i)) >= a.N) { mask &= ~(uint64_t(1) << i); }
          }
        }
        if (mask && a.filter.kind != Filter::kNone) {
          CUTE_UNROLL
          for (int i = 0; i < 64; ++i) {
            if (((mask >> i) & 1) &&
                !a.filter.pass(int64_t(m_blk) * kM + get<0>(tCcC(i)), n0 + get<1>(tCcC(i)))) {
              mask &= ~(uint64_t(1) << i);
            }
          }
        }
        if (mask) { write_mask(mask, m_blk * kM, n0); }
      }
    } else {
      // the best column of every row in this warp's half of the tile, reduced over the quad
      float bv[2][2] = {{-FLT_MAX, -FLT_MAX}, {-FLT_MAX, -FLT_MAX}};
      int bi[2][2]   = {{0, 0}, {0, 0}};
      // a sample that the filter rejects never wins (one loop per case keeps registers low)
      auto scan = [&](auto filtered) {
        CUTE_UNROLL
        for (int m = 0; m < size<1>(tCrC); ++m) {
          CUTE_UNROLL
          for (int n = 0; n < size<2>(tCrC); ++n) {
            CUTE_UNROLL
            for (int v = 0; v < size<0>(tCrC); ++v) {
              const int s = (v >> 1) & 1;
              const int c = get<1>(tCcC(v, m, n));
              bool ok     = n0 + c < a.N;
              if constexpr (decltype(filtered)::value) {
                ok = ok &&
                     a.filter.pass(int64_t(m_blk) * kM + rows[m][s], (n0 + c) * int64_t(a.stride));
              }
              const float x = ok ? tCrC(v, m, n) : -FLT_MAX;
              if (x > bv[m][s] || (x == bv[m][s] && c < bi[m][s])) {
                bv[m][s] = x;
                bi[m][s] = c;
              }
            }
          }
        }
      };
      if (a.filter.kind == Filter::kNone) {
        scan(std::false_type{});
      } else {
        scan(std::true_type{});
      }
      const int wn = (threadIdx.x / 32) / 4;  // warp column: 0 or 1
      CUTE_UNROLL
      for (int m = 0; m < 2; ++m) {
        CUTE_UNROLL
        for (int s = 0; s < 2; ++s) {
          CUTE_UNROLL
          for (int off = 1; off <= 2; off <<= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv[m][s], off);
            const int oi   = __shfl_xor_sync(0xffffffffu, bi[m][s], off);
            if (ov > bv[m][s] || (ov == bv[m][s] && oi < bi[m][s])) {
              bv[m][s] = ov;
              bi[m][s] = oi;
            }
          }
          if ((threadIdx.x & 3) == 0) {
            smem.part_v[rows[m][s]][wn] = bv[m][s];
            smem.part_i[rows[m][s]][wn] = bi[m][s];
          }
        }
      }
      __syncthreads();
      if (threadIdx.x < kM) {
        const int r = threadIdx.x;
        const int w =
          (smem.part_v[r][1] > smem.part_v[r][0] ||
           (smem.part_v[r][1] == smem.part_v[r][0] && smem.part_i[r][1] < smem.part_i[r][0]))
            ? 1
            : 0;
        const int64_t m = int64_t(m_blk) * kM + r;
        if (m < a.M) {
          a.best_v[m * a.groups + t] = smem.part_v[r][w];
          a.best_i[m * a.groups + t] = int32_t(n0 + smem.part_i[r][w]);
        }
      }
      __syncthreads();
    }
  }
  if constexpr (!kAnchor) {
    __syncthreads();
    if (threadIdx.x == 0) { a.seg_count[blockIdx.y * gridDim.x + blockIdx.x] = smem.kept; }
  }
#endif
}

auto grid_x(int M, int64_t N) -> int
{
  const int m_tiles = (M + kM - 1) / kM;
  const int n_tiles = int((N + kN - 1) / kN);
  int G             = (2 * sm_count() + m_tiles - 1) / m_tiles;
  return G < 1 ? 1 : (G > n_tiles ? n_tiles : G);
}

auto routed_grid(int n_tiles) -> int { return std::max(1, std::min(2 * sm_count(), n_tiles)); }

template <bool kAnchor, bool kRouted, int KT>
auto launch_kt(Args a, cudaStream_t stream) -> int
{
  const int m_tiles = kRouted ? 1 : (a.M + kM - 1) / kM;
  const int G       = kRouted ? routed_grid(a.n_tiles) : grid_x(a.M, a.N);
  const int smem    = int(sizeof(Storage<KT>));
  auto* kernel      = &sm80_kernel<kAnchor, kRouted, KT>;
  check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem),
        "smem attribute");
  kernel<<<dim3(G, m_tiles), kThreads, smem, stream>>>(a);
  check(cudaGetLastError(), "sm80 kernel launch");
  return G * m_tiles;
}

template <bool kAnchor, bool kRouted = false>
auto launch(Args a, cudaStream_t stream) -> int
{
  if (a.K % kK != 0) { throw std::invalid_argument("certex: row length must be a multiple of 64"); }
  if constexpr (!kRouted) {
    switch (a.K / kK) {
      case 1: return launch_kt<kAnchor, false, 1>(a, stream);
      case 2: return launch_kt<kAnchor, false, 2>(a, stream);
      case 4: return launch_kt<kAnchor, false, 4>(a, stream);
      default: break;
    }
  }
  return launch_kt<kAnchor, kRouted, 0>(a, stream);
}

}  // namespace

auto sm80_routed_segments(int n_tiles) -> int { return n_tiles == 0 ? 0 : routed_grid(n_tiles); }

auto screen_routed_sm80(const half* q_rows,
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
                        cudaStream_t stream) -> int
{
  if (B == 0 || N == 0 || n_tiles == 0) { return 0; }
  Args a{};
  a.q         = reinterpret_cast<const half_t*>(q_rows);
  a.x         = reinterpret_cast<const half_t*>(x_rows);
  a.M         = B;
  a.N         = int(N);
  a.K         = row_len;
  a.theta     = theta;
  a.seg_q     = seg_q;
  a.seg_x     = seg_x;
  a.seg_v     = seg_v;
  a.seg_count = seg_count;
  a.seg_cap   = seg_cap;
  a.route     = groups;
  a.n_groups  = n_groups;
  a.n_tiles   = n_tiles;
  return launch<false, true>(a, stream);
}

auto sm80_segments(int B, int64_t N) -> int
{
  if (B == 0 || N == 0) { return 0; }
  return grid_x(B, N) * ((B + kM - 1) / kM);
}

auto screen_sm80(const half* q_rows,
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
                 cudaStream_t stream) -> int
{
  if (B == 0 || N == 0) { return 0; }
  Args a{};
  a.q         = reinterpret_cast<const half_t*>(q_rows);
  a.x         = reinterpret_cast<const half_t*>(x_rows);
  a.M         = B;
  a.N         = int(N);
  a.K         = row_len;
  a.theta     = theta;
  a.seg_q     = seg_q;
  a.seg_x     = seg_x;
  a.seg_v     = seg_v;
  a.seg_count = seg_count;
  a.seg_cap   = seg_cap;
  a.filter    = filter;
  return launch<false>(a, stream);
}

void anchor_groups_sm80(const half* q_rows,
                        int B,
                        const half* s_rows,
                        int S,
                        int row_len,
                        float* best_v,
                        int32_t* best_i,
                        Filter filter,
                        int stride,
                        cudaStream_t stream)
{
  if (B == 0 || S == 0) { return; }
  Args a{};
  a.q      = reinterpret_cast<const half_t*>(q_rows);
  a.x      = reinterpret_cast<const half_t*>(s_rows);
  a.M      = B;
  a.N      = S;
  a.K      = row_len;
  a.best_v = best_v;
  a.best_i = best_i;
  a.groups = (S + kN - 1) / kN;
  a.stride = stride;
  a.filter = filter;
  launch<true>(a, stream);
}

}  // namespace cuvs::neighbors::certex::detail
