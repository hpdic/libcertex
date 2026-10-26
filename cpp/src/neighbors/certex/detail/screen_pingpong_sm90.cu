/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// Warp-specialized certified pass on Hopper (sm_90a) for rows of 64, 128, or 256 halves.
//
// A thread block keeps one tile of 128 query rows resident in shared memory for the whole pass and
// splits its work over a producer warpgroup, which streams tiles of 64 database rows with TMA, and
// NC consumer warpgroups (two or three, by row length), which take the database tiles in turn.
// Each consumer has its own TMA pipeline, so that the epilogue of one consumer (threshold test and
// survivor output) overlaps the tensor-core work of the others. The reduction over the row length
// runs in one sequential loop per tile (no split-K), so that the accumulation error model of the
// rounding certificate holds.

#include <cuvs/core/cuda_fp16.hpp>  // first: __half with default visibility, as in cuVS

#include "kernels.hpp"

#include <cfloat>
#include <stdexcept>
#include <string>

#include <cute/tensor.hpp>

#include "cutlass/arch/barrier.h"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

namespace cuvs::neighbors::certex::detail {

using namespace cute;

namespace {

constexpr int kQ  = 128;  // query rows per block
constexpr int kX  = 64;   // database rows per consumer tile
constexpr int kBK = 64;   // halves per K tile (one 128-byte swizzle atom)

void check(cudaError_t e, const char* what)
{
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("certex: ") + what + ": " + cudaGetErrorString(e));
  }
}

template <int KB, int S, int NC>
struct Config {
  using SmemA = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                                       make_shape(Int<kQ>{}, Int<kBK>{}, Int<KB>{})));
  using SmemB = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<half_t>{},
                                       make_shape(Int<kX>{}, Int<kBK>{}, Int<KB * S * NC>{})));
  using Mma   = decltype(make_tiled_mma(
    SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{}));  // one warpgroup, M = 128
  struct Storage {
    alignas(128) cute::ArrayEngine<half_t, cosize_v<SmemA>> A;
    alignas(128) cute::ArrayEngine<half_t, cosize_v<SmemB>> B;
    uint64_t a_full;
    uint64_t full[NC][S];
    uint64_t empty[NC][S];
    float theta[kQ];
    uint32_t kept;
  };
};

struct Out {
  const float* theta;
  int32_t* q;
  int32_t* x;
  float* v;
  uint32_t* count;
  uint32_t cap;
  Filter filter;
};

template <int KB, int S, int NC, class TmaA, class TmaB>
__global__ __launch_bounds__(128 * (NC + 1),
                             1) void pingpong_kernel(int M,
                                                     int N,
                                                     CUTLASS_GRID_CONSTANT TmaA const tma_a,
                                                     CUTLASS_GRID_CONSTANT TmaB const tma_b,
                                                     Out out)
{
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)  // sm_90a only: TMA and WGMMA
  using C            = Config<KB, S, NC>;
  using Storage      = typename C::Storage;
  constexpr int kTpb = 128 * (NC + 1);
  // registers: the producer drops to 40, the NC consumers share the rest (multiples of 8)
  constexpr int kConsumerRegs = ((65536 - 128 * 40) / (128 * NC)) / 8 * 8 > 232
                                  ? 232
                                  : ((65536 - 128 * 40) / (128 * NC)) / 8 * 8;
  extern __shared__ char shared_memory[];
  Storage& smem = *reinterpret_cast<Storage*>(shared_memory);

  const int m_blk   = blockIdx.y;
  const int G       = gridDim.x;
  const int n_tiles = (N + kX - 1) / kX;
  const int mine    = blockIdx.x < n_tiles ? (n_tiles - 1 - blockIdx.x) / G + 1 : 0;  // tiles
  const int K       = KB * kBK;

  Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));
  Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));
  Tensor gA = local_tile(mA, make_shape(Int<kQ>{}, Int<kBK>{}), make_coord(m_blk, _));  // (Q,BK,KB)
  Tensor gB = local_tile(mB, make_shape(Int<kX>{}, Int<kBK>{}), make_coord(_, _));  // (X,BK,NT,KB)
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), typename C::SmemA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), typename C::SmemB{});
  auto [tAgA, tAsA] =
    tma_partition(tma_a, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sA), group_modes<0, 2>(gA));
  auto [tBgB, tBsB] =
    tma_partition(tma_b, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sB), group_modes<0, 2>(gB));
  constexpr int a_bytes = KB * sizeof(make_tensor_like(tensor<0>(tAsA)));
  constexpr int b_bytes = KB * sizeof(make_tensor_like(tensor<0>(tBsB)));

  using FullBar  = cutlass::arch::ClusterTransactionBarrier;
  using EmptyBar = cutlass::arch::ClusterBarrier;
  const int wg   = threadIdx.x / 128;
  if (threadIdx.x == 0) {
    FullBar::init(&smem.a_full, 1);
    for (int c = 0; c < NC; ++c) {
      for (int s = 0; s < S; ++s) {
        FullBar::init(&smem.full[c][s], 1);
        EmptyBar::init(&smem.empty[c][s], 128);
      }
    }
    smem.kept = 0;
  }
  for (int i = threadIdx.x; i < kQ; i += kTpb) {
    const int m   = m_blk * kQ + i;
    smem.theta[i] = m < M ? out.theta[m] : FLT_MAX;
  }
  cutlass::arch::fence_barrier_init();
  __syncthreads();

  if (wg == 0) {
    // producer: the query tile once, then the database tiles alternately into the two pipelines
    cutlass::arch::warpgroup_reg_dealloc<40>();
    if (threadIdx.x == 0) {
      FullBar::arrive_and_expect_tx(&smem.a_full, a_bytes);
      for (int kb = 0; kb < KB; ++kb) {
        copy(tma_a.with(smem.a_full), tAgA(_, kb), tAsA(_, kb));
      }
      cutlass::PipelineState<S> ws[NC];
      for (int i = 0; i < mine; ++i) {
        const int c = i % NC, t = blockIdx.x + i * G;
        const int s = ws[c].index();
        if (i >= NC * S) { EmptyBar::wait(&smem.empty[c][s], ws[c].phase() ^ 1); }
        FullBar::arrive_and_expect_tx(&smem.full[c][s], b_bytes);
        for (int kb = 0; kb < KB; ++kb) {
          copy(tma_b.with(smem.full[c][s]), tBgB(_, t, kb), tBsB(_, (c * S + s) * KB + kb));
        }
        ++ws[c];
      }
    }
    return;
  }

  // consumers: warpgroup 1 takes the even tiles of this block, warpgroup 2 the odd ones
  cutlass::arch::warpgroup_reg_alloc<kConsumerRegs>();
  const int c = wg - 1;
  typename C::Mma mma;
  ThrMMA thr  = mma.get_thread_slice(threadIdx.x % 128);
  Tensor tCsA = thr.partition_A(sA);  // (MMA, MMA_M, MMA_K, KB)
  Tensor tCsB = thr.partition_B(sB);  // (MMA, MMA_N, MMA_K, KB * S * 2)
  Tensor tCrA = thr.make_fragment_A(tCsA);
  Tensor tCrB = thr.make_fragment_B(tCsB);
  Tensor cC   = make_identity_tensor(make_shape(Int<kQ>{}, Int<kX>{}));
  Tensor tCcC = thr.partition_C(cC);  // (MMA, MMA_M = 2, MMA_N = 1)
  Tensor tCrC = thr.make_fragment_C(tCcC);
  static_assert(decltype(size(tCrC))::value == 64, "two 64 x 64 atoms per consumer thread");

  // rows of this thread: atom m holds rows rm and rm + 8, element v lies in row (v >> 1) & 1
  float th[2][2];
  int rows[2][2];
  CUTE_UNROLL
  for (int m = 0; m < 2; ++m) {
    rows[m][0] = get<0>(tCcC(0, m, 0));
    rows[m][1] = get<0>(tCcC(2, m, 0));
    th[m][0]   = smem.theta[rows[m][0]];
    th[m][1]   = smem.theta[rows[m][1]];
  }
  const int64_t seg = int64_t(blockIdx.y * gridDim.x + blockIdx.x) * out.cap;
  FullBar::wait(&smem.a_full, 0);

  cutlass::PipelineState<S> rs;
  for (int i = c; i < mine; i += NC) {
    const int s = rs.index();
    FullBar::wait(&smem.full[c][s], rs.phase());
    warpgroup_fence_operand(tCrC);
    warpgroup_arrive();
    // the first product overwrites the accumulators (scale-out zero): no clearing per tile
    CUTE_UNROLL
    for (int kb = 0; kb < KB; ++kb) {
      CUTE_UNROLL
      for (int k = 0; k < size<2>(tCrA); ++k) {
        mma.accumulate_ = (kb == 0 && k == 0) ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
        gemm(mma, tCrA(_, _, k, kb), tCrB(_, _, k, (c * S + s) * KB + kb), tCrC);
      }
    }
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);
    EmptyBar::arrive(&smem.empty[c][s]);
    ++rs;

    // epilogue of database tile t: the maximum of each of the four rows of this thread first; only
    // a row whose maximum reaches its threshold builds its mask and writes its kept pairs
    const int64_t n0 = int64_t(blockIdx.x + i * G) * kX;
    float mx[2][2]   = {{-FLT_MAX, -FLT_MAX}, {-FLT_MAX, -FLT_MAX}};
    CUTE_UNROLL
    for (int m = 0; m < 2; ++m) {
      CUTE_UNROLL
      for (int v = 0; v < size<0>(tCrC); ++v) {
        mx[m][(v >> 1) & 1] = fmaxf(mx[m][(v >> 1) & 1], tCrC(v, m, 0));
      }
    }
    const bool hot[2][2] = {{mx[0][0] >= th[0][0], mx[0][1] >= th[0][1]},
                            {mx[1][0] >= th[1][0], mx[1][1] >= th[1][1]}};
    if (!(hot[0][0] || hot[0][1] || hot[1][0] || hot[1][1])) { continue; }
    const bool last = n0 + kX > N;  // columns beyond N exist only in the last tile
    // one 16-bit mask per row: element j of row (m, r) is accumulator element v = 4 (j / 2) + 2 r +
    // j % 2; a branch per row keeps the work of a hot row at its 16 elements
    uint32_t mask[2][2] = {{0, 0}, {0, 0}};
    int hits            = 0;
    CUTE_UNROLL
    for (int m = 0; m < 2; ++m) {
      CUTE_UNROLL
      for (int r = 0; r < 2; ++r) {
        if (hot[m][r]) {
          CUTE_UNROLL
          for (int j = 0; j < 16; ++j) {
            const int v = 4 * (j / 2) + 2 * r + (j % 2);
            if (tCrC(v, m, 0) >= th[m][r] && (!last || n0 + get<1>(tCcC(v, m, 0)) < N) &&
                (out.filter.kind == Filter::kNone ||
                 out.filter.pass(m_blk * kQ + rows[m][r], n0 + get<1>(tCcC(v, m, 0))))) {
              mask[m][r] |= 1u << j;
            }
          }
          hits += __popc(mask[m][r]);
        }
      }
    }
    if (!hits) { continue; }
    uint32_t pos = atomicAdd(&smem.kept, uint32_t(hits));
    CUTE_UNROLL
    for (int m = 0; m < 2; ++m) {
      CUTE_UNROLL
      for (int r = 0; r < 2; ++r) {
        if (mask[m][r]) {
          CUTE_UNROLL
          for (int j = 0; j < 16; ++j) {
            const int v = 4 * (j / 2) + 2 * r + (j % 2);
            if ((mask[m][r] >> j) & 1) {
              if (pos < out.cap) {
                out.q[seg + pos] = m_blk * kQ + rows[m][r];
                out.x[seg + pos] = int32_t(n0 + get<1>(tCcC(v, m, 0)));
                out.v[seg + pos] = tCrC(v, m, 0);
              }
              ++pos;
            }
          }
        }
      }
    }
  }
  // the block's count, once both consumers are done
  asm volatile("bar.sync 1, %0;\n" ::"n"(128 * NC) : "memory");
  if (threadIdx.x == 128) { out.count[blockIdx.y * gridDim.x + blockIdx.x] = smem.kept; }
#endif
}

auto grid_x(int M) -> int
{
  const int m_tiles = (M + kQ - 1) / kQ;
  return std::max(1, sm_count() / m_tiles);
}

template <int KB, int S, int NC>
auto launch(const half* q_rows, int B, const half* x_rows, int64_t N, Out out, cudaStream_t stream)
  -> int
{
  using C     = Config<KB, S, NC>;
  auto* A     = reinterpret_cast<half_t const*>(q_rows);
  auto* Bp    = reinterpret_cast<half_t const*>(x_rows);
  const int K = KB * kBK;
  Tensor mA   = make_tensor(make_gmem_ptr(A), make_shape(B, K), make_stride(K, Int<1>{}));
  Tensor mB   = make_tensor(make_gmem_ptr(Bp), make_shape(int(N), K), make_stride(K, Int<1>{}));
  auto tma_a  = make_tma_atom(
    SM90_TMA_LOAD{}, mA, typename C::SmemA{}(_, _, 0), make_shape(Int<kQ>{}, Int<kBK>{}));
  auto tma_b = make_tma_atom(
    SM90_TMA_LOAD{}, mB, typename C::SmemB{}(_, _, 0), make_shape(Int<kX>{}, Int<kBK>{}));
  const int m_tiles = (B + kQ - 1) / kQ;
  const int n_tiles = int((N + kX - 1) / kX);
  const int G       = std::min(grid_x(B), n_tiles);
  const int smem    = int(sizeof(typename C::Storage));
  auto* kernel      = &pingpong_kernel<KB, S, NC, decltype(tma_a), decltype(tma_b)>;
  check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem),
        "smem attribute");
  kernel<<<dim3(G, m_tiles), 128 * (NC + 1), smem, stream>>>(B, int(N), tma_a, tma_b, out);
  check(cudaGetLastError(), "pingpong kernel launch");
  return G * m_tiles;
}

}  // namespace

auto pingpong_supported(int row_len) -> bool
{
  return row_len == 64 || row_len == 128 || row_len == 256;
}

auto pingpong_preferred(int row_len, int64_t n_rows) -> bool
{
  // Rows of 64 halves give a consumer only a short MMA per tile, so that the consumers cannot hide
  // the epilogue of tiles with survivors, and the cooperative pass (four warpgroups per SM) serves
  // collections of up to some 60M rows faster. Over more rows, the resident query tile of the
  // warp-specialized pass wins (BIGANN m = 59 on an H100: 0.76x at 20M, 0.97x at 50M, 1.07x at
  // 100M).
  return row_len == 128 || row_len == 256 || (row_len == 64 && n_rows >= (int64_t(1) << 26));
}

auto pingpong_segments(int B, int64_t N) -> int
{
  if (B == 0 || N == 0) { return 0; }
  return std::min<int64_t>(grid_x(B), (N + kX - 1) / kX) * ((B + kQ - 1) / kQ);
}

auto screen_pingpong(const half* q_rows,
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
  if (N >= (int64_t(1) << 31)) {
    throw std::invalid_argument("certex: at most 2^31 - 1 rows per shard");
  }
  const Out out{theta, seg_q, seg_x, seg_v, seg_count, seg_cap, filter};
  switch (row_len) {
    case 64: return launch<1, 6, 3>(q_rows, B, x_rows, N, out, stream);
    // three consumers measured fastest at 128 halves (BIGANN-10M, m = 123: 163k against 152k QPS)
    case 128: return launch<2, 3, 3>(q_rows, B, x_rows, N, out, stream);
    // three pipelines of 32 KiB stages do not fit next to the query tile: two consumers
    case 256: return launch<4, 2, 2>(q_rows, B, x_rows, N, out, stream);
    default: throw std::invalid_argument("certex: no warp-specialized pass for this row length");
  }
}

}  // namespace cuvs::neighbors::certex::detail
