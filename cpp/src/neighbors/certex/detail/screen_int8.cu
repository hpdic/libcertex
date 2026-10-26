/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// Certified pass over int8 screening rows (index_params::int8_rows).
//
// Every row holds the int8 codes c_x of its projection coordinates, z_x = s_x (c_x + e_x) with
// |e_x,j| <= 1/2, and five fp32 scalars: a_x / 2, s_x, |e_x|, |c_x|, and the residual norm r_x.
// The tensor cores accumulate <c_q, c_x> exactly in int32 (127^2 m < 2^24 converts exactly to
// fp32), and the epilogue keeps x iff
//
//   s_q s_x (<c_q, c_x> + |e_x| (|c_q| + |e_q|) + |e_q| |c_x|) + r_q r_x >= h_q + a_x / 2,
//
// with h_q = (a_q - U'(q)) / 2, which is the lifted bound a_q + a_x - 2 <l(q), l(x)> <= U'(q) of
// CertEx with the quantization error bounded by Cauchy-Schwarz. U'(q) adds to U the fp32 rounding
// margin of the epilogue (the caller's eta(q)). The pass writes the fp32 bound of every kept pair,
// from which the caller subtracts eta(q) to obtain a certified lower bound.
//
// The int8 products are cheap next to the per-pair epilogue, so that this kernel uses mma.sync with
// int32 accumulation on every GPU of compute capability 8.0 and later, Hopper included.

#include <cuvs/core/cuda_fp16.hpp>  // first: __half with default visibility, as in cuVS

#include "kernels.hpp"

#include <cfloat>
#include <stdexcept>
#include <string>

#include <cute/tensor.hpp>

#include "cutlass/cutlass.h"

namespace cuvs::neighbors::certex::detail {

using namespace cute;

namespace {

constexpr int kM       = 128;  // query rows per block
constexpr int kN       = 128;  // database rows per tile
constexpr int kK       = 64;   // int8 codes per K tile
constexpr int kPad     = 80;   // bytes per row in shared memory (16-byte aligned, fewer conflicts)
constexpr int kStages  = 3;
constexpr int kThreads = 256;

void check(cudaError_t e, const char* what)
{
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("certex: ") + what + ": " + cudaGetErrorString(e));
  }
}

using SmemA = decltype(make_layout(make_shape(Int<kM>{}, Int<kK>{}, Int<kStages>{}),
                                   make_stride(Int<kPad>{}, _1{}, Int<kM * kPad>{})));
using SmemB = decltype(make_layout(make_shape(Int<kN>{}, Int<kK>{}, Int<kStages>{}),
                                   make_stride(Int<kPad>{}, _1{}, Int<kN * kPad>{})));
using Mma   = decltype(make_tiled_mma(
  SM80_16x8x32_S32S8S8S32_TN{}, Layout<Shape<_4, _2, _1>>{}, Tile<_64, _32, _32>{}));

struct Storage {
  alignas(128) int8_t A[kM * kPad * kStages];
  alignas(128) int8_t B[kN * kPad * kStages];
  float hx[kN], sx[kN], dx[kN], ncx[kN], rx[kN];  // the scalars of the current database tile
  uint32_t kept;
};

struct Args {
  const int8_t* q;
  const int8_t* x;
  int M, N, K;
  const float* q_scal;  // [M, 5]: h_q, s_q, |e_q|, |c_q|, r_q
  const float* x_scal;  // [N, 5]: a_x / 2, s_x, |e_x|, |c_x|, r_x
  int32_t* seg_q;
  int32_t* seg_x;
  float* seg_v;
  uint32_t* seg_count;
  uint32_t seg_cap;
  Filter filter;
};

template <class SmemTensor>
__device__ __forceinline__ void load_tile(
  SmemTensor& s, int stage, const int8_t* base, int row0, int rows, int K, int kb)
{
  // 128 rows x 4 chunks of 16 bytes; 256 threads take 2 chunks each
  CUTE_UNROLL
  for (int t = 0; t < 2; ++t) {
    const int idx = threadIdx.x + t * kThreads;
    const int r = idx / 4, c = idx % 4;
    const int g = min(row0 + r, rows - 1);
    SM80_CP_ASYNC_CACHEALWAYS<uint128_t>::copy(
      *reinterpret_cast<const uint128_t*>(base + int64_t(g) * K + kb * kK + c * 16),
      *reinterpret_cast<uint128_t*>(&s(r, c * 16, stage)));
  }
}

__global__ __launch_bounds__(kThreads, 2) void int8_kernel(Args a)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  extern __shared__ char shared_memory[];
  Storage& smem = *reinterpret_cast<Storage*>(shared_memory);
  Tensor sA     = make_tensor(make_smem_ptr(smem.A), SmemA{});
  Tensor sB     = make_tensor(make_smem_ptr(smem.B), SmemB{});

  const int m_blk   = blockIdx.y;
  const int G       = gridDim.x;
  const int n_tiles = (a.N + kN - 1) / kN;
  const int k_tiles = a.K / kK;
  const int mine    = blockIdx.x < n_tiles ? (n_tiles - 1 - blockIdx.x) / G + 1 : 0;
  const int iters   = mine * k_tiles;
  if (threadIdx.x == 0) { smem.kept = 0; }
  const int64_t seg = int64_t(blockIdx.y * gridDim.x + blockIdx.x) * a.seg_cap;

  auto issue = [&](int it) {
    if (it < iters) {
      const int t = blockIdx.x + (it / k_tiles) * G, kb = it % k_tiles, st = it % kStages;
      load_tile(sA, st, a.q, m_blk * kM, a.M, a.K, kb);
      load_tile(sB, st, a.x, t * kN, a.N, a.K, kb);
    }
    cp_async_fence();
  };
  for (int it = 0; it < kStages - 1; ++it) {
    issue(it);
  }

  Mma mma;
  ThrMMA thr  = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr.partition_A(sA);  // (MMA, MMA_M, MMA_K, STAGE)
  Tensor tCsB = thr.partition_B(sB);
  Tensor tCrA = thr.partition_fragment_A(sA(_, _, 0));
  Tensor tCrB = thr.partition_fragment_B(sB(_, _, 0));
  Tensor cC   = make_identity_tensor(make_shape(Int<kM>{}, Int<kN>{}));
  Tensor tCcC = thr.partition_C(cC);
  Tensor tCrC = thr.make_fragment_C(tCcC);  // int32
  static_assert(decltype(size(tCrC))::value == 64, "one bit per accumulator element");
  constexpr int KB = decltype(size<2>(tCrA))::value;

  // the scalars of the four accumulator rows of this thread: element i lies in row
  // rows[(i >> 2) & 1][(i >> 1) & 1]
  int rows[2][2];
  float hq[2][2], sq[2][2], pq[2][2], dq[2][2], rq[2][2];
  CUTE_UNROLL
  for (int m = 0; m < 2; ++m) {
    CUTE_UNROLL
    for (int s = 0; s < 2; ++s) {
      rows[m][s]    = get<0>(tCcC(2 * s, m, 0));
      const int q   = m_blk * kM + rows[m][s];
      const bool ok = q < a.M;
      hq[m][s]      = ok ? a.q_scal[q * 5 + 0] : FLT_MAX;
      sq[m][s]      = ok ? a.q_scal[q * 5 + 1] : 0.0f;
      dq[m][s]      = ok ? a.q_scal[q * 5 + 2] : 0.0f;
      pq[m][s]      = ok ? a.q_scal[q * 5 + 3] + dq[m][s] : 0.0f;  // |c_q| + |e_q|
      rq[m][s]      = ok ? a.q_scal[q * 5 + 4] : 0.0f;
    }
  }

  for (int it = 0; it < iters; ++it) {
    const int kb = it % k_tiles, st = it % kStages;
    cp_async_wait<kStages - 2>();
    __syncthreads();
    issue(it + kStages - 1);
    if (kb == 0) { clear(tCrC); }
    CUTE_UNROLL
    for (int k = 0; k < KB; ++k) {
      copy(tCsA(_, _, k, st), tCrA(_, _, k));
      copy(tCsB(_, _, k, st), tCrB(_, _, k));
      gemm(mma, tCrA(_, _, k), tCrB(_, _, k), tCrC);
    }
    if (kb != k_tiles - 1) { continue; }

    // epilogue of database tile t: the scalars of its rows in shared memory, then one test per pair
    const int t      = blockIdx.x + (it / k_tiles) * G;
    const int64_t n0 = int64_t(t) * kN;
    __syncthreads();  // the previous tile's scalars are no longer read
    if (threadIdx.x < kN) {
      const int64_t x       = min(n0 + threadIdx.x, int64_t(a.N) - 1);
      const float* xs       = a.x_scal + x * 5;
      smem.hx[threadIdx.x]  = n0 + threadIdx.x < a.N ? xs[0] : FLT_MAX;
      smem.sx[threadIdx.x]  = xs[1];
      smem.dx[threadIdx.x]  = xs[2];
      smem.ncx[threadIdx.x] = xs[3];
      smem.rx[threadIdx.x]  = xs[4];
    }
    __syncthreads();
    uint64_t mask = 0;
    CUTE_UNROLL
    for (int i = 0; i < 64; ++i) {
      const int m = (i >> 2) & 1, s = (i >> 1) & 1;
      const int c = get<1>(tCcC(i));
      const float up =
        float(tCrC(i)) + smem.dx[c] * pq[m][s] + dq[m][s] * smem.ncx[c];  // exact int32 -> fp32
      const float v = sq[m][s] * smem.sx[c] * up + rq[m][s] * smem.rx[c];
      mask |= uint64_t(v >= hq[m][s] + smem.hx[c]) << i;
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
    if (mask) {
      uint32_t pos = atomicAdd(&smem.kept, uint32_t(__popcll(mask)));
      CUTE_UNROLL
      for (int i = 0; i < 64; ++i) {
        if ((mask >> i) & 1) {
          if (pos < a.seg_cap) {
            const int m = (i >> 2) & 1, s = (i >> 1) & 1;
            const int c    = get<1>(tCcC(i));
            const float up = float(tCrC(i)) + smem.dx[c] * pq[m][s] + dq[m][s] * smem.ncx[c];
            const float v  = sq[m][s] * smem.sx[c] * up + rq[m][s] * smem.rx[c];
            // the fp32 bound a_q + a_x - 2 v = 2 (h_q + a_x / 2 - v) + U'
            a.seg_q[seg + pos] = m_blk * kM + rows[m][s];
            a.seg_x[seg + pos] = int32_t(n0 + c);
            a.seg_v[seg + pos] = 2.0f * (hq[m][s] + smem.hx[c] - v);
          }
          ++pos;
        }
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) { a.seg_count[blockIdx.y * gridDim.x + blockIdx.x] = smem.kept; }
#endif
}

auto grid_x(int M, int64_t N) -> int
{
  const int m_tiles = (M + kM - 1) / kM;
  const int n_tiles = int((N + kN - 1) / kN);
  // at most two blocks per SM in all: one more block than that would start a second wave
  int G = 2 * sm_count() / m_tiles;
  return G < 1 ? 1 : (G > n_tiles ? n_tiles : G);
}

}  // namespace

auto int8_segments(int B, int64_t N) -> int
{
  if (B == 0 || N == 0) { return 0; }
  return grid_x(B, N) * ((B + kM - 1) / kM);
}

auto screen_int8(const int8_t* q_codes,
                 const float* q_scal,
                 int B,
                 const int8_t* x_codes,
                 const float* x_scal,
                 int64_t N,
                 int row_len,
                 int32_t* seg_q,
                 int32_t* seg_x,
                 float* seg_v,
                 uint32_t* seg_count,
                 uint32_t seg_cap,
                 Filter filter,
                 cudaStream_t stream) -> int
{
  require_supported_device();
  if (B == 0 || N == 0) { return 0; }
  if (row_len % kK != 0) {
    throw std::invalid_argument("certex: int8 rows hold a multiple of 64 codes");
  }
  if (N >= (int64_t(1) << 31)) {
    throw std::invalid_argument("certex: at most 2^31 - 1 rows per shard");
  }
  Args a{q_codes,
         x_codes,
         B,
         int(N),
         row_len,
         q_scal,
         x_scal,
         seg_q,
         seg_x,
         seg_v,
         seg_count,
         seg_cap,
         filter};
  const int m_tiles = (B + kM - 1) / kM;
  const int G       = grid_x(B, N);
  const int smem    = int(sizeof(Storage));
  check(cudaFuncSetAttribute(int8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem),
        "smem attribute");
  int8_kernel<<<dim3(G, m_tiles), kThreads, smem, stream>>>(a);
  check(cudaGetLastError(), "int8 kernel launch");
  return G * m_tiles;
}

}  // namespace cuvs::neighbors::certex::detail
