/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>

// Kernel launchers of the certified search. The interface uses raw device pointers and a CUDA
// stream only, so that the tensor-core kernels (CUTLASS / CuTe) and the cuVS-facing layer (RAFT)
// compile in separate translation units.

#include <cstdint>
#include <stdexcept>

#include <cuvs/core/cuda_fp16.hpp>  // first: __half with default visibility, as in cuVS

#include <cuda_runtime.h>

namespace raft {
class resources;  // raft/core/resources.hpp, which needs C++20 (the kernels build as C++17)
}  // namespace raft

namespace CUVS_EXPORT cuvs {
namespace neighbors::certex::detail {

/** A batch whose survivors exceed the 32-bit segment addressing of a certified pass; the search
 * splits the batch. */
struct batch_too_large : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/**
 * Sample filter of a search, tested on the rare pairs that pass the certified test (and on the
 * anchors): a cuVS bitset over the rows (bit set = kept), a cuVS bitmap with one row of bits per
 * query, or one label per row and per query (a row is kept for the queries of its label).
 */
struct Filter {
  enum Kind : int { kNone = 0, kBitset = 1, kBitmap = 2, kLabels = 3 };
  int kind                    = kNone;
  const uint32_t* bits        = nullptr;  // bitset or bitmap words
  int64_t n_cols              = 0;        // bits per bitmap row
  const int32_t* row_labels   = nullptr;  // label of every row
  const int32_t* query_labels = nullptr;  // label of every query
  int64_t q0                  = 0;        // index of the first query of the batch

  /** Whether row `row` may be a neighbor of query `q` of the batch. In an index with lists, the
   *  caller translates bitsets and bitmaps from ids to rows first. */
  __device__ __forceinline__ bool pass(int64_t q, int64_t row) const
  {
    switch (kind) {
      case kBitset: return (bits[row >> 5] >> (row & 31)) & 1u;
      case kBitmap: {
        const int64_t b = (q0 + q) * n_cols + row;
        return (bits[b >> 5] >> (b & 31)) & 1u;
      }
      case kLabels: return row_labels[row] == query_labels[q0 + q];
      default: return true;
    }
  }
};

/** Columns of the database tile of the tensor-core kernels; also the group size of the anchor
 * kernel. */
constexpr int kTileN = 128;
/** Query rows per thread block of the tensor-core kernels. */
constexpr int kTileM = 128;

/** Kernel of the certified pass: the fastest for the row length, or one of the two by name. */
enum class screen_kernel {
  automatic,    // Hopper: warp-specialized for rows of 128 and 256 halves, cooperative otherwise;
                // compute capability 8.x: sm80
  cooperative,  // screen_sm90.cu: two consumer warpgroups on 128 x 128 tiles, any row length
  pingpong,     // screen_pingpong_sm90.cu: producer and consumers, rows of 64, 128, or 256 halves
  sm80          // screen_sm80.cu: cp.async and mma.sync, for compute capability 8.0 and later
};

/**
 * Kernel that `automatic` resolves to in this process (automatic: by the device), so that tests
 * and benchmarks can run the sm80 kernels on a Hopper GPU. Not part of the public interface.
 */
void set_kernel_override(screen_kernel kernel);

/**
 * Throws std::invalid_argument unless the current device has compute capability 8.0 or later: the
 * screening kernels use mma.sync (sm_80) or TMA and WGMMA (sm_90a), and on older GPUs they would
 * compile to empty bodies.
 */
void require_supported_device();

/**
 * Certified pass: for every query row i < B and database row j < N, keep the pair if the fp32
 * tensor-core product of the half-precision rows satisfies dot(q_i, x_j) >= theta[i]. Every thread
 * block b appends its kept pairs to its own segment [b * seg_cap, (b + 1) * seg_cap) of (seg_q,
 * seg_x, seg_v) and reports their number in seg_count[b], so that no two blocks contend for one
 * counter. A count above seg_cap marks an overflow of that segment (the pairs beyond seg_cap are
 * not written), and the caller repeats with a larger seg_cap. Rows hold `row_len` halves (a
 * multiple of 64). Returns the number of segments, which screen_segments() gives in advance.
 */
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
           screen_kernel kernel = screen_kernel::automatic);

/** Number of segments (thread blocks) of screen() for B queries, N rows, and the row length. */
int screen_segments(int B, int64_t N, int row_len, screen_kernel kernel = screen_kernel::automatic);

/**
 * Anchor scoring: for every query row i < B and every group g of kTileN consecutive sample rows,
 * the largest product best_v[i * G + g] and the sample index best_i[i * G + g] that attains it,
 * with G = ceil(S / kTileN). Sample row s is database row s * stride; rows that the filter rejects
 * never win a group (a group without any passing row reports -FLT_MAX).
 */
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
                   screen_kernel kernel = screen_kernel::automatic);

/** Warp-specialized certified pass (screen_pingpong_sm90.cu) for rows of 64, 128, or 256 halves;
 *  same contract as screen(). */
auto pingpong_supported(int row_len) -> bool;
/** Whether the warp-specialized pass is the faster one for the row length. */
auto pingpong_preferred(int row_len, int64_t n_rows = 0) -> bool;
auto pingpong_segments(int B, int64_t N) -> int;
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
                     cudaStream_t stream) -> int;

/** One label of a routed pass: queries [q0, q0 + nq) of the batch and rows [x0, x0 + nx). */
struct route_group {
  int32_t q0, nq;
  int32_t x0, nx;
  int32_t tile0;    // index of the group's first tile in the routed pass
  int32_t x_tiles;  // ceil(nx / kTileN); the group has ceil(nq / kTileM) x_tiles tiles
};

/** The tile t of a routed pass: query rows [q0, q1) of the batch against rows [x0, x1). */
struct RouteTile {
  int q0, q1, x0, x1;
};

#ifdef __CUDACC__
__device__ __forceinline__ RouteTile route_tile(const route_group* groups, int n_groups, int t)
{
  int lo = 0, hi = n_groups - 1;  // the last group whose first tile is at most t
  while (lo < hi) {
    const int mid = (lo + hi + 1) / 2;
    if (groups[mid].tile0 <= t) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  const route_group g = groups[lo];
  const int l = t - g.tile0, qt = l / g.x_tiles, xt = l % g.x_tiles;
  const int q0 = g.q0 + qt * kTileM, x0 = g.x0 + xt * kTileN;
  return {q0, min(q0 + kTileM, g.q0 + g.nq), x0, min(x0 + kTileN, g.x0 + g.nx)};
}
#endif

/**
 * Routed certified pass (cooperative kernel) over rows sorted by label: the queries of every group
 * meet only the rows of their group, with the contract of screen() otherwise. Returns the number
 * of segments, which routed_segments() gives in advance.
 */
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
                  screen_kernel kernel = screen_kernel::automatic);
int routed_segments(int n_tiles, screen_kernel kernel = screen_kernel::automatic);

/** The kernels of screen_sm80.cu, with the contracts of screen(), anchor_groups(), and
 *  screen_routed(). */
auto sm80_segments(int B, int64_t N) -> int;
auto sm80_routed_segments(int n_tiles) -> int;
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
                 cudaStream_t stream) -> int;
void anchor_groups_sm80(const half* q_rows,
                        int B,
                        const half* s_rows,
                        int S,
                        int row_len,
                        float* best_v,
                        int32_t* best_i,
                        Filter filter,
                        int stride,
                        cudaStream_t stream);
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
                        cudaStream_t stream) -> int;

/** Counters of the searches of the calling thread, for the tuner and the benchmarks. */
struct search_stats {
  int64_t queries   = 0;
  int64_t survivors = 0;  // pairs kept by the certified pass
  int64_t verified  = 0;  // exact scores computed by ordered verification
};
/** Accumulate the counters of later searches of this thread into s (null: stop). */
void collect_stats(search_stats* s);
auto stats_sink() -> search_stats*;

/**
 * Thresholds shared across shards (certex_shards.cu), for the searches of the calling thread: with
 * u_out, a search stops after the threshold stage and writes U of every query to u_out[b0 + q];
 * with u_cap, it lowers U of every query to u_cap[b0 + q] before the certified pass. Both are
 * device arrays over the queries of the call; b0 is the first query of the current batch.
 */
struct threshold_hooks {
  const double* u_cap = nullptr;
  double* u_out       = nullptr;
  int64_t b0          = 0;
};
void set_threshold_hooks(threshold_hooks* h);
auto threshold_hooks_sink() -> threshold_hooks*;

/** out[i] = i for i < n, on the stream of res. */
void iota_fill(raft::resources const& res, int32_t* out, int64_t n);
/** inverse[perm[i]] = i for i < n, on the stream of res. */
void invert_into(raft::resources const& res, const int32_t* perm, int64_t n, int32_t* inverse);

/**
 * Certified pass over int8 screening rows (screen_int8.cu): q_codes [B, row_len] and x_codes
 * [N, row_len] int8, q_scal [B, 5] = (h_q, s_q, |e_q|, |c_q|, r_q), x_scal [N, 5] = (a_x / 2, s_x,
 * |e_x|, |c_x|, r_x). A pair is kept when its fp32 bound does not exceed U'(q), and seg_v holds the
 * bound minus U'(q). Segments as for screen().
 */
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
                 cudaStream_t stream) -> int;
auto int8_segments(int B, int64_t N) -> int;

/** Number of SMs of the current device (cached). */
int sm_count();

}  // namespace neighbors::certex::detail
}  // namespace CUVS_EXPORT cuvs
