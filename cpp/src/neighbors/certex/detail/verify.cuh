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

// Ordered verification in rounds over all queries of a batch.
//
// The survivors of every query arrive sorted by certified lower bound. Round r takes, for every
// query that is not done, the next w_r = 32 * 2^r survivors whose lower bound is at most the
// query's current k-th exact score, scores all these (query, survivor) pairs in one flat launch
// over the GPU, and merges each query's scores into its top k. A query is done when its window
// ends at a lower bound above its k-th score or at its last survivor. Every survivor that the
// one-warp-per-query verification (ordered_verify) scores is scored here as well, because the
// k-th score at the start of a round is never below the k-th score that the sequential order
// would use, so that the result is the same exact top k. The flat launch spreads the long survivor
// lists of the hardest queries over the whole GPU, which matters when the rows are read from
// pinned host memory over PCIe.

#pragma once

#include "certex_kernels.cuh"

#include <cub/device/device_segmented_sort.cuh>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <cstdint>

namespace cuvs::neighbors::certex::detail {

constexpr int kPairsPerWarp = 8;  // pairs that one warp scores at once (row reads in flight)

/** Exact scores of G (query, row) pairs (xs[g] < 0: none), accumulated by one warp in fp64. */
template <typename T, score_kind S, int G>
__device__ void warp_pair_scores(const float* const (&qs)[G],
                                 const T* vectors,
                                 int64_t dim,
                                 const int32_t (&xs)[G],
                                 int lane,
                                 double (&out)[G])
{
  double acc[G], qq[G], xx[G];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    acc[g] = 0;
    qq[g]  = 0;
    xx[g]  = 0;
  }
  for (int64_t j = lane; j < dim; j += 32) {
    T raw[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {  // all row loads first, so that G reads are in flight
      raw[g] = xs[g] >= 0 ? vectors[int64_t(xs[g]) * dim + j] : T(0);
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const double qv = xs[g] >= 0 ? qs[g][j] : 0.0;
      const double xv = to_f(raw[g]);
      if constexpr (S == score_kind::l2) {
        acc[g] += (qv - xv) * (qv - xv);
      } else {
        acc[g] += qv * xv;
        qq[g] += qv * qv;
        if constexpr (S == score_kind::cosine) { xx[g] += xv * xv; }
      }
    }
  }
  for (int off = 16; off; off >>= 1) {
#pragma unroll
    for (int g = 0; g < G; ++g) {
      acc[g] += __shfl_xor_sync(0xffffffffu, acc[g], off);
      if constexpr (S != score_kind::l2) { qq[g] += __shfl_xor_sync(0xffffffffu, qq[g], off); }
      if constexpr (S == score_kind::cosine) { xx[g] += __shfl_xor_sync(0xffffffffu, xx[g], off); }
    }
  }
#pragma unroll
  for (int g = 0; g < G; ++g) {
    if (xs[g] < 0) {
      out[g] = INFINITY;
    } else if constexpr (S == score_kind::l2) {
      out[g] = acc[g];
    } else if constexpr (S == score_kind::inner) {
      out[g] = qq[g] - 2.0 * acc[g];
    } else {
      out[g] = 2.0 - 2.0 * acc[g] / sqrt(fmax(qq[g] * xx[g], 1e-300));
    }
  }
}

/** Initial state: every query at its first survivor, an empty top k, done without survivors. */
__global__ void rounds_init(const int64_t* start,
                            int n_queries,
                            int k,
                            int64_t* pos,
                            uint8_t* done,
                            double* best_score,
                            int64_t* best_id)
{
  const int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t < n_queries) {
    pos[t]  = start[t];
    done[t] = start[t] == start[t + 1];
  }
  if (t < int64_t(n_queries) * k) {
    best_score[t] = INFINITY;
    best_id[t]    = -1;
  }
}

/**
 * Pairs of every query in this round: the survivors in [pos, pos + w) whose lower bound is at most
 * the k-th score (binary search, the bounds are sorted). A query whose window stops short of w, or
 * reaches its last survivor, is done after this round. count[n_queries] = 0 for the scan.
 */
__global__ void rounds_count(const uint64_t* keys,
                             const int64_t* start,
                             const int64_t* pos,
                             const double* best_score,
                             int n_queries,
                             int k,
                             int64_t w,
                             uint8_t* done,
                             int64_t* count,
                             unsigned long long* verified)
{
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > n_queries) { return; }
  if (q == n_queries) {
    count[q] = 0;
    return;
  }
  int64_t c = 0;
  if (!done[q]) {
    const double kth = best_score[int64_t(q) * k + k - 1];
    const int64_t b = pos[q], end = start[q + 1];
    const int64_t e = b + w < end ? b + w : end;
    int64_t lo = b, hi = e;  // first survivor in [b, e) with a lower bound above kth
    while (lo < hi) {
      const int64_t mid = (lo + hi) / 2;
      if (double(key_float(uint32_t(keys[mid]))) <= kth) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    c = lo - b;
    if (lo < e || e == end) { done[q] = 1; }
    if (verified && c) { atomicAdd(verified, (unsigned long long)c); }
  }
  count[q] = c;
}

/** Exact scores of all pairs of the round, kPairsPerWarp consecutive pairs per warp. */
template <typename T, score_kind S>
__global__ void rounds_score(const int64_t* offset,
                             const int64_t* pos,
                             const int32_t* ids,
                             const float* queries,
                             const T* vectors,
                             int64_t dim,
                             int n_queries,
                             const uint8_t* removed,
                             Filter filter,
                             double* scores)
{
  const int64_t total = offset[n_queries];
  const int lane      = threadIdx.x & 31;
  const int64_t warp  = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  const int64_t warps = int64_t(gridDim.x) * blockDim.x / 32;
  for (int64_t p0 = warp * kPairsPerWarp; p0 < total; p0 += warps * kPairsPerWarp) {
    const float* qs[kPairsPerWarp];
    int32_t xs[kPairsPerWarp];
#pragma unroll
    for (int g = 0; g < kPairsPerWarp; ++g) {
      const int64_t p = p0 + g;
      qs[g]           = queries;
      xs[g]           = -1;
      if (p < total) {
        int lo = 0, hi = n_queries;  // the query of pair p: last q with offset[q] <= p
        while (hi - lo > 1) {
          const int mid = (lo + hi) / 2;
          if (offset[mid] <= p) {
            lo = mid;
          } else {
            hi = mid;
          }
        }
        const int32_t x = ids[pos[lo] + (p - offset[lo])];
        qs[g]           = queries + int64_t(lo) * dim;
        // a removed row, or one that the filter rejects, scores +inf and never enters
        xs[g] = removed[x] || !filter.pass(lo, x) ? -1 : x;
      }
    }
    double v[kPairsPerWarp];
    warp_pair_scores<T, S, kPairsPerWarp>(qs, vectors, dim, xs, lane, v);
    if (lane < kPairsPerWarp && p0 + lane < total) {
#pragma unroll
      for (int g = 0; g < kPairsPerWarp; ++g) {
        if (g == lane) { scores[p0 + g] = v[g]; }
      }
    }
  }
}

/** Merge of every query's scores of the round into its top k, one warp per query. */
template <int KPL>
__global__ void rounds_merge(const int64_t* offset,
                             const int32_t* ids,
                             const double* scores,
                             int n_queries,
                             int k,
                             int64_t* pos,
                             double* best_score,
                             int64_t* best_id)
{
  const int q    = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x & 31;
  if (q >= n_queries) { return; }
  const int64_t b = offset[q], n = offset[q + 1] - b;
  if (n == 0) { return; }
  const int64_t p = pos[q];
  double best[KPL];
  int64_t bid[KPL];
#pragma unroll
  for (int t = 0; t < KPL; ++t) {
    const int slot = t * 32 + lane;
    best[t]        = slot < k ? best_score[int64_t(q) * k + slot] : INFINITY;
    bid[t]         = slot < k ? best_id[int64_t(q) * k + slot] : -1;
  }
  const int kt = (k - 1) / 32, kl = (k - 1) % 32;  // the slot of the k-th score
  auto kth = [&]() {
    double v = INFINITY;
#pragma unroll
    for (int t = 0; t < KPL; ++t) {
      if (t == kt) { v = best[t]; }
    }
    return __shfl_sync(0xffffffffu, v, kl);
  };
  for (int64_t c = 0; c < n; c += 32) {
    const bool valid = c + lane < n;
    const double s   = valid ? scores[b + c + lane] : INFINITY;
    const int32_t x  = valid ? ids[p + c + lane] : -1;
    unsigned enter   = __ballot_sync(0xffffffffu, s < kth());
    while (enter) {
      const int src = __ffs(enter) - 1;
      enter &= enter - 1;
      const double v   = __shfl_sync(0xffffffffu, s, src);
      const int32_t xs = __shfl_sync(0xffffffffu, x, src);
      if (!(v < kth())) { continue; }
      int at = 0;
#pragma unroll
      for (int t = 0; t < KPL; ++t) {
        at += __popc(__ballot_sync(0xffffffffu, t * 32 + lane < k && best[t] <= v));
      }
#pragma unroll
      for (int t = KPL - 1; t >= 0; --t) {  // shift the slots above `at` up by one, top entry first
        const double up     = __shfl_up_sync(0xffffffffu, best[t], 1);
        const int64_t up_id = __shfl_up_sync(0xffffffffu, bid[t], 1);
        double carry        = INFINITY;
        int64_t carry_id    = -1;
        if (t > 0) {
          carry    = __shfl_sync(0xffffffffu, best[t > 0 ? t - 1 : 0], 31);
          carry_id = __shfl_sync(0xffffffffu, bid[t > 0 ? t - 1 : 0], 31);
        }
        const int slot = t * 32 + lane;
        if (slot < k) {
          if (slot == at) {
            best[t] = v;
            bid[t]  = xs;
          } else if (slot > at) {
            best[t] = lane == 0 ? carry : up;
            bid[t]  = lane == 0 ? carry_id : up_id;
          }
        }
      }
    }
  }
#pragma unroll
  for (int t = 0; t < KPL; ++t) {
    const int slot = t * 32 + lane;
    if (slot < k) {
      best_score[int64_t(q) * k + slot] = best[t];
      best_id[int64_t(q) * k + slot]    = bid[t];
    }
  }
  if (lane == 0) { pos[q] = p + n; }
}

/**
 * k above 1024 (KPL > 32): instead of a warp merge in registers, every query's top k and the pairs
 * of this round form one segment, which a segmented sort by exact score orders; the first k entries
 * become the new top k. A pair with an infinite score (removed or filtered) keeps the id -1, as the
 * warp merge never admits it.
 */
__global__ void rounds_gather(const int64_t* offset,
                              const int32_t* ids,
                              const double* scores,
                              int n_queries,
                              int k,
                              const int64_t* pos,
                              const double* best_score,
                              const int64_t* best_id,
                              double* seg_score,
                              int64_t* seg_id,
                              int64_t* seg_begin,
                              int64_t* seg_end)
{
  const int q = blockIdx.x;
  if (q >= n_queries) { return; }
  const int64_t b = offset[q], n = offset[q + 1] - b, p = pos[q];
  const int64_t base = int64_t(q) * k + b;
  if (threadIdx.x == 0) {
    seg_begin[q] = base;
    seg_end[q]   = base + k + n;
  }
  for (int64_t i = threadIdx.x; i < k + n; i += blockDim.x) {
    if (i < k) {
      seg_score[base + i] = best_score[int64_t(q) * k + i];
      seg_id[base + i]    = best_id[int64_t(q) * k + i];
    } else {
      const double v      = scores[b + i - k];
      seg_score[base + i] = v;
      seg_id[base + i]    = v < INFINITY ? int64_t(ids[p + i - k]) : int64_t(-1);
    }
  }
}

/** The first k entries of every sorted segment become the top k; the query advances past this
 * round's pairs. */
__global__ void rounds_scatter(const int64_t* offset,
                               const int64_t* seg_begin,
                               const double* sorted_score,
                               const int64_t* sorted_id,
                               int n_queries,
                               int k,
                               int64_t* pos,
                               double* best_score,
                               int64_t* best_id)
{
  const int q = blockIdx.x;
  if (q >= n_queries) { return; }
  const int64_t s = seg_begin[q];
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    best_score[int64_t(q) * k + i] = sorted_score[s + i];
    best_id[int64_t(q) * k + i]    = sorted_id[s + i];
  }
  if (threadIdx.x == 0) { pos[q] += offset[q + 1] - offset[q]; }
}

/**
 * Ordered verification of a batch in rounds (see the top of this file): the exact top k of every
 * query into best_score and best_id [n_queries, k], ascending; verified (if not null) counts the
 * scored pairs. One host synchronization per round reads the number of pairs.
 */
template <typename T, score_kind S, int KPL>
void verify_rounds(raft::resources const& res,
                   const uint64_t* keys,
                   const int32_t* ids,
                   const int64_t* start,
                   const float* queries,
                   const T* vectors,
                   int64_t dim,
                   int n_queries,
                   int k,
                   const uint8_t* removed,
                   double* best_score,
                   int64_t* best_id,
                   unsigned long long* verified,
                   Filter filter = {})
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  const int B         = n_queries;
  rmm::device_uvector<int64_t> pos(B, stream), count(B + 1, stream), offset(B + 1, stream);
  rmm::device_uvector<uint8_t> done(B, stream);
  const auto blocks = [](int64_t n) { return unsigned((n + 255) / 256); };
  rounds_init<<<blocks(std::max<int64_t>(B + 1, int64_t(B) * k)), 256, 0, stream>>>(
    start, B, k, pos.data(), done.data(), best_score, best_id);
  RAFT_CUDA_TRY(cudaGetLastError());
  size_t tmp_bytes = 0;
  RAFT_CUDA_TRY(
    cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes, count.data(), offset.data(), B + 1, stream));
  rmm::device_buffer tmp(tmp_bytes, stream);
  int device = 0, sms = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&device));
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  rmm::device_uvector<double> scores(0, stream);
  rmm::device_uvector<double> seg_score(0, stream), seg_sorted(0, stream);
  rmm::device_uvector<int64_t> seg_id(0, stream), seg_id_sorted(0, stream);
  rmm::device_uvector<int64_t> seg_begin(KPL > 32 ? B : 0, stream),
    seg_end(KPL > 32 ? B : 0, stream);
  rmm::device_buffer seg_tmp(0, stream);
  // large k starts with a window of k: the top k must hold k scored pairs before the windows can
  // prune
  for (int64_t w = KPL > 32 ? std::max<int64_t>(32, k) : 32;; w *= 2) {
    rounds_count<<<blocks(B + 1), 256, 0, stream>>>(
      keys, start, pos.data(), best_score, B, k, w, done.data(), count.data(), verified);
    RAFT_CUDA_TRY(cudaGetLastError());
    RAFT_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
      tmp.data(), tmp_bytes, count.data(), offset.data(), B + 1, stream));
    int64_t total = 0;
    raft::copy(&total, offset.data() + B, 1, stream);
    raft::resource::sync_stream(res);
    if (total == 0) { break; }
    if (int64_t(scores.size()) < total) { scores.resize(total, stream); }
    const int64_t warps = (total + kPairsPerWarp - 1) / kPairsPerWarp;
    const unsigned grid = unsigned(std::min<int64_t>((warps * 32 + 255) / 256, int64_t(sms) * 16));
    rounds_score<T, S><<<grid, 256, 0, stream>>>(
      offset.data(), pos.data(), ids, queries, vectors, dim, B, removed, filter, scores.data());
    RAFT_CUDA_TRY(cudaGetLastError());
    if constexpr (KPL > 32) {
      const int64_t items = int64_t(B) * k + total;
      if (int64_t(seg_score.size()) < items) {
        seg_score.resize(items, stream);
        seg_sorted.resize(items, stream);
        seg_id.resize(items, stream);
        seg_id_sorted.resize(items, stream);
      }
      rounds_gather<<<B, 256, 0, stream>>>(offset.data(),
                                           ids,
                                           scores.data(),
                                           B,
                                           k,
                                           pos.data(),
                                           best_score,
                                           best_id,
                                           seg_score.data(),
                                           seg_id.data(),
                                           seg_begin.data(),
                                           seg_end.data());
      RAFT_CUDA_TRY(cudaGetLastError());
      size_t sort_bytes = 0;
      RAFT_CUDA_TRY(cub::DeviceSegmentedSort::SortPairs(nullptr,
                                                        sort_bytes,
                                                        seg_score.data(),
                                                        seg_sorted.data(),
                                                        seg_id.data(),
                                                        seg_id_sorted.data(),
                                                        items,
                                                        B,
                                                        seg_begin.data(),
                                                        seg_end.data(),
                                                        stream));
      if (seg_tmp.size() < sort_bytes) { seg_tmp.resize(sort_bytes, stream); }
      RAFT_CUDA_TRY(cub::DeviceSegmentedSort::SortPairs(seg_tmp.data(),
                                                        sort_bytes,
                                                        seg_score.data(),
                                                        seg_sorted.data(),
                                                        seg_id.data(),
                                                        seg_id_sorted.data(),
                                                        items,
                                                        B,
                                                        seg_begin.data(),
                                                        seg_end.data(),
                                                        stream));
      rounds_scatter<<<B, 256, 0, stream>>>(offset.data(),
                                            seg_begin.data(),
                                            seg_sorted.data(),
                                            seg_id_sorted.data(),
                                            B,
                                            k,
                                            pos.data(),
                                            best_score,
                                            best_id);
      RAFT_CUDA_TRY(cudaGetLastError());
    } else {
      rounds_merge<KPL><<<blocks(int64_t(B) * 32), 256, 0, stream>>>(
        offset.data(), ids, scores.data(), B, k, pos.data(), best_score, best_id);
      RAFT_CUDA_TRY(cudaGetLastError());
    }
  }
}

}  // namespace cuvs::neighbors::certex::detail
