/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// k-means lists of a CertEx index (index_params::n_lists) and the threshold that they provide.
//
// A build with lists trains k-means centers on a sample, assigns every vector to its nearest
// center, and stores the vectors in list order, so that every list occupies a contiguous block of
// rows; row_ids maps a row to the id of its vector. A search then screens, for every query, the
// lists nearest to it under the anchor threshold, verifies their survivors in order, and lowers
// the threshold to the k-th exact score found. Any k distinct vectors bound the k-th score from
// above, so that the lists change the speed of a search and never its result.
//
// This header belongs to certex.cu, which includes it after the helpers that it uses (sources,
// rows_to_float, project, blocks_for).

#include <cuvs/cluster/kmeans.hpp>

#include <algorithm>
#include <cstring>
#include <vector>

namespace cuvs::neighbors::certex::detail {

/** dst[i] = src[order[i]] for rows of d elements (one thread per element). */
template <typename T>
__global__ void gather_rows(const T* src, const int32_t* order, int64_t n, int64_t d, T* dst)
{
  for (int64_t t = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; t < n * d;
       t += int64_t(gridDim.x) * blockDim.x) {
    const int64_t i = t / d, j = t % d;
    dst[t] = src[int64_t(order[i]) * d + j];
  }
}

/** The rows of a device source in the order of a permutation: row i is source row perm[i]. */
template <typename T>
struct permuted_device_source {
  device_source<T> base;
  const int32_t* perm;  // device
  int64_t n, d;
  rmm::device_uvector<T> staging{0, cudaStream_t{}};  // any rmm: a raw stream converts
  auto rows(raft::resources const& res, int64_t i, int64_t cn) -> const T*
  {
    cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
    staging.resize(cn * d, stream);
    gather_rows<T><<<blocks_for(std::min<int64_t>(cn * d, int64_t(1) << 24)), 256, 0, stream>>>(
      base.x, perm + i, cn, d, staging.data());
    RAFT_CUDA_TRY(cudaGetLastError());
    return staging.data();
  }
  auto strided(raft::resources const& res, int64_t step, int64_t j0, int64_t cnt) -> const T*
  {
    return base.strided(res, step, j0, cnt);  // a sample: the order does not matter
  }
};

/** The rows of a host source in the order of a permutation, gathered on the host. */
template <typename T>
struct permuted_host_source {
  host_source<T> base;
  std::vector<int32_t> perm;  // host
  int64_t n, d;
  std::vector<T> packed;
  rmm::device_uvector<T> staging{0, cudaStream_t{}};  // any rmm: a raw stream converts
  auto rows(raft::resources const& res, int64_t i, int64_t cn) -> const T*
  {
    packed.resize(cn * d);
    for (int64_t j = 0; j < cn; ++j) {
      std::memcpy(packed.data() + j * d, base.x + int64_t(perm[i + j]) * d, d * sizeof(T));
    }
    cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
    staging.resize(cn * d, stream);
    raft::copy(staging.data(), packed.data(), cn * d, stream);
    raft::resource::sync_stream(res);  // packed is reused by the next chunk
    return staging.data();
  }
  auto strided(raft::resources const& res, int64_t step, int64_t j0, int64_t cnt) -> const T*
  {
    return base.strided(res, step, j0, cnt);
  }
};

template <typename T>
auto permuted(raft::resources const&, device_source<T>& src, const int32_t* perm)
  -> permuted_device_source<T>
{
  return permuted_device_source<T>{device_source<T>{src.x, src.n, src.d}, perm, src.n, src.d};
}

template <typename T>
auto permuted(raft::resources const& res, host_source<T>& src, const int32_t* perm)
  -> permuted_host_source<T>
{
  permuted_host_source<T> p{
    host_source<T>{src.x, src.n, src.d}, std::vector<int32_t>(src.n), src.n, src.d};
  raft::copy(p.perm.data(), perm, src.n, raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  return p;
}

/** First position of every list j <= n_lists in labels sorted ascending (binary search). */
__global__ void list_starts(const int32_t* sorted, int64_t n, int n_lists, int64_t* start)
{
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j > n_lists) { return; }
  int64_t lo = 0, hi = n;
  while (lo < hi) {
    const int64_t mid = (lo + hi) / 2;
    if (sorted[mid] < j) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  start[j] = lo;
}

__global__ void invert_permutation(const int32_t* perm, int64_t n, int32_t* inverse)
{
  const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (i < n) { inverse[perm[i]] = int32_t(i); }
}

/** k-means lists of a source: centers [P, d] and the order of the vectors by list. */
struct lists_layout {
  raft::device_matrix<float, int64_t, raft::row_major> centers;
  raft::device_vector<int32_t, int64_t> row_ids;  // row -> id: the vectors in list order
  raft::device_vector<int32_t, int64_t> id_rows;  // id -> row
  std::vector<int64_t> offsets;                   // [P + 1]
};

template <typename T, typename Src>
auto fit_lists(raft::resources const& res, index_params const& params, Src& src) -> lists_layout
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  const int64_t n = src.n, d = src.d;
  const int P = params.n_lists;
  RAFT_EXPECTS(P >= 1 && P <= n, "n_lists must lie in [1, number of vectors]");
  RAFT_EXPECTS(n < (int64_t(1) << 31), "at most 2^31 - 1 vectors in an index with lists");
  const bool cosine = params.metric == DistanceType::CosineExpanded;
  cuvs::cluster::kmeans::balanced_params kp;  // L2 lists: they change only the speed
  // centers from a strided sample
  const int64_t s    = std::max<int64_t>(P, std::min<int64_t>(n, params.kmeans_sample));
  const int64_t step = std::max<int64_t>(1, n / s);
  const int64_t cnt  = std::min<int64_t>(s, (n + step - 1) / step);
  auto sample        = raft::make_device_matrix<float, int64_t>(res, cnt, d);
  const T* srows     = src.strided(res, step, 0, cnt);
  rows_to_float<T>
    <<<blocks_for(cnt * 32), 256, 0, stream>>>(srows, d, cnt, cosine, sample.data_handle());
  RAFT_CUDA_TRY(cudaGetLastError());
  lists_layout out{raft::make_device_matrix<float, int64_t>(res, P, d),
                   raft::make_device_vector<int32_t, int64_t>(res, n),
                   raft::make_device_vector<int32_t, int64_t>(res, n),
                   std::vector<int64_t>(P + 1)};
  cuvs::cluster::kmeans::fit(res, kp, raft::make_const_mdspan(sample.view()), out.centers.view());
  // the list of every vector, in chunks
  rmm::device_uvector<int32_t> labels(n, stream), labels_sorted(n, stream), ids(n, stream);
  const int64_t blk = std::max<int64_t>(1, std::min<int64_t>(1 << 20, (int64_t(1) << 27) / d));
  auto f            = raft::make_device_matrix<float, int64_t>(res, std::min(blk, n), d);
  for (int64_t i = 0; i < n; i += blk) {
    const int64_t cn = std::min(blk, n - i);
    const T* rows    = src.rows(res, i, cn);
    rows_to_float<T><<<blocks_for(cn * 32), 256, 0, stream>>>(rows, d, cn, cosine, f.data_handle());
    RAFT_CUDA_TRY(cudaGetLastError());
    cuvs::cluster::kmeans::predict(
      res,
      kp,
      raft::make_device_matrix_view<const float, int64_t>(f.data_handle(), cn, d),
      raft::make_const_mdspan(out.centers.view()),
      raft::make_device_vector_view<int, int64_t>(labels.data() + i, cn));
    raft::resource::sync_stream(res);  // a host source reuses its staging buffer
  }
  // a stable sort of the ids by list gives the rows in list order
  iota_from<<<blocks_for(n), 256, 0, stream>>>(ids.data(), n);
  RAFT_CUDA_TRY(cudaGetLastError());
  size_t tmp_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(nullptr,
                                                tmp_bytes,
                                                labels.data(),
                                                labels_sorted.data(),
                                                ids.data(),
                                                out.row_ids.data_handle(),
                                                n,
                                                0,
                                                32,
                                                stream));
  rmm::device_buffer tmp(tmp_bytes, stream);
  RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(tmp.data(),
                                                tmp_bytes,
                                                labels.data(),
                                                labels_sorted.data(),
                                                ids.data(),
                                                out.row_ids.data_handle(),
                                                n,
                                                0,
                                                32,
                                                stream));
  rmm::device_uvector<int64_t> starts(P + 1, stream);
  list_starts<<<blocks_for(P + 1), 256, 0, stream>>>(labels_sorted.data(), n, P, starts.data());
  invert_permutation<<<blocks_for(n), 256, 0, stream>>>(
    out.row_ids.data_handle(), n, out.id_rows.data_handle());
  RAFT_CUDA_TRY(cudaGetLastError());
  raft::copy(out.offsets.data(), starts.data(), P + 1, stream);
  raft::resource::sync_stream(res);
  return out;
}

__global__ void square_values(double* v, int64_t n)
{
  const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (i < n) { v[i] *= v[i]; }
}

/** The p nearest lists of every query (one warp per query): dist = |c|^2 - 2 <q, c>. */
__global__ void nearest_lists(
  const float* dots, const double* center_sq, int P, int64_t B, int p, int32_t* out)
{
  const int64_t q = (blockIdx.x * int64_t(blockDim.x) + threadIdx.x) / 32;
  const int lane  = threadIdx.x & 31;
  if (q >= B) { return; }
  int32_t taken[8];
  for (int r = 0; r < p; ++r) {
    double best = INFINITY;
    int bj      = -1;
    for (int j = lane; j < P; j += 32) {
      bool used = false;
      for (int t = 0; t < r; ++t) {
        used |= taken[t] == j;
      }
      const double v = center_sq[j] - 2.0 * double(dots[q * P + j]);
      if (!used && (v < best || (v == best && j < bj))) {
        best = v;
        bj   = j;
      }
    }
    for (int off = 16; off; off >>= 1) {
      const double ov = __shfl_xor_sync(0xffffffffu, best, off);
      const int oj    = __shfl_xor_sync(0xffffffffu, bj, off);
      if (ov < best || (ov == best && oj >= 0 && (bj < 0 || oj < bj))) {
        best = ov;
        bj   = oj;
      }
    }
    taken[r] = bj;
    if (lane == 0) { out[q * p + r] = bj; }
  }
}

/** Per-query state of a search batch that the probe reads and tightens. */
struct batch_state {
  int64_t B;
  int k;
  int row_len, L;
  const half* q_rows;    // [B, row_len]
  const double* aqh;     // [B]
  const float* margin;   // [B]
  const double* q_norm;  // [B]
  float* theta;          // [B], lowered by the probe
  double* u;             // [B], the threshold U of every query, lowered by the probe
  double gamma, m_nu, m_high, alpha, slack, slack_ip;
};

/** Lower U and theta of every query to the k-th exact score among its probed survivors. */
template <score_kind S>
__global__ void tighten_threshold(const double* best, int64_t B, int k, batch_state st)
{
  const int64_t q = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (q >= B) { return; }
  const double kth = best[q * k + k - 1];
  if (!isfinite(kth)) { return; }
  const double u1 = S == score_kind::inner ? kth + 2 * kTol * st.q_norm[q] * st.slack_ip
                                           : kth * (1 + kTol) + st.slack;
  if (u1 < st.u[q]) {
    st.u[q] = u1;
    st.theta[q] =
      __double2float_rd((st.aqh[q] - u1 / (st.alpha * st.alpha)) / 2 - double(st.margin[q]));
  }
}

/** Gather the probe rows: the query row, aqh, margin, and threshold of every (query, list) pair. */
__global__ void gather_probe(const int32_t* order,
                             int64_t n,
                             int row_len,
                             batch_state st,
                             half* rows,
                             double* aqh,
                             float* margin,
                             float* theta)
{
  const int64_t t = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (t >= n * row_len) { return; }
  const int64_t i = t / row_len, j = t % row_len, q = order[i];
  rows[t] = st.q_rows[q * row_len + j];
  if (j == 0) {
    aqh[i]    = st.aqh[q];
    margin[i] = st.margin[q];
    theta[i]  = st.theta[q];
  }
}

/**
 * Per-list query rows of local projections (local.cuh), for a probe: row j * B + q encodes query q
 * in the frame of list j; consts holds the alpha, M_nu, and M_high of every list ([3 P]).
 */
struct local_probe {
  const half* q_rows;    // [P * B, row_len]
  const double* aqh;     // [P * B]
  const float* margin;   // [P * B]
  const float* theta;    // [P * B]
  const double* consts;  // [3 P]
};

/** Gather the probe rows of local projections: pair i of query order[i] and list list_of[i]. */
__global__ void gather_probe_local(const int32_t* order,
                                   const int32_t* list_of,
                                   int64_t n,
                                   int64_t B,
                                   int P,
                                   int row_len,
                                   local_probe lp,
                                   half* rows,
                                   double* aqh,
                                   float* margin,
                                   float* theta,
                                   double* consts)
{
  const int64_t t = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (t >= n * row_len) { return; }
  const int64_t i = t / row_len, c = t % row_len, j = list_of[i], r = j * B + order[i];
  rows[t] = lp.q_rows[r * row_len + c];
  if (c == 0) {
    aqh[i]            = lp.aqh[r];
    margin[i]         = lp.margin[r];
    theta[i]          = lp.theta[r];
    consts[i]         = lp.consts[j];
    consts[n + i]     = lp.consts[P + j];
    consts[2 * n + i] = lp.consts[2 * P + j];
  }
}

/**
 * The probe of a batch: every query meets the rows of its p nearest lists in one routed pass under
 * its current threshold; the exact scores of its 4k survivors of smallest certified lower bound
 * give k distinct vectors, whose k-th score lowers U and theta. qf holds the queries as lifted
 * (normalized for cosine), q_exact as given.
 */
template <typename T, score_kind S>
void probe_lists(raft::resources const& res,
                 index<T> const& idx,
                 int p,
                 const float* qf,
                 const float* q_exact,
                 batch_state st,
                 const local_probe* lp = nullptr,
                 Filter filter         = {})
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  const int P         = idx.n_lists();
  const int64_t B = st.B, d = idx.dim();
  const int k = st.k, MP = st.row_len;
  p = std::min(std::min(p, 8), P);
  // nearest lists
  rmm::device_uvector<float> dots(B * P, stream);
  rmm::device_uvector<double> csq(P, stream);
  project(res, qf, B, d, idx.list_centers().data_handle(), P, dots.data());
  row_norms<<<blocks_for(int64_t(P) * 32), 256, 0, stream>>>(
    idx.list_centers().data_handle(), d, P, csq.data());
  square_values<<<blocks_for(P), 256, 0, stream>>>(csq.data(), P);
  rmm::device_uvector<int32_t> near(B * p, stream);
  nearest_lists<<<blocks_for(B * 32), 256, 0, stream>>>(
    dots.data(), csq.data(), P, B, p, near.data());
  RAFT_CUDA_TRY(cudaGetLastError());
  std::vector<int32_t> h_near(B * p);
  raft::copy(h_near.data(), near.data(), B * p, stream);
  raft::resource::sync_stream(res);
  // the (query, list) pairs grouped by list: one routed group per list
  const auto& off = idx.list_offsets();
  std::vector<int64_t> count(P + 1, 0);
  for (auto j : h_near) {
    if (j >= 0) { ++count[j + 1]; }
  }
  for (int j = 0; j < P; ++j) {
    count[j + 1] += count[j];
  }
  const int64_t n_pairs = count[P];
  std::vector<int32_t> order(std::max<int64_t>(1, n_pairs)), list_of(std::max<int64_t>(1, n_pairs));
  {
    std::vector<int64_t> fill(count.begin(), count.end() - 1);
    for (int64_t q = 0; q < B; ++q) {
      for (int r = 0; r < p; ++r) {
        const int32_t j = h_near[q * p + r];
        if (j >= 0) {
          list_of[fill[j]] = j;
          order[fill[j]++] = int32_t(q);
        }
      }
    }
  }
  std::vector<route_group> groups;
  int tiles = 0;
  for (int j = 0; j < P; ++j) {
    const int64_t nq = count[j + 1] - count[j], nx = off[j + 1] - off[j];
    if (nq == 0 || nx == 0) { continue; }
    route_group g;
    g.q0      = int32_t(count[j]);
    g.nq      = int32_t(nq);
    g.x0      = int32_t(off[j]);
    g.nx      = int32_t(nx);
    g.tile0   = tiles;
    g.x_tiles = int32_t((nx + kTileN - 1) / kTileN);
    tiles += int((nq + kTileM - 1) / kTileM) * g.x_tiles;
    groups.push_back(g);
  }
  if (groups.empty()) { return; }
  rmm::device_uvector<int32_t> d_order(n_pairs, stream);
  rmm::device_uvector<route_group> d_groups(groups.size(), stream);
  raft::copy(d_order.data(), order.data(), n_pairs, stream);
  RAFT_CUDA_TRY(cudaMemcpyAsync(d_groups.data(),
                                groups.data(),
                                groups.size() * sizeof(route_group),
                                cudaMemcpyHostToDevice,
                                stream));
  rmm::device_uvector<half> rows(n_pairs * MP, stream);
  rmm::device_uvector<double> aqh(n_pairs, stream);
  rmm::device_uvector<float> margin(n_pairs, stream), theta(n_pairs, stream);
  rmm::device_uvector<double> pair_consts(lp ? 3 * n_pairs : 0,
                                          stream);  // local: alpha, M_nu, M_high
  if (lp) {
    rmm::device_uvector<int32_t> d_list(n_pairs, stream);
    raft::copy(d_list.data(), list_of.data(), n_pairs, stream);
    gather_probe_local<<<blocks_for(n_pairs * MP), 256, 0, stream>>>(d_order.data(),
                                                                     d_list.data(),
                                                                     n_pairs,
                                                                     B,
                                                                     P,
                                                                     MP,
                                                                     *lp,
                                                                     rows.data(),
                                                                     aqh.data(),
                                                                     margin.data(),
                                                                     theta.data(),
                                                                     pair_consts.data());
    RAFT_CUDA_TRY(cudaGetLastError());
    raft::resource::sync_stream(res);  // d_list leaves scope
  } else {
    gather_probe<<<blocks_for(n_pairs * MP), 256, 0, stream>>>(
      d_order.data(), n_pairs, MP, st, rows.data(), aqh.data(), margin.data(), theta.data());
  }
  RAFT_CUDA_TRY(cudaGetLastError());
  // routed pass over the listed rows, into per-block segments
  const int64_t n_listed = off[P];
  const int nseg         = std::max(1, routed_segments(tiles));
  uint32_t seg_cap       = uint32_t(std::max<int64_t>(4096, n_pairs * 2000 / nseg));
  rmm::device_uvector<uint32_t> seg_count(nseg, stream), seg_offset(nseg, stream);
  rmm::device_uvector<int32_t> seg_q(0, stream), seg_x(0, stream);
  rmm::device_uvector<float> seg_v(0, stream);
  std::vector<uint32_t> h_count(nseg);
  for (;;) {
    seg_q.resize(int64_t(seg_cap) * nseg, stream);
    seg_x.resize(int64_t(seg_cap) * nseg, stream);
    seg_v.resize(int64_t(seg_cap) * nseg, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(seg_count.data(), 0, nseg * sizeof(uint32_t), stream));
    screen_routed(rows.data(),
                  int(n_pairs),
                  idx.rows().data_handle(),
                  n_listed,
                  MP,
                  theta.data(),
                  d_groups.data(),
                  int(groups.size()),
                  tiles,
                  seg_q.data(),
                  seg_x.data(),
                  seg_v.data(),
                  seg_count.data(),
                  seg_cap,
                  stream);
    raft::copy(h_count.data(), seg_count.data(), nseg, stream);
    raft::resource::sync_stream(res);
    const uint32_t worst = *std::max_element(h_count.begin(), h_count.end());
    if (worst <= seg_cap) { break; }
    if (int64_t(seg_cap) * nseg >= (int64_t(1) << 31)) {
      throw batch_too_large("too many survivors in a probe");
    }
    seg_cap = uint32_t(std::min<int64_t>(int64_t(worst) * 2, ((int64_t(1) << 31) - 1) / nseg));
  }
  std::vector<uint32_t> h_offset(nseg);
  int64_t total = 0;
  for (int i = 0; i < nseg; ++i) {
    h_offset[i] = uint32_t(total);
    total += h_count[i];
  }
  if (total == 0) { return; }
  raft::copy(seg_offset.data(), h_offset.data(), nseg, stream);
  // order the survivors of every query by their certified lower bounds, then verify in order
  rmm::device_uvector<uint64_t> keys(total, stream), keys_sorted(total, stream);
  rmm::device_uvector<int32_t> ids(total, stream), ids_sorted(total, stream);
  key_source probe{rows.data(), aqh.data(), margin.data(), st.alpha, st.m_nu, st.m_high};
  probe.q_map = d_order.data();
  if (lp) {  // per-pair constants of the pair's list: period 1 indexes them by pair
    probe.list_alpha  = pair_consts.data();
    probe.list_m_nu   = pair_consts.data() + n_pairs;
    probe.list_m_high = pair_consts.data() + 2 * n_pairs;
    probe.period      = 1;
  }
  survivor_keys<<<unsigned(nseg), 256, 0, stream>>>(seg_q.data(),
                                                    seg_x.data(),
                                                    seg_v.data(),
                                                    seg_count.data(),
                                                    seg_offset.data(),
                                                    seg_cap,
                                                    MP,
                                                    st.L,
                                                    st.gamma,
                                                    st.slack,
                                                    0,
                                                    probe,
                                                    probe,
                                                    keys.data(),
                                                    ids.data());
  RAFT_CUDA_TRY(cudaGetLastError());
  int q_bits = 1;
  while ((int64_t(1) << q_bits) < B) {
    ++q_bits;
  }
  size_t tmp_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(nullptr,
                                                tmp_bytes,
                                                keys.data(),
                                                keys_sorted.data(),
                                                ids.data(),
                                                ids_sorted.data(),
                                                total,
                                                0,
                                                32 + q_bits,
                                                stream));
  rmm::device_buffer tmp(tmp_bytes, stream);
  RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(tmp.data(),
                                                tmp_bytes,
                                                keys.data(),
                                                keys_sorted.data(),
                                                ids.data(),
                                                ids_sorted.data(),
                                                total,
                                                0,
                                                32 + q_bits,
                                                stream));
  rmm::device_uvector<int64_t> start(B + 1, stream);
  query_starts<<<blocks_for(B + 1), 256, 0, stream>>>(
    keys_sorted.data(), total, int(B), start.data());
  rmm::device_uvector<double> best(B * k, stream);
  rmm::device_uvector<int64_t> best_id(B * k, stream);
  auto verify = [&](auto kpl) {
    ordered_verify<T, S, decltype(kpl)::value>
      <<<blocks_for(B * 32), 256, 0, stream>>>(keys_sorted.data(),
                                               ids_sorted.data(),
                                               start.data(),
                                               q_exact,
                                               idx.vectors_ptr(),
                                               d,
                                               int(B),
                                               k,
                                               idx.removed().data_handle(),
                                               best.data(),
                                               best_id.data(),
                                               nullptr,
                                               int64_t(4) * k,  // as candidates: an upper bound
                                               filter);         // admitted vectors only
  };
  if (k <= 32) {
    verify(std::integral_constant<int, 1>{});
  } else if (k <= 64) {
    verify(std::integral_constant<int, 2>{});
  } else if (k <= 128) {
    verify(std::integral_constant<int, 4>{});
  } else if (k <= 256) {
    verify(std::integral_constant<int, 8>{});
  } else if (k <= 512) {
    verify(std::integral_constant<int, 16>{});
  } else if (k <= 1024) {
    verify(std::integral_constant<int, 32>{});
  } else {  // up to kMaxK: 64 slots per lane, which spill to local memory but keep the merge exact
    verify(std::integral_constant<int, 64>{});
  }
  tighten_threshold<S><<<blocks_for(B), 256, 0, stream>>>(best.data(), B, k, st);
  RAFT_CUDA_TRY(cudaGetLastError());
}

}  // namespace cuvs::neighbors::certex::detail
