/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Local projections of a CertEx index (index_params::local_projections).
//
// Every k-means list j gets its own center c_j (the mean of its vectors), its own principal
// directions V_j, its own scale alpha_j, and its own certificate maxima, and its rows are encoded
// with them. The lifted bound |l_j(q) - l_j(x)| <= |q - x| holds for every orthonormal V_j and
// every center, and the certificate of a row needs only the maxima of the rows that it is compared
// with, so that a query lifted with (c_j, V_j) and screened against the rows of list j under the
// common threshold U(q) keeps every true neighbor in list j. The anchors and the vectors added
// after the build keep the global projection.
//
// This header belongs to certex.cu, which includes it after the helpers that it uses.

#include <raft/linalg/eig.cuh>

#include <vector>

namespace cuvs::neighbors::certex::detail {

/** Rows [off, off + n) of a source. */
template <typename Src>
struct range_source {
  Src& base;
  int64_t off, n, d;
  auto rows(raft::resources const& res, int64_t i, int64_t cn)
  {
    return base.rows(res, off + i, cn);
  }
};

/** sum += rows (one thread per column); the Gram matrix follows by a GEMM. */
__global__ void add_columns(const double* y, int64_t n, int64_t d, double* sum)
{
  const int64_t c = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (c >= d) { return; }
  double s = 0;
  for (int64_t i = 0; i < n; ++i) {
    s += y[i * d + c];
  }
  sum[c] += s;
}

/** cov = gram / n - mean mean^T (column major), mean = sum / n. */
__global__ void covariance_from(
  const double* gram, const double* sum, int64_t n, int64_t d, double* cov, double* mean)
{
  const int64_t t = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (t >= d * d) { return; }
  const int64_t a = t % d, b = t / d;
  cov[t] = gram[t] / double(n) - (sum[a] / double(n)) * (sum[b] / double(n));
  if (b == 0) { mean[a] = sum[a] / double(n); }
}

/**
 * Center and principal directions of the rows of a source: the mean and the m leading eigenvectors
 * of the covariance, accumulated in double precision over chunks.
 */
template <typename T, typename Src>
void local_basis(raft::resources const& res,
                 Src& src,
                 int64_t d,
                 int m,
                 bool normalize,
                 bool centered,
                 float* center,
                 float* basis)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  const int64_t n     = src.n;
  const int64_t blk   = std::max<int64_t>(1, std::min<int64_t>(1 << 16, (int64_t(1) << 25) / d));
  rmm::device_uvector<float> f(std::min(blk, n) * d, stream);
  rmm::device_uvector<double> y(std::min(blk, n) * d, stream), sum(d, stream);
  auto gram = raft::make_device_matrix<double, int64_t, raft::col_major>(res, d, d);
  RAFT_CUDA_TRY(cudaMemsetAsync(sum.data(), 0, d * sizeof(double), stream));
  auto handle = raft::resource::get_cublas_handle(res);
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  for (int64_t i = 0; i < n; i += blk) {
    const int64_t cn = std::min(blk, n - i);
    const auto* rows = src.rows(res, i, cn);
    using R          = std::remove_cv_t<std::remove_pointer_t<decltype(rows)>>;
    rows_to_float<R><<<blocks_for(cn * 32), 256, 0, stream>>>(rows, d, cn, normalize, f.data());
    float_to_double<<<blocks_for(cn * d), 256, 0, stream>>>(f.data(), cn * d, y.data());
    add_columns<<<blocks_for(d), 256, 0, stream>>>(y.data(), cn, d, sum.data());
    RAFT_CUDA_TRY(cudaGetLastError());
    const double one = 1, beta = i == 0 ? 0.0 : 1.0;
    RAFT_CUBLAS_TRY(cublasDgemm(handle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_T,
                                int(d),
                                int(d),
                                int(cn),
                                &one,
                                y.data(),
                                int(d),
                                y.data(),
                                int(d),
                                &beta,
                                gram.data_handle(),
                                int(d)));
    raft::resource::sync_stream(res);  // a host source reuses its staging buffer
  }
  auto cov = raft::make_device_matrix<double, int64_t, raft::col_major>(res, d, d);
  rmm::device_uvector<double> mean(d, stream);
  if (centered) {
    covariance_from<<<blocks_for(d * d), 256, 0, stream>>>(
      gram.data_handle(), sum.data(), n, d, cov.data_handle(), mean.data());
  } else {  // inner products: the second moment around the origin
    raft::copy(cov.data_handle(), gram.data_handle(), d * d, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(mean.data(), 0, d * sizeof(double), stream));
  }
  RAFT_CUDA_TRY(cudaGetLastError());
  auto vecs = raft::make_device_matrix<double, int64_t, raft::col_major>(res, d, d);
  auto vals = raft::make_device_vector<double, int64_t>(res, d);
  {  // unsigned extents: eig_dc compares the vector size with an extent of its index type
    const auto D = uint32_t(d);
    raft::linalg::eig_dc(
      res,
      raft::make_device_matrix_view<const double, uint32_t, raft::col_major>(
        cov.data_handle(), D, D),
      raft::make_device_matrix_view<double, uint32_t, raft::col_major>(vecs.data_handle(), D, D),
      raft::make_device_vector_view<double, uint32_t>(vals.data_handle(), D));
  }

  double_to_float<<<blocks_for(d), 256, 0, stream>>>(mean.data(), d, center);
  top_eigenvectors<<<blocks_for(int64_t(m) * d), 256, 0, stream>>>(vecs.data_handle(), d, m, basis);
  RAFT_CUDA_TRY(cudaGetLastError());
}

/** Per-list constants of the query rows: 1 / alpha_j and the certificate maxima of list j. */
struct list_constants {
  float inv_alpha, m_norm, m_error, m_low;
};

/**
 * Query rows for local projections, one warp per row: row i belongs to list j0 + i / period and
 * is encoded as by encode_queries with the scale and maxima of that list.
 */
__global__ void encode_queries_lists(const float* z,
                                     const float* r,
                                     const float* a,
                                     int m,
                                     int64_t n,
                                     int64_t period,
                                     const list_constants* consts,
                                     int row_len,
                                     float one_minus_eps,
                                     float gamma,
                                     half* rows,
                                     double* aqh,
                                     float* margin)
{
  const int64_t row = (blockIdx.x * int64_t(blockDim.x) + threadIdx.x) / 32;
  const int lane    = threadIdx.x & 31;
  if (row >= n) { return; }
  const list_constants k = consts[row / period];
  const int L            = m + 1;
  half* out              = rows + row * row_len;
  float dq = 0, n16 = 0;
  for (int j = lane; j < row_len; j += 32) {
    if (j < L) {
      const float v  = (j < m ? z[row * m + j] : r[row]) * k.inv_alpha;
      const half h   = __float2half_rn(v);
      const float hv = __half2float(h);
      dq += (v - hv) * (v - hv);
      n16 += hv * hv;
      out[j] = h;
    } else if (j >= L + 4) {
      out[j] = __float2half_rn(0.0f);
    }
  }
  for (int off = 16; off; off >>= 1) {
    dq += __shfl_xor_sync(0xffffffffu, dq, off);
    n16 += __shfl_xor_sync(0xffffffffu, n16, off);
  }
  if (lane == 0) {
    const half c_err  = up16(sqrtf(dq));
    const half c_norm = up16(sqrtf(n16));
    out[L]            = c_err;
    out[L + 1]        = c_norm;
    out[L + 2]        = __float2half_rn(one_minus_eps);
    out[L + 3]        = __float2half_rn(1.0f);
    aqh[row]          = double(a[row]) * k.inv_alpha * k.inv_alpha;
    margin[row] =
      gamma * (__half2float(c_err) * k.m_norm + __half2float(c_norm) * k.m_error + k.m_low);
  }
}

/** out[(j, q), :] = q_rows[q, :] - centers[j, :] for lists j < nl and queries q < B. */
__global__ void center_per_list(
  const float* q, int64_t B, int64_t d, const float* centers, int64_t nl, float* out)
{
  for (int64_t t = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; t < nl * B * d;
       t += int64_t(gridDim.x) * blockDim.x) {
    const int64_t c = t % d, q_ = (t / d) % B, j = t / (B * d);
    out[t] = q[q_ * d + c] - centers[j * d + c];
  }
}

/** theta of query row i (list i / B, query i % B) from the threshold U of its query. */
__global__ void local_thresholds(const double* u,
                                 const double* aqh,
                                 const float* margin,
                                 const double* list_alpha,
                                 int64_t B,
                                 int64_t n,
                                 float* theta)
{
  const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (i >= n) { return; }
  const double al = list_alpha[i / B];
  theta[i]        = __double2float_rd((aqh[i] - u[i % B] / (al * al)) / 2 - double(margin[i]));
}

/** The query rows of a batch for every list: [P * B, row_len], with aqh and margin. */
template <typename T>
void encode_local_queries(raft::resources const& res,
                          index<T> const& idx,
                          const float* qf,
                          int64_t B,
                          half* rows,
                          double* aqh,
                          float* margin)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  const int P = idx.n_lists(), m = idx.width(), MP = idx.row_length();
  const int64_t d = idx.dim();
  const double g = gamma_of(MP), eps = eps_of(g);
  std::vector<list_constants> h(P);
  for (int j = 0; j < P; ++j) {
    const auto& M = idx.local_maxima()[j];
    h[j] = {float(1.0 / idx.local_alpha()[j]), float(M.norm), float(M.error), float(M.low)};
  }
  rmm::device_uvector<list_constants> consts(P, stream);
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    consts.data(), h.data(), P * sizeof(list_constants), cudaMemcpyHostToDevice, stream));
  // lists in chunks, so that the centered queries of a chunk stay within 2^26 floats
  const int64_t cl = std::max<int64_t>(1, std::min<int64_t>(P, (int64_t(1) << 26) / (B * d)));
  rmm::device_uvector<float> p(cl * B * d, stream), z(cl * B * m, stream), r(cl * B, stream),
    a(cl * B, stream);
  auto handle = raft::resource::get_cublas_handle(res);
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  cublasMath_t mode;
  RAFT_CUBLAS_TRY(cublasGetMathMode(handle, &mode));
  RAFT_CUBLAS_TRY(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
  for (int64_t j0 = 0; j0 < P; j0 += cl) {
    const int64_t nl = std::min<int64_t>(cl, P - j0);
    center_per_list<<<blocks_for(std::min<int64_t>(nl * B * d, int64_t(1) << 24)),
                      256,
                      0,
                      stream>>>(qf, B, d, idx.local_centers().data_handle() + j0 * d, nl, p.data());
    RAFT_CUDA_TRY(cudaGetLastError());
    const float one = 1, zero = 0;
    // column major per list: z^T (m x B) = V_j (m x d, stored d x m)^T * p_j^T (d x B)
    RAFT_CUBLAS_TRY(cublasSgemmStridedBatched(handle,
                                              CUBLAS_OP_T,
                                              CUBLAS_OP_N,
                                              m,
                                              int(B),
                                              int(d),
                                              &one,
                                              idx.local_bases().data_handle() + j0 * m * d,
                                              int(d),
                                              int64_t(m) * d,
                                              p.data(),
                                              int(d),
                                              B * d,
                                              &zero,
                                              z.data(),
                                              m,
                                              B * m,
                                              int(nl)));
    lift_norms<<<blocks_for(nl * B * 32), 256, 0, stream>>>(
      p.data(), d, z.data(), m, nl * B, r.data(), a.data(), nullptr);
    encode_queries_lists<<<blocks_for(nl * B * 32), 256, 0, stream>>>(z.data(),
                                                                      r.data(),
                                                                      a.data(),
                                                                      m,
                                                                      nl * B,
                                                                      B,
                                                                      consts.data() + j0,
                                                                      MP,
                                                                      float(1.0 - eps),
                                                                      float(g),
                                                                      rows + j0 * B * MP,
                                                                      aqh + j0 * B,
                                                                      margin + j0 * B);
    RAFT_CUDA_TRY(cudaGetLastError());
  }
  RAFT_CUBLAS_TRY(cublasSetMathMode(handle, mode));
  raft::resource::sync_stream(res);  // the host constants go out of scope
}

/**
 * Local projections of the lists of a build (the rows in list order): center, directions, and
 * scale of every list from its own vectors, its rows encoded with them, and its certificate
 * maxima. The anchors are encoded again with the global projection, which the anchor stage uses.
 */
template <typename T, typename Src>
void build_local(raft::resources const& res, index<T>& idx, Src& in_order)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  const int P = idx.n_lists(), m = idx.width(), MP = idx.row_length();
  const int64_t d     = idx.dim();
  const auto& off     = idx.list_offsets();
  const bool cosine   = idx.metric() == DistanceType::CosineExpanded;
  const bool centered = idx.metric() != DistanceType::InnerProduct;
  auto centers        = raft::make_device_matrix<float, int64_t>(res, P, d);
  auto bases          = raft::make_device_matrix<float, int64_t>(res, int64_t(P) * m, d);
  std::vector<double> alpha(P, 1.0);
  std::vector<typename index<T>::certificate_maxima> maxima(P);
  rmm::device_buffer amax_buf(sizeof(float), stream);
  auto* amax = static_cast<float*>(amax_buf.data());
  for (int j = 0; j < P; ++j) {
    float* cj         = centers.data_handle() + int64_t(j) * d;
    float* vj         = bases.data_handle() + int64_t(j) * m * d;
    const int64_t n_j = off[j + 1] - off[j];
    if (n_j == 0) {  // an empty list: the global projection keeps its slot orthonormal
      raft::copy(cj, idx.center().data_handle(), d, stream);
      raft::copy(vj, idx.basis().data_handle(), int64_t(m) * d, stream);
      continue;
    }
    range_source<Src> rs{in_order, off[j], n_j, d};
    local_basis<T>(res, rs, d, m, cosine, centered, cj, vj);
    const projection pj{cj, vj, m};
    RAFT_CUDA_TRY(cudaMemsetAsync(amax, 0, sizeof(float), stream));
    for_lifted_chunks(
      res,
      idx,
      pj,
      rs,
      n_j,
      cosine,
      [&](int64_t, int64_t cn, const auto*, const float*, float*, float*, float* a) {
        max_of<<<blocks_for(cn), 256, 0, stream>>>(a, cn, amax);
        RAFT_CUDA_TRY(cudaGetLastError());
      });
    float h_amax = 0;
    raft::copy(&h_amax, amax, 1, stream);
    raft::resource::sync_stream(res);
    // every lifted row of the list gets norm at most 64; the floor at alpha / 16 keeps the query
    // rows of a tight list in half range for queries within about 64 radii of the collection
    alpha[j]  = std::max(std::sqrt(std::max(double(h_amax), 1e-30)) / 64.0, idx.alpha / 16.0);
    maxima[j] = append_rows(res, idx, rs, off[j], pj, alpha[j], {});
  }
  idx.set_local(std::move(centers), std::move(bases), std::move(alpha), std::move(maxima));
  // global maxima: no row is encoded globally yet, but the bound u_all that every vector satisfies
  // (a query with fewer than k anchors) reads the lifted norms of all rows, which the scale alpha
  // of the build keeps at most 64
  idx.maxima.lifted = idx.maxima.norm = 64.0 * (1 + std::ldexp(1.0, -10));
  // the anchor rows with the global projection
  // the size of the index is set after the build: the anchors of all rows of the lists
  const int64_t na = (off[P] + idx.anchor_stride() - 1) / idx.anchor_stride();
  rmm::device_buffer mx_buf(sizeof(maxima_bits), stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(mx_buf.data(), 0, sizeof(maxima_bits), stream));
  device_source<T> anchors{idx.anchor_vectors().data_handle(), na, d};
  const double g = gamma_of(MP);
  for_lifted_chunks(
    res,
    idx,
    anchors,
    na,
    cosine,
    [&](int64_t i, int64_t cn, const T*, const float*, float* z, float* r, float* a) {
      encode_rows<<<blocks_for(cn * 32), 256, 0, stream>>>(
        z,
        r,
        a,
        m,
        cn,
        MP,
        float(1.0 / idx.alpha),
        float(g),
        centered,
        idx.anchor_rows().data_handle() + i * MP,
        static_cast<maxima_bits*>(mx_buf.data()));
      RAFT_CUDA_TRY(cudaGetLastError());
    });
  raft::resource::sync_stream(res);
}

}  // namespace cuvs::neighbors::certex::detail
