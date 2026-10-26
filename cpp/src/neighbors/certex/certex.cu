/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/cuda_fp16.hpp>  // first: __half with default visibility, as in cuVS

#include "../../core/nvtx.hpp"
#include "detail/certex_kernels.cuh"
#include "detail/kernels.hpp"

#include <cuvs/neighbors/certex.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/selection/select_k.hpp>

#include <raft/core/cublas_macros.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/eig.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/error.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace cuvs::neighbors::certex {

namespace detail {

using cuvs::distance::DistanceType;

inline auto pow2_at_least(int v) -> int
{
  int p = 1;
  while (p < v) {
    p <<= 1;
  }
  return p;
}

/** Row length of width m: m + 1 lifted and 4 certificate coordinates, padded to 64 or 128n. */
inline auto row_length_of(int m) -> int
{
  const int need = m + 5;
  return need <= 64 ? 64 : (need <= 128 ? 128 : (need + 127) / 128 * 128);
}

inline auto gamma_of(int row_len) -> double { return 16.0 * row_len * std::ldexp(1.0, -24); }

/** Smallest multiple of 2^-11 above gamma + 2^-22, so that 1 - eps is exact in half precision. */
inline auto eps_of(double gamma) -> double
{
  const double step = std::ldexp(1.0, -11);
  return std::ceil((gamma + std::ldexp(1.0, -22)) / step) * step;
}

inline void check_metric(DistanceType metric)
{
  RAFT_EXPECTS(metric == DistanceType::L2Expanded || metric == DistanceType::InnerProduct ||
                 metric == DistanceType::CosineExpanded,
               "certex supports L2Expanded, InnerProduct, and CosineExpanded");
}

/** Blocks that cover `threads` threads; the warp-per-row kernels rely on a full cover (no cap). */
inline auto blocks_for(int64_t threads, int block = 256) -> unsigned
{
  const int64_t blocks = (threads + block - 1) / block;
  RAFT_EXPECTS(blocks < (int64_t(1) << 31), "grid too large");
  return unsigned(std::max<int64_t>(1, blocks));
}

/** fp32 GEMM without reduced-precision math: out[n, m] = p[n, d] basis[m, d]^T (row major). */
inline void project(raft::resources const& res,
                    const float* p,
                    int64_t n,
                    int64_t d,
                    const float* basis,
                    int m,
                    float* out)
{
  auto handle = raft::resource::get_cublas_handle(res);
  RAFT_CUBLAS_TRY(cublasSetStream(handle, raft::resource::get_cuda_stream(res).get()));
  cublasMath_t mode;
  RAFT_CUBLAS_TRY(cublasGetMathMode(handle, &mode));
  RAFT_CUBLAS_TRY(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
  const float one = 1, zero = 0;
  // column major: out^T (m x n) = basis (m x d, stored as d x m) ^T * p^T (d x n)
  RAFT_CUBLAS_TRY(cublasSgemm(handle,
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              m,
                              int(n),
                              int(d),
                              &one,
                              basis,
                              int(d),
                              p,
                              int(d),
                              &zero,
                              out,
                              m));
  RAFT_CUBLAS_TRY(cublasSetMathMode(handle, mode));
}

/** out[i] = i for i < n. */
__global__ void iota_from(int32_t* out, int64_t n)
{
  const int64_t t = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (t < n) { out[t] = int32_t(t); }
}

/** Float copy of n rows, unit-normalized for cosine; one warp per row. */
template <typename T>
__global__ void rows_to_float(const T* x, int64_t dim, int64_t n, bool normalize, float* out)
{
  const int64_t row = (blockIdx.x * int64_t(blockDim.x) + threadIdx.x) / 32;
  const int lane    = threadIdx.x & 31;
  if (row >= n) { return; }
  float ss = 0;
  for (int64_t j = lane; j < dim; j += 32) {
    const float v = to_f(x[row * dim + j]);
    ss += v * v;
  }
  for (int off = 16; off; off >>= 1) {
    ss += __shfl_xor_sync(0xffffffffu, ss, off);
  }
  const float s = normalize ? 1.0f / fmaxf(sqrtf(ss), 1e-30f) : 1.0f;
  for (int64_t j = lane; j < dim; j += 32) {
    out[row * dim + j] = to_f(x[row * dim + j]) * s;
  }
}

__global__ void row_norms(const float* x, int64_t dim, int64_t n, double* out)
{
  const int64_t row = (blockIdx.x * int64_t(blockDim.x) + threadIdx.x) / 32;
  const int lane    = threadIdx.x & 31;
  if (row >= n) { return; }
  double ss = 0;
  for (int64_t j = lane; j < dim; j += 32) {
    const double v = x[row * dim + j];
    ss += v * v;
  }
  for (int off = 16; off; off >>= 1) {
    ss += __shfl_xor_sync(0xffffffffu, ss, off);
  }
  if (lane == 0) { out[row] = sqrt(ss); }
}

__global__ void max_of(const float* a, int64_t n, float* out)
{
  float m = 0;
  for (int64_t t = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; t < n;
       t += int64_t(gridDim.x) * blockDim.x) {
    m = fmaxf(m, a[t]);
  }
  for (int off = 16; off; off >>= 1) {
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
  }
  if ((threadIdx.x & 31) == 0) { atomicMax(reinterpret_cast<int*>(out), __float_as_int(m)); }
}

__global__ void strided_rows(const float* x, int64_t dim, int64_t n_out, int64_t step, double* out)
{
  const int64_t total = n_out * dim;
  for (int64_t t = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; t < total;
       t += int64_t(gridDim.x) * blockDim.x) {
    const int64_t r = t / dim, j = t - r * dim;
    out[t] = x[r * step * dim + j];
  }
}

__global__ void column_means(const double* y, int64_t n, int64_t dim, double* mean)
{
  const int64_t j = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (j >= dim) { return; }
  double s = 0;
  for (int64_t r = 0; r < n; ++r) {
    s += y[r * dim + j];
  }
  mean[j] = s / double(n);
}

__global__ void center_columns(double* y, int64_t n, int64_t dim, const double* mean)
{
  const int64_t total = n * dim;
  for (int64_t t = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; t < total;
       t += int64_t(gridDim.x) * blockDim.x) {
    y[t] -= mean[t % dim];
  }
}

__global__ void top_eigenvectors(const double* vecs, int64_t dim, int m, float* basis)
{
  const int64_t total = int64_t(m) * dim;
  for (int64_t t = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; t < total;
       t += int64_t(gridDim.x) * blockDim.x) {
    const int64_t i = t / dim, r = t - i * dim;
    basis[t] = float(vecs[r + (dim - 1 - i) * dim]);  // eigenvalues ascend
  }
}

__global__ void double_to_float(const double* in, int64_t n, float* out)
{
  const int64_t t = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (t < n) { out[t] = float(in[t]); }
}

template <score_kind S>
__global__ void finish(const double* best_score,
                       const int64_t* best_id,
                       const double* q_norm,
                       int64_t n,
                       int k,
                       const int32_t* row_ids,
                       float* distances,
                       int64_t* neighbors)
{
  const int64_t t = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (t >= n * k) { return; }
  const double s = best_score[t];
  double d;
  if constexpr (S == score_kind::inner) {
    const double qn = q_norm[t / k];
    d               = (qn * qn - s) / 2;  // inner product from s = |q|^2 - 2 <q, x>
  } else if constexpr (S == score_kind::cosine) {
    d = s / 2;  // 1 - cos from s = 2 - 2 cos
  } else {
    d = s;
  }
  const bool found = best_id[t] >= 0;
  distances[t]     = found ? float(d) : (S == score_kind::inner ? -INFINITY : INFINITY);
  // the id of the row (rows and ids differ in an index with lists)
  neighbors[t] = (found && row_ids) ? int64_t(row_ids[best_id[t]]) : best_id[t];
}

/** Rows of a dataset in device memory: a chunk is a pointer into the data. */
template <typename T>
struct device_source {
  const T* x;
  int64_t n, d;
  auto rows(raft::resources const&, int64_t i, int64_t) -> const T* { return x + i * d; }
  /** Rows i = j0 * step, (j0 + 1) * step, ... gathered contiguously on the device. */
  auto strided(raft::resources const& res, int64_t step, int64_t j0, int64_t cnt) -> const T*;
  rmm::device_uvector<T> staging{0, cudaStream_t{}};  // any rmm: a raw stream converts
};

/** Rows of a dataset in host memory: a chunk is copied into a device staging buffer. */
template <typename T>
struct host_source {
  const T* x;
  int64_t n, d;
  rmm::device_uvector<T> staging{0, cudaStream_t{}};  // any rmm: a raw stream converts
  std::vector<T> packed;
  auto rows(raft::resources const& res, int64_t i, int64_t cn) -> const T*
  {
    cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
    staging.resize(cn * d, stream);
    raft::copy(staging.data(), x + i * d, cn * d, stream);
    return staging.data();
  }
  auto strided(raft::resources const& res, int64_t step, int64_t j0, int64_t cnt) -> const T*
  {
    packed.resize(cnt * d);
    for (int64_t j = 0; j < cnt; ++j) {
      std::copy_n(x + (j0 + j) * step * d, d, packed.data() + j * d);
    }
    cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
    staging.resize(cnt * d, stream);
    raft::copy(staging.data(), packed.data(), cnt * d, stream);
    raft::resource::sync_stream(res);  // packed is reused by the next call
    return staging.data();
  }
};

template <typename T>
__global__ void strided_copy(const T* x, int64_t d, int64_t step, int64_t j0, int64_t cnt, T* out)
{
  const int64_t total = cnt * d;
  for (int64_t t = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; t < total;
       t += int64_t(gridDim.x) * blockDim.x) {
    const int64_t j = t / d, c = t - j * d;
    out[t] = x[(j0 + j) * step * d + c];
  }
}

template <typename T>
auto device_source<T>::strided(raft::resources const& res, int64_t step, int64_t j0, int64_t cnt)
  -> const T*
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  staging.resize(cnt * d, stream);
  strided_copy<T><<<blocks_for(cnt * d), 256, 0, stream>>>(x, d, step, j0, cnt, staging.data());
  RAFT_CUDA_TRY(cudaGetLastError());
  return staging.data();
}

__global__ void float_to_double(const float* in, int64_t n, double* out)
{
  for (int64_t t = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; t < n;
       t += int64_t(gridDim.x) * blockDim.x) {
    out[t] = in[t];
  }
}

__global__ void center_into(const float* x, int64_t n, int64_t d, const float* c, float* out)
{
  const int64_t total = n * d;
  for (int64_t t = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; t < total;
       t += int64_t(gridDim.x) * blockDim.x) {
    out[t] = x[t] - c[t % d];
  }
}

/** Copy the anchors (global ids divisible by stride) of a chunk of rows g0, g0 + 1, ... */
template <typename T>
__global__ void gather_anchor_vectors(
  const T* chunk, int64_t d, int64_t g0, int64_t cn, int stride, T* anchors)
{
  const int64_t first = (g0 + stride - 1) / stride, last = (g0 + cn + stride - 1) / stride;
  const int64_t total = (last - first) * d;
  for (int64_t t = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; t < total;
       t += int64_t(gridDim.x) * blockDim.x) {
    const int64_t a = first + t / d, c = t % d;
    anchors[a * d + c] = chunk[(a * stride - g0) * d + c];
  }
}

}  // namespace detail
}  // namespace cuvs::neighbors::certex

// k-means lists: their build and the probe that tightens the threshold
#include "detail/lists.cuh"

namespace cuvs::neighbors::certex {
namespace detail {

/** Principal directions of a strided sample of the vectors, and the width. */
template <typename T, typename Src>
auto fit_projection(raft::resources const& res, index_params const& params, Src& src) -> index<T>
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  const int64_t n = src.n, d = src.d;
  const int64_t s     = std::max<int64_t>(1, std::min(params.pca_sample, n));
  const int64_t step  = std::max<int64_t>(1, n / s);
  const bool centered = params.metric != DistanceType::InnerProduct;
  const bool cosine   = params.metric == DistanceType::CosineExpanded;
  rmm::device_uvector<double> y(s * d, stream);
  {
    const int64_t blk = std::max<int64_t>(1, std::min<int64_t>(s, (int64_t(1) << 26) / d));
    rmm::device_uvector<float> f(blk * d, stream);
    for (int64_t j0 = 0; j0 < s; j0 += blk) {
      const int64_t cnt = std::min(blk, s - j0);
      const T* rows     = src.strided(res, step, j0, cnt);
      rows_to_float<T><<<blocks_for(cnt * 32), 256, 0, stream>>>(rows, d, cnt, cosine, f.data());
      float_to_double<<<blocks_for(cnt * d), 256, 0, stream>>>(
        f.data(), cnt * d, y.data() + j0 * d);
      RAFT_CUDA_TRY(cudaGetLastError());
    }
  }
  rmm::device_uvector<double> mean(d, stream);
  if (centered) {
    column_means<<<blocks_for(d), 256, 0, stream>>>(y.data(), s, d, mean.data());
    center_columns<<<blocks_for(s * d), 256, 0, stream>>>(y.data(), s, d, mean.data());
  } else {
    RAFT_CUDA_TRY(cudaMemsetAsync(mean.data(), 0, d * sizeof(double), stream));
  }
  RAFT_CUDA_TRY(cudaGetLastError());
  auto cov = raft::make_device_matrix<double, int64_t, raft::col_major>(res, d, d);
  {
    auto handle = raft::resource::get_cublas_handle(res);
    RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
    const double one = 1, zero = 0;
    RAFT_CUBLAS_TRY(cublasDgemm(handle,
                                CUBLAS_OP_N,
                                CUBLAS_OP_T,
                                int(d),
                                int(d),
                                int(s),
                                &one,
                                y.data(),
                                int(d),
                                y.data(),
                                int(d),
                                &zero,
                                cov.data_handle(),
                                int(d)));
  }
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
  std::vector<double> h_vals(d);
  raft::copy(h_vals.data(), vals.data_handle(), d, stream);
  raft::resource::sync_stream(res);
  int m = params.width;
  if (m <= 0) {  // smallest tile-aligned width whose directions capture the variance target
    double total = 0;
    for (auto v : h_vals) {
      total += std::max(v, 0.0);
    }
    std::vector<double> frac(d);
    double acc = 0;
    for (int64_t i = 0; i < d; ++i) {
      acc += std::max(h_vals[d - 1 - i], 0.0);
      frac[i] = acc / std::max(total, 1e-300);
    }
    m = int(d);
    for (int L = 64; L <= 4096; L = L < 128 ? L + 64 : L + 128) {
      const int cand = L - 5;
      if (cand >= d || frac[cand - 1] >= params.variance) {
        m = int(std::min<int64_t>(cand, d));
        break;
      }
    }
  }
  m = int(std::max<int64_t>(1, std::min<int64_t>(m, d)));
  index<T> idx(res, params.metric, d, m, params.anchor_stride, params.host_vectors);
  double_to_float<<<blocks_for(d), 256, 0, stream>>>(mean.data(), d, idx.center().data_handle());
  top_eigenvectors<<<blocks_for(int64_t(m) * d), 256, 0, stream>>>(
    vecs.data_handle(), d, m, idx.basis().data_handle());
  RAFT_CUDA_TRY(cudaGetLastError());
  return idx;
}

/**
 * Lift the rows [0, cnt) of a source in chunks: f(i, cn, rows, xf, z, r, a) receives the rows as T
 * on the device, their float copy (normalized for cosine), and their lifted coordinates.
 */
/** A projection: rows of `basis` ([m, d], orthonormal) around `center` ([d]). */
struct projection {
  const float* center;
  const float* basis;
  int m;
};

template <typename T>
auto global_projection(index<T> const& idx) -> projection
{
  return {idx.center().data_handle(), idx.basis().data_handle(), idx.width()};
}

template <typename T, typename Src, typename F>
void for_lifted_chunks(raft::resources const& res,
                       index<T> const& idx,
                       projection pj,
                       Src& src,
                       int64_t cnt,
                       bool normalize,
                       F&& f)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  const int64_t d     = idx.dim();
  const int m         = pj.m;
  const int64_t blk   = std::max<int64_t>(1, std::min<int64_t>(1 << 20, (int64_t(1) << 27) / d));
  const int64_t cap   = std::min(blk, cnt);
  rmm::device_uvector<float> xf(cap * d, stream), p(cap * d, stream), z(cap * m, stream),
    r(cap, stream), a(cap, stream);
  for (int64_t i = 0; i < cnt; i += blk) {
    const int64_t cn = std::min(blk, cnt - i);
    const auto* rows = src.rows(res, i, cn);  // the element type of the source (queries: float)
    using R          = std::remove_cv_t<std::remove_pointer_t<decltype(rows)>>;
    rows_to_float<R><<<blocks_for(cn * 32), 256, 0, stream>>>(rows, d, cn, normalize, xf.data());
    center_into<<<blocks_for(cn * d), 256, 0, stream>>>(xf.data(), cn, d, pj.center, p.data());
    RAFT_CUDA_TRY(cudaGetLastError());
    project(res, p.data(), cn, d, pj.basis, m, z.data());
    lift_norms<<<blocks_for(cn * 32), 256, 0, stream>>>(
      p.data(), d, z.data(), m, cn, r.data(), a.data(), nullptr);
    RAFT_CUDA_TRY(cudaGetLastError());
    f(i, cn, rows, xf.data(), z.data(), r.data(), a.data());
  }
}

template <typename T, typename Src, typename F>
void for_lifted_chunks(
  raft::resources const& res, index<T> const& idx, Src& src, int64_t cnt, bool normalize, F&& f)
{
  for_lifted_chunks(res, idx, global_projection(idx), src, cnt, normalize, std::forward<F>(f));
}

/**
 * Append the rows of a source as the vectors g0, g0 + 1, ...: store them (device or pinned host),
 * encode their screening rows, raise the certificate maxima, and copy their anchors.
 */
template <typename T, typename Src>
auto append_rows(raft::resources const& res,
                 index<T>& idx,
                 Src& src,
                 int64_t g0,
                 projection pj,
                 double alpha,
                 typename index<T>::certificate_maxima const& start) ->
  typename index<T>::certificate_maxima
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  const int64_t d     = idx.dim();
  const bool cosine   = idx.metric() == DistanceType::CosineExpanded;
  rmm::device_buffer mx_buf(sizeof(maxima_bits), stream);
  auto* mx = static_cast<maxima_bits*>(mx_buf.data());
  const maxima_bits h0{std::bit_cast<int>(float(start.lifted)),
                       std::bit_cast<int>(float(start.norm)),
                       std::bit_cast<int>(float(start.error)),
                       std::bit_cast<int>(float(start.high)),
                       std::bit_cast<int>(float(start.low))};
  RAFT_CUDA_TRY(cudaMemcpyAsync(mx, &h0, sizeof(h0), cudaMemcpyHostToDevice, stream));
  const bool with_half_norm = idx.metric() != DistanceType::InnerProduct;
  const double g            = gamma_of(idx.row_length());
  for_lifted_chunks(
    res,
    idx,
    pj,
    src,
    src.n,
    cosine,
    [&](int64_t i, int64_t cn, const T* rows, const float*, float* z, float* r, float* a) {
      encode_rows<<<blocks_for(cn * 32), 256, 0, stream>>>(
        z,
        r,
        a,
        pj.m,
        cn,
        idx.row_length(),
        float(1.0 / alpha),
        float(g),
        with_half_norm,
        idx.rows().data_handle() + (g0 + i) * idx.row_length(),
        mx);
      RAFT_CUDA_TRY(cudaGetLastError());
      // the vectors are stored as given; for cosine, only the screening rows lift normalized copies
      raft::copy(idx.vectors_ptr() + (g0 + i) * d, rows, cn * d, stream);
      gather_anchor_vectors<T><<<blocks_for(cn * d / idx.anchor_stride() + d), 256, 0, stream>>>(
        rows, d, g0 + i, cn, idx.anchor_stride(), idx.anchor_vectors().data_handle());
      RAFT_CUDA_TRY(cudaGetLastError());
      raft::resource::sync_stream(res);  // the source may reuse its staging buffer
    });
  maxima_bits h1;
  RAFT_CUDA_TRY(cudaMemcpyAsync(&h1, mx, sizeof(h1), cudaMemcpyDeviceToHost, stream));
  raft::resource::sync_stream(res);
  const typename index<T>::certificate_maxima raised{std::bit_cast<float>(h1.lifted),
                                                     std::bit_cast<float>(h1.norm),
                                                     std::bit_cast<float>(h1.error),
                                                     std::bit_cast<float>(h1.high),
                                                     std::bit_cast<float>(h1.low)};
  const int s           = idx.anchor_stride();
  const int64_t first_a = (g0 + s - 1) / s, last_a = (g0 + src.n + s - 1) / s;
  if (last_a > first_a) {
    gather_anchor_rows<<<blocks_for((last_a - first_a) * idx.row_length()), 256, 0, stream>>>(
      idx.rows().data_handle(),
      idx.row_length(),
      first_a,
      last_a,
      s,
      idx.anchor_rows().data_handle());
    RAFT_CUDA_TRY(cudaGetLastError());
  }
  return raised;
}

/** append_rows with the global projection, scale, and certificate maxima of the index. */
template <typename T, typename Src>
auto append_rows(raft::resources const& res, index<T>& idx, Src& src, int64_t g0) ->
  typename index<T>::certificate_maxima
{
  return append_rows(res, idx, src, g0, global_projection(idx), idx.alpha, idx.maxima);
}

/** Float copy of n rows of T (normalized for cosine). */
template <typename T>
auto to_float(raft::resources const& res, const T* x, int64_t n, int64_t d, bool normalize)
  -> rmm::device_uvector<float>
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  rmm::device_uvector<float> out(n * d, stream);
  if (n > 0) {
    rows_to_float<T><<<blocks_for(n * 32), 256, 0, stream>>>(x, d, n, normalize, out.data());
    RAFT_CUDA_TRY(cudaGetLastError());
  }
  return out;
}

}  // namespace detail
}  // namespace cuvs::neighbors::certex

// local projections of the lists: their build and the query rows of a search
#include "detail/local.cuh"
// int8 screening rows: their encoding and the query codes of a search
#include "detail/int8.cuh"
#include "detail/verify.cuh"

namespace cuvs::neighbors::certex {
namespace detail {

namespace {
thread_local search_stats* stats_target = nullptr;
}  // namespace

thread_local threshold_hooks* hooks_target = nullptr;

void collect_stats(search_stats* s) { stats_target = s; }
void set_threshold_hooks(threshold_hooks* h) { hooks_target = h; }
auto threshold_hooks_sink() -> threshold_hooks* { return hooks_target; }
auto stats_sink() -> search_stats* { return stats_target; }

void iota_fill(raft::resources const& res, int32_t* out, int64_t n)
{
  if (n == 0) { return; }
  iota_from<<<blocks_for(n), 256, 0, raft::resource::get_cuda_stream(res).get()>>>(out, n);
  RAFT_CUDA_TRY(cudaGetLastError());
}

void invert_into(raft::resources const& res, const int32_t* perm, int64_t n, int32_t* inverse)
{
  if (n == 0) { return; }
  invert_permutation<<<blocks_for(n), 256, 0, raft::resource::get_cuda_stream(res).get()>>>(
    perm, n, inverse);
  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename T, typename Src>
auto build_from(raft::resources const& res, index_params const& params, Src& src) -> index<T>
{
  cuvs::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "certex::build(%zu, %zu)", size_t(src.n), size_t(src.d));
  detail::require_supported_device();
  check_metric(params.metric);
  RAFT_EXPECTS(params.anchor_stride >= 1, "anchor_stride must be positive");
  RAFT_EXPECTS(!params.local_projections || params.n_lists > 0,
               "local projections need lists (n_lists > 0)");
  RAFT_EXPECTS(!params.int8_rows || params.n_lists == 0, "int8 rows are not combined with lists");
  RAFT_EXPECTS(!params.int8_rows || params.width * 127 * 127 < (1 << 24),
               "int8 rows need 127^2 m < 2^24 (m <= 1040)");
  RAFT_EXPECTS(src.n > 0 && src.d > 0, "the dataset is empty");
  const bool cosine       = params.metric == DistanceType::CosineExpanded;
  constexpr bool is_float = std::is_same_v<T, float> || std::is_same_v<T, half>;
  RAFT_EXPECTS(!cosine || is_float, "cosine needs float or half vectors");
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  auto idx            = fit_projection<T>(res, params, src);
  if (params.int8_rows) { idx.use_int8_rows(); }
  idx.reserve(res, src.n);
  // common scale: every lifted row gets norm at most 64
  rmm::device_buffer amax_buf(sizeof(float), stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(amax_buf.data(), 0, sizeof(float), stream));
  auto* amax = static_cast<float*>(amax_buf.data());
  for_lifted_chunks(res,
                    idx,
                    src,
                    src.n,
                    cosine,
                    [&](int64_t, int64_t cn, const T*, const float*, float*, float*, float* a) {
                      max_of<<<blocks_for(cn), 256, 0, stream>>>(a, cn, amax);
                      RAFT_CUDA_TRY(cudaGetLastError());
                    });
  float h_amax = 0;
  raft::copy(&h_amax, amax, 1, stream);
  raft::resource::sync_stream(res);
  idx.alpha = std::sqrt(std::max(double(h_amax), 1e-30)) / 64.0;
  if (params.int8_rows) {  // int8.cuh
    idx.max_a = append_rows_int8(res, idx, src, 0);
    // the bound u_all (fewer than k anchors) reads the lifted norms of all rows: sqrt(max a)
    idx.maxima.lifted = idx.maxima.norm =
      std::sqrt(idx.max_a) / idx.alpha * (1 + std::ldexp(1.0, -10));
  } else if (params.n_lists >
             0) {  // the vectors in list order (lists.cuh); ids stay the input order
    auto lists    = fit_lists<T>(res, params, src);
    auto in_order = permuted(res, src, lists.row_ids.data_handle());
    // the device buffers keep their addresses when the index takes them over
    idx.set_lists(std::move(lists.offsets),
                  std::move(lists.centers),
                  std::move(lists.row_ids),
                  std::move(lists.id_rows));
    if (params.local_projections) {  // local.cuh; vectors added later use the global projection
      build_local(res, idx, in_order);
    } else {
      idx.maxima = append_rows(res, idx, in_order, 0);
    }
  } else {
    idx.maxima = append_rows(res, idx, src, 0);
  }
  idx.set_size(src.n);
  return idx;
}

template <typename T, typename Src>
void extend_from(raft::resources const& res, Src& src, index<T>* idx)
{
  cuvs::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "certex::extend(%zu, %zu)", size_t(src.n), size_t(src.d));
  RAFT_EXPECTS(src.d == idx->dim(), "dimension mismatch");
  if (src.n == 0) { return; }
  std::lock_guard writer(idx->sync().writer);  // one update at a time
  const int64_t n0 = idx->size();
  if (n0 + src.n > idx->capacity()) {  // reallocation: wait for the running search batches
    std::unique_lock grow(idx->sync().rw);
    idx->reserve(res, n0 + src.n);
  }
  // the new rows lie beyond the published vectors, so that running searches never read them
  if (idx->int8_rows()) {
    const double max_a = append_rows_int8(res, *idx, src, n0);
    std::unique_lock publish(idx->sync().rw);
    idx->max_a         = max_a;
    idx->maxima.lifted = idx->maxima.norm =
      std::sqrt(max_a) / idx->alpha * (1 + std::ldexp(1.0, -10));
    idx->set_size(n0 + src.n);
    return;
  }
  const auto raised = append_rows(res, *idx, src, n0);
  std::unique_lock publish(idx->sync().rw);
  idx->maxima = raised;
  idx->set_size(n0 + src.n);
}

template <typename T>
auto build(raft::resources const& res,
           index_params const& params,
           raft::device_matrix_view<const T, int64_t, raft::row_major> dataset) -> index<T>
{
  device_source<T> src{dataset.data_handle(), dataset.extent(0), dataset.extent(1)};
  return build_from<T>(res, params, src);
}

template <typename T>
auto build(raft::resources const& res,
           index_params const& params,
           raft::host_matrix_view<const T, int64_t, raft::row_major> dataset) -> index<T>
{
  host_source<T> src{dataset.data_handle(), dataset.extent(0), dataset.extent(1)};
  return build_from<T>(res, params, src);
}

template <typename T>
void extend(raft::resources const& res,
            raft::device_matrix_view<const T, int64_t, raft::row_major> new_vectors,
            index<T>* idx)
{
  device_source<T> src{new_vectors.data_handle(), new_vectors.extent(0), new_vectors.extent(1)};
  extend_from(res, src, idx);
}

template <typename T>
void extend(raft::resources const& res,
            raft::host_matrix_view<const T, int64_t, raft::row_major> new_vectors,
            index<T>* idx)
{
  host_source<T> src{new_vectors.data_handle(), new_vectors.extent(0), new_vectors.extent(1)};
  extend_from(res, src, idx);
}

/**
 * NVTX ranges over the stages of a search batch, for Nsight Systems: a range opens at
 * construction, and mark(i) closes stage i and opens stage i + 1.
 */
class stage_ranges {
 public:
  stage_ranges() { open(0); }
  void mark(int stage)
  {
    cuvs::common::nvtx::pop_range<cuvs::common::nvtx::domain::cuvs>();
    open_ = false;
    if (stage + 1 < kStages) { open(stage + 1); }
  }
  ~stage_ranges()
  {
    if (open_) { cuvs::common::nvtx::pop_range<cuvs::common::nvtx::domain::cuvs>(); }
  }
  stage_ranges(const stage_ranges&)                    = delete;
  auto operator=(const stage_ranges&) -> stage_ranges& = delete;

 private:
  static constexpr int kStages = 8;
  void open(int stage)
  {
    static constexpr const char* names[kStages] = {
      "encode", "anchors", "threshold", "screen", "keys", "sort", "verify", "output"};
    cuvs::common::nvtx::push_range<cuvs::common::nvtx::domain::cuvs>("certex::search::%s",
                                                                     names[stage]);
    open_ = true;
  }
  bool open_ = false;
};

template <typename T>
void remove(raft::resources const& res,
            raft::device_vector_view<const int64_t, int64_t> ids,
            index<T>* idx)
{
  cuvs::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("certex::remove(%zu)",
                                                                        size_t(ids.extent(0)));
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  const int64_t n_ids = ids.extent(0);
  if (n_ids == 0) { return; }
  std::lock_guard writer(idx->sync().writer);
  rmm::device_uvector<unsigned long long> newly(1, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(newly.data(), 0, sizeof(unsigned long long), stream));
  remove_rows<<<blocks_for(n_ids), 256, 0, stream>>>(
    ids.data_handle(),
    idx->mapped() ? idx->id_rows().data_handle() : nullptr,
    n_ids,
    idx->size(),
    idx->int8_rows() ? nullptr : idx->rows().data_handle(),  // int8 rows: remove_int8_rows below
    idx->anchor_rows().data_handle(),
    idx->row_length(),
    idx->width() + 1,
    idx->anchor_stride(),
    idx->removed().data_handle(),
    newly.data());
  RAFT_CUDA_TRY(cudaGetLastError());
  if (idx->int8_rows()) {
    remove_int8_rows<<<blocks_for(n_ids), 256, 0, stream>>>(
      ids.data_handle(),
      idx->mapped() ? idx->id_rows().data_handle() : nullptr,
      n_ids,
      idx->size(),
      idx->row_scalars().data_handle());
    RAFT_CUDA_TRY(cudaGetLastError());
  }
  unsigned long long h = 0;
  raft::copy(&h, newly.data(), 1, stream);
  raft::resource::sync_stream(res);
  std::unique_lock publish(idx->sync().rw);
  idx->add_removed(int64_t(h));
}

/**
 * Largest gap between the exact cosine score 2 - 2 cos(q, x) and the squared distance of the
 * float32-normalized copies q', x' that the screening rows lift. rows_to_float() computes a norm
 * from a sum of ceil(d / 32) + 5 rounded additions per lane, a correctly rounded square root and
 * reciprocal, and one rounded product per coordinate, so that |x' - x / |x|| <= eta_1 = ((ceil(d /
 * 32) + 6) / 2 + 4) u (1 + 2^-10) with u = 2^-24, and the same for q. With eta = eta_q + eta_x,
 * | |q' - x'| - |q / |q| - x / |x|| | <= eta and |q' - x'| <= 2 + eta, so that the exact score is
 * at least |q' - x'|^2 - 2 eta (2 + eta). The bound uses (d + 16) u per vector, which exceeds
 * eta_1.
 */
inline auto cosine_slack(int64_t d) -> double
{
  const double eta = 2.0 * double(d + 16) * std::ldexp(1.0, -24);
  return 2.0 * eta * (2.0 + eta);
}

/**
 * Label partition of one batch (queries sorted by label): the partitioned rows and their ids, the
 * groups of the routed pass, and the label filter for the rows added after the partition.
 */
struct route_plan {
  const half* rows;
  int64_t n_rows;
  const int32_t* ids;  // vector row of every partition row (null: the rows themselves)
  const route_group* groups;
  int n_groups;
  int n_tiles;
  Filter tail;
  // local projections: the query rows of the routed pass, their thresholds, and the constants of
  // their lower bounds (null: the batch's own query rows)
  const half* q_rows     = nullptr;
  const float* theta     = nullptr;
  int64_t n_qrows        = 0;
  const key_source* keys = nullptr;
};

/**
 * One batch of queries: qf as float (normalized for cosine), which the screening rows lift, and
 * q_exact, the queries as given, from which the exact scores are computed (qf for l2 and inner).
 */
template <typename T, score_kind S>
void search_batch(raft::resources const& res,
                  search_params const& params,
                  index<T> const& idx,
                  const float* qf,
                  const float* q_exact,
                  int64_t B,
                  int k,
                  int64_t* neighbors,
                  float* distances,
                  Filter filter,
                  const route_plan* route_in)
{
  cudaStream_t stream     = raft::resource::get_cuda_stream(res).get();
  const route_plan* route = route_in;
  const bool local        = idx.local_projections() && route_in == nullptr;
  const int64_t d = idx.dim(), n = idx.size();
  const int m = idx.width(), L = m + 1, MP = idx.row_length();
  const double g = gamma_of(MP), eps = eps_of(g), alpha = idx.alpha;
  const auto& M       = idx.maxima;
  const double m_nu   = (1 + std::ldexp(1.0, -10)) * std::max(M.norm, M.lifted);
  const double slack  = S == score_kind::cosine ? cosine_slack(d) : 0.0;
  const double c_norm = [&] {
    std::vector<float> c(d);
    raft::copy(c.data(), idx.center().data_handle(), d, stream);
    raft::resource::sync_stream(res);
    double s = 0;
    for (auto v : c) {
      s += double(v) * v;
    }
    return std::sqrt(s);
  }();

  stage_ranges stages;
  // query rows
  rmm::device_uvector<half> q_rows(B * MP, stream);
  rmm::device_uvector<double> aqh(B, stream), q_norm(B, stream);
  rmm::device_uvector<float> margin(B, stream), theta(B, stream);
  rmm::device_uvector<double> u(B, stream);
  device_source<float> qsrc{qf, B, d};
  for_lifted_chunks(
    res,
    idx,
    qsrc,
    B,
    false,
    [&](int64_t i, int64_t cn, const float*, const float*, float* z, float* r, float* a) {
      encode_queries<<<blocks_for(cn * 32), 256, 0, stream>>>(z,
                                                              r,
                                                              a,
                                                              m,
                                                              cn,
                                                              MP,
                                                              float(1.0 / alpha),
                                                              float(1.0 - eps),
                                                              float(g),
                                                              float(M.norm),
                                                              float(M.error),
                                                              float(M.low),
                                                              q_rows.data() + i * MP,
                                                              aqh.data() + i,
                                                              margin.data() + i);
      RAFT_CUDA_TRY(cudaGetLastError());
    });
  row_norms<<<blocks_for(B * 32), 256, 0, stream>>>(qf, d, B, q_norm.data());
  RAFT_CUDA_TRY(cudaGetLastError());
  stages.mark(0);

  // a shard under a threshold that all shards share (certex_shards.cu) needs no threshold stage of
  // its own: the shared threshold is the smallest over the shards, its own included
  threshold_hooks* hooks = threshold_hooks_sink();
  const bool given_u     = hooks && hooks->u_cap;
  if (!given_u) {
    // threshold from the anchors: the best anchor of every group of kTileN anchors, then the best
    // groups, then the k-th smallest exact score among them
    const int64_t n_anchors = idx.n_anchors();
    const int64_t groups    = (n_anchors + kTileN - 1) / kTileN;
    // at least 2k anchors, so that the k-th smallest anchor score exists and stays tight
    const int per_query = int(std::min<int64_t>(std::max(params.anchors, 2 * k), groups));
    rmm::device_uvector<double> a_scores(std::max<int64_t>(1, B * per_query), stream);
    if (per_query >= 1 && n_anchors >= k) {
      rmm::device_uvector<float> g_val(B * groups, stream);
      rmm::device_uvector<int32_t> g_id(B * groups, stream);
      anchor_groups(q_rows.data(),
                    int(B),
                    idx.anchor_rows().data_handle(),
                    int(n_anchors),
                    MP,
                    g_val.data(),
                    g_id.data(),
                    filter,
                    idx.anchor_stride(),
                    stream);
      auto top_val = raft::make_device_matrix<float, int64_t>(res, B, per_query);
      auto top_id  = raft::make_device_matrix<int, int64_t>(res, B, per_query);
      cuvs::selection::select_k(
        res,
        raft::make_device_matrix_view<const float, int64_t>(g_val.data(), B, groups),
        std::make_optional(
          raft::make_device_matrix_view<const int, int64_t>(g_id.data(), B, groups)),
        top_val.view(),
        top_id.view(),
        false);
      stages.mark(1);
      anchor_scores<T, S>
        <<<blocks_for(B * per_query * 32), 256, 0, stream>>>(q_exact,
                                                             idx.anchor_vectors().data_handle(),
                                                             d,
                                                             top_id.data_handle(),
                                                             int(B),
                                                             per_query,
                                                             idx.anchor_stride(),
                                                             n,
                                                             idx.removed().data_handle(),
                                                             filter,
                                                             a_scores.data());
      RAFT_CUDA_TRY(cudaGetLastError());
      anchor_threshold<S><<<unsigned(B), 256, pow2_at_least(per_query) * sizeof(double), stream>>>(
        a_scores.data(),
        per_query,
        k,
        aqh.data(),
        margin.data(),
        q_norm.data(),
        alpha,
        2 * m_nu * alpha + c_norm,
        m_nu * alpha,
        slack,
        theta.data(),
        u.data());
    } else {  // too few anchors: a bound that every stored vector satisfies
      RAFT_CUDA_TRY(cudaMemsetAsync(a_scores.data(), 0x7f, sizeof(double), stream));
      anchor_threshold<S><<<unsigned(B), 128, sizeof(double), stream>>>(a_scores.data(),
                                                                        0,
                                                                        k,
                                                                        aqh.data(),
                                                                        margin.data(),
                                                                        q_norm.data(),
                                                                        alpha,
                                                                        2 * m_nu * alpha + c_norm,
                                                                        m_nu * alpha,
                                                                        slack,
                                                                        theta.data(),
                                                                        u.data());
    }
    RAFT_CUDA_TRY(cudaGetLastError());
    // an index with lists: the nearest lists of every query, screened first, lower its threshold
    if (idx.n_lists() > 0 && params.probe_lists > 0 && route_in == nullptr && !local) {
      batch_state st{B,
                     k,
                     MP,
                     L,
                     q_rows.data(),
                     aqh.data(),
                     margin.data(),
                     q_norm.data(),
                     theta.data(),
                     u.data(),
                     g,
                     m_nu,
                     M.high,
                     alpha,
                     slack,
                     m_nu * alpha};
      probe_lists<T, S>(res, idx, params.probe_lists, qf, q_exact, st, nullptr, filter);
    }
  } else {
    RAFT_CUDA_TRY(cudaMemsetAsync(u.data(), 0x7f, B * sizeof(double), stream));  // +huge
  }

  // shards (certex_shards.cu): report U, or lower it to the threshold shared by all shards
  if (auto* h = hooks) {
    if (h->u_cap) {
      cap_threshold<<<blocks_for(B), 256, 0, stream>>>(
        u.data(), h->u_cap + h->b0, aqh.data(), margin.data(), alpha, B, theta.data());
      RAFT_CUDA_TRY(cudaGetLastError());
    }
    if (h->u_out) {
      raft::copy(h->u_out + h->b0, u.data(), B, stream);
      return;
    }
  }
  // local projections (local.cuh): the query rows of every list, their thresholds from U, and a
  // routed pass that pairs every list's query rows with its rows; later vectors form the tail
  const int P = local ? idx.n_lists() : 0;
  rmm::device_uvector<half> lq_rows(int64_t(P) * B * MP, stream);
  rmm::device_uvector<double> laqh(int64_t(P) * B, stream), list_consts(3 * P, stream);
  rmm::device_uvector<float> lmargin(int64_t(P) * B, stream), ltheta(int64_t(P) * B, stream);
  rmm::device_uvector<route_group> lgroups(P, stream);
  key_source local_keys{};
  route_plan local_route{};
  if (local) {
    encode_local_queries(res, idx, qf, B, lq_rows.data(), laqh.data(), lmargin.data());
    std::vector<double> h(3 * P);
    for (int j = 0; j < P; ++j) {
      const auto& Mj = idx.local_maxima()[j];
      h[j]           = idx.local_alpha()[j];
      h[P + j]       = (1 + std::ldexp(1.0, -10)) * std::max(Mj.norm, Mj.lifted);
      h[2 * P + j]   = Mj.high;
    }
    raft::copy(list_consts.data(), h.data(), 3 * P, stream);
    local_thresholds<<<blocks_for(int64_t(P) * B), 256, 0, stream>>>(
      u.data(), laqh.data(), lmargin.data(), list_consts.data(), B, int64_t(P) * B, ltheta.data());
    RAFT_CUDA_TRY(cudaGetLastError());
    // the nearest lists of every query, screened first in their own frames, lower its threshold
    if (!given_u && params.probe_lists > 0) {
      batch_state st{B,
                     k,
                     MP,
                     L,
                     q_rows.data(),
                     aqh.data(),
                     margin.data(),
                     q_norm.data(),
                     theta.data(),
                     u.data(),
                     g,
                     m_nu,
                     M.high,
                     alpha,
                     slack,
                     m_nu * alpha};
      const local_probe lp{
        lq_rows.data(), laqh.data(), lmargin.data(), ltheta.data(), list_consts.data()};
      probe_lists<T, S>(res, idx, params.probe_lists, qf, q_exact, st, &lp, filter);
      local_thresholds<<<blocks_for(int64_t(P) * B), 256, 0, stream>>>(u.data(),
                                                                       laqh.data(),
                                                                       lmargin.data(),
                                                                       list_consts.data(),
                                                                       B,
                                                                       int64_t(P) * B,
                                                                       ltheta.data());
      RAFT_CUDA_TRY(cudaGetLastError());
    }
    const auto& off = idx.list_offsets();
    std::vector<route_group> groups;
    int tiles = 0;
    for (int j = 0; j < P; ++j) {
      const int64_t nx = off[j + 1] - off[j];
      if (nx == 0) { continue; }
      route_group gr;
      gr.q0      = int32_t(int64_t(j) * B);
      gr.nq      = int32_t(B);
      gr.x0      = int32_t(off[j]);
      gr.nx      = int32_t(nx);
      gr.tile0   = tiles;
      gr.x_tiles = int32_t((nx + kTileN - 1) / kTileN);
      tiles += int((B + kTileM - 1) / kTileM) * gr.x_tiles;
      groups.push_back(gr);
    }
    RAFT_CUDA_TRY(cudaMemcpyAsync(lgroups.data(),
                                  groups.data(),
                                  groups.size() * sizeof(route_group),
                                  cudaMemcpyHostToDevice,
                                  stream));
    local_keys.q_rows      = lq_rows.data();
    local_keys.aqh         = laqh.data();
    local_keys.margin      = lmargin.data();
    local_keys.list_alpha  = list_consts.data();
    local_keys.list_m_nu   = list_consts.data() + P;
    local_keys.list_m_high = list_consts.data() + 2 * P;
    local_keys.period      = B;
    local_route.rows       = idx.rows().data_handle();
    local_route.n_rows     = off[P];
    local_route.ids        = nullptr;
    local_route.groups     = lgroups.data();
    local_route.n_groups   = int(groups.size());
    local_route.n_tiles    = tiles;
    local_route.tail       = filter;
    local_route.q_rows     = lq_rows.data();
    local_route.theta      = ltheta.data();
    local_route.n_qrows    = int64_t(P) * B;
    local_route.keys       = &local_keys;
    route                  = &local_route;
    raft::resource::sync_stream(res);  // the host groups go out of scope
  }
  stages.mark(2);

  // int8 rows (int8.cuh): the query codes with h_q from U, and the base of the survivor bounds
  const bool int8 = idx.int8_rows();
  rmm::device_uvector<int8_t> q_codes(int8 ? B * idx.code_length() : 0, stream);
  rmm::device_uvector<float> q_scal(int8 ? B * 5 : 0, stream);
  rmm::device_uvector<double> lb_base(int8 ? B : 0, stream);
  if (int8) {
    device_source<float> q8{qf, B, d};
    for_lifted_chunks(
      res,
      idx,
      q8,
      B,
      false,
      [&](int64_t i, int64_t cn, const float*, const float*, float* z, float* r, float* a) {
        encode_int8_queries<<<blocks_for(cn * 32), 256, 0, stream>>>(
          z,
          r,
          a,
          m,
          cn,
          idx.code_length(),
          u.data() + i,
          idx.max_a,
          slack,
          q_codes.data() + i * idx.code_length(),
          q_scal.data() + i * 5,
          lb_base.data() + i);
        RAFT_CUDA_TRY(cudaGetLastError());
      });
  }

  // certified pass into per-block segments; repeat with larger segments on an overflow. With a
  // label partition, the routed pass screens the partitioned rows (segments [0, nseg_routed)) and
  // the label filter the rows added after it (the tail)
  const int64_t n_tail  = route ? n - route->n_rows : n;
  const int nseg_routed = route ? routed_segments(route->n_tiles) : 0;
  const int nseg_tail   = int8         ? int8_segments(int(B), n)
                          : n_tail > 0 ? screen_segments(int(B), n_tail, MP)
                                       : 0;
  const int nseg        = std::max(1, nseg_routed + nseg_tail);
  uint32_t seg_cap      = uint32_t(std::max<int64_t>(4096, B * 20000 / nseg));
  rmm::device_uvector<uint32_t> seg_count(nseg, stream), seg_offset(nseg, stream);
  rmm::device_uvector<int32_t> seg_q(0, stream), seg_x(0, stream);
  rmm::device_uvector<float> seg_v(0, stream);
  std::vector<uint32_t> h_count(nseg);
  for (;;) {
    seg_q.resize(int64_t(seg_cap) * nseg, stream);
    seg_x.resize(int64_t(seg_cap) * nseg, stream);
    seg_v.resize(int64_t(seg_cap) * nseg, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(seg_count.data(), 0, nseg * sizeof(uint32_t), stream));
    if (int8) {
      screen_int8(q_codes.data(),
                  q_scal.data(),
                  int(B),
                  idx.codes().data_handle(),
                  idx.row_scalars().data_handle(),
                  n,
                  idx.code_length(),
                  seg_q.data(),
                  seg_x.data(),
                  seg_v.data(),
                  seg_count.data(),
                  seg_cap,
                  filter,
                  stream);
    } else if (route) {
      screen_routed(route->q_rows ? route->q_rows : q_rows.data(),
                    int(route->q_rows ? route->n_qrows : B),
                    route->rows,
                    route->n_rows,
                    MP,
                    route->theta ? route->theta : theta.data(),
                    route->groups,
                    route->n_groups,
                    route->n_tiles,
                    seg_q.data(),
                    seg_x.data(),
                    seg_v.data(),
                    seg_count.data(),
                    seg_cap,
                    stream);
    }
    if (n_tail > 0 && !int8) {
      const int64_t o = int64_t(nseg_routed) * seg_cap;
      screen(q_rows.data(),
             int(B),
             idx.rows().data_handle() + (n - n_tail) * MP,
             n_tail,
             MP,
             theta.data(),
             seg_q.data() + o,
             seg_x.data() + o,
             seg_v.data() + o,
             seg_count.data() + nseg_routed,
             seg_cap,
             route ? route->tail : filter,
             stream);
    }
    raft::copy(h_count.data(), seg_count.data(), nseg, stream);
    raft::resource::sync_stream(res);
    const uint32_t worst = *std::max_element(h_count.begin(), h_count.end());
    if (worst <= seg_cap) { break; }
    if (int64_t(seg_cap) * nseg >= (int64_t(1) << 31)) {
      throw batch_too_large("too many survivors in one batch; search with a smaller batch_size");
    }
    seg_cap = uint32_t(std::min<int64_t>(int64_t(worst) * 2, ((int64_t(1) << 31) - 1) / nseg));
  }
  std::vector<uint32_t> h_offset(nseg);
  int64_t total = 0;
  for (int i = 0; i < nseg; ++i) {
    h_offset[i] = uint32_t(total);
    total += h_count[i];
  }
  raft::copy(seg_offset.data(), h_offset.data(), nseg, stream);
  stages.mark(3);

  // certified lower bounds and sort keys (query, bound), sorted by query and then by bound
  rmm::device_uvector<uint64_t> keys(std::max<int64_t>(1, total), stream),
    keys_sorted(std::max<int64_t>(1, total), stream);
  rmm::device_uvector<int32_t> ids(std::max<int64_t>(1, total), stream),
    ids_sorted(std::max<int64_t>(1, total), stream);
  // the routed pass (label partition: rows map to the vector rows; local projections: per-list
  // query rows and constants) and the plain pass (the batch's own query rows)
  const key_source plain{q_rows.data(), aqh.data(), margin.data(), alpha, m_nu, M.high};
  key_source routed = route && route->keys ? *route->keys : plain;
  if (route && !route->keys) { routed.ids = route->ids; }
  key_source rest = plain;
  rest.base       = route ? int32_t(route->n_rows) : 0;
  if (int8) { rest.lb_base = lb_base.data(); }
  survivor_keys<<<unsigned(nseg), 256, 0, stream>>>(seg_q.data(),
                                                    seg_x.data(),
                                                    seg_v.data(),
                                                    seg_count.data(),
                                                    seg_offset.data(),
                                                    seg_cap,
                                                    MP,
                                                    L,
                                                    g,
                                                    slack,
                                                    route ? nseg_routed : 0,
                                                    routed,
                                                    rest,
                                                    keys.data(),
                                                    ids.data());
  RAFT_CUDA_TRY(cudaGetLastError());
  stages.mark(4);
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
  RAFT_CUDA_TRY(cudaGetLastError());
  stages.mark(5);

  // ordered verification and the output
  rmm::device_uvector<double> best_score(B * k, stream);
  rmm::device_uvector<int64_t> best_id(B * k, stream);
  search_stats* stats = stats_sink();
  rmm::device_uvector<unsigned long long> verified(stats ? 1 : 0, stream);
  if (stats) {
    RAFT_CUDA_TRY(cudaMemsetAsync(verified.data(), 0, sizeof(unsigned long long), stream));
  }
  auto verify = [&](auto kpl) {
    verify_rounds<T, S, decltype(kpl)::value>(res,
                                              keys_sorted.data(),
                                              ids_sorted.data(),
                                              start.data(),
                                              q_exact,
                                              idx.vectors_ptr(),
                                              d,
                                              int(B),
                                              k,
                                              idx.removed().data_handle(),
                                              best_score.data(),
                                              best_id.data(),
                                              stats ? verified.data() : nullptr,
                                              // the local pass screens without the filter
                                              local ? filter : Filter{});
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
  RAFT_CUDA_TRY(cudaGetLastError());
  if (stats) {
    unsigned long long h = 0;
    raft::copy(&h, verified.data(), 1, stream);
    raft::resource::sync_stream(res);
    stats->queries += B;
    stats->survivors += total;
    stats->verified += int64_t(h);
  }
  stages.mark(6);
  finish<S>
    <<<blocks_for(B * k), 256, 0, stream>>>(best_score.data(),
                                            best_id.data(),
                                            q_norm.data(),
                                            B,
                                            k,
                                            idx.mapped() ? idx.row_ids().data_handle() : nullptr,
                                            distances,
                                            neighbors);
  RAFT_CUDA_TRY(cudaGetLastError());
  stages.mark(7);
}

/** One batch of B queries qb into neighbors and distances [B, k]; the caller holds the view. */
template <typename T>
void search_one_batch(raft::resources const& res,
                      search_params const& params,
                      index<T> const& idx,
                      const T* qb,
                      int64_t B,
                      int k,
                      int64_t* neighbors,
                      float* distances,
                      Filter filter,
                      const route_plan* route)
{
  const bool cosine = idx.metric() == DistanceType::CosineExpanded;
  auto qf           = to_float(res, qb, B, idx.dim(), cosine);
  // cosine: the exact scores use the queries as given, the screening rows their normalized copies
  auto q_given = to_float(res, qb, cosine ? B : 0, idx.dim(), false);
  auto run     = [&](auto kind) {
    search_batch<T, decltype(kind)::value>(res,
                                           params,
                                           idx,
                                           qf.data(),
                                           cosine ? q_given.data() : qf.data(),
                                           B,
                                           k,
                                           neighbors,
                                           distances,
                                           filter,
                                           route);
  };
  switch (idx.metric()) {
    case DistanceType::InnerProduct:
      run(std::integral_constant<score_kind, score_kind::inner>{});
      break;
    case DistanceType::CosineExpanded:
      run(std::integral_constant<score_kind, score_kind::cosine>{});
      break;
    default: run(std::integral_constant<score_kind, score_kind::l2>{});
  }
}

template <typename T>
void check_search(index<T> const& idx,
                  int64_t nq,
                  int64_t dim,
                  raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
                  raft::device_matrix_view<float, int64_t, raft::row_major> distances)
{
  const int64_t k = neighbors.extent(1);
  RAFT_EXPECTS(dim == idx.dim(), "dimension mismatch");
  RAFT_EXPECTS(neighbors.extent(0) == nq && distances.extent(0) == nq && distances.extent(1) == k,
               "output shapes do not match the queries");
  RAFT_EXPECTS(k >= 1 && k <= 2048, "certex returns at most 2048 neighbors per query");
  RAFT_EXPECTS(idx.size() >= k, "the index holds fewer than k vectors");
}

template <typename T>
void search(raft::resources const& res,
            search_params const& params,
            index<T> const& idx,
            raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            Filter filter)
{
  const int64_t nq = queries.extent(0);
  const int k      = int(neighbors.extent(1));
  cuvs::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "certex::search(%zu, %d)", size_t(nq), k);
  check_search(idx, nq, queries.extent(1), neighbors, distances);
  const int64_t bs = std::max(1, params.batch_size);
  // A batch whose survivors do not fit in device memory (or in the 32-bit segments) is split in
  // halves, which return the same exact answers; every temporary of the failed attempt is freed.
  std::function<void(int64_t, int64_t)> run = [&](int64_t b0, int64_t B) {
    Filter batch_filter = filter;
    batch_filter.q0     = b0;
    if (auto* h = threshold_hooks_sink()) { h->b0 = b0; }
    try {
      search_one_batch(res,
                       params,
                       idx,
                       queries.data_handle() + b0 * idx.dim(),
                       B,
                       k,
                       neighbors.data_handle() + b0 * k,
                       distances.data_handle() + b0 * k,
                       batch_filter,
                       nullptr);
    } catch (rmm::bad_alloc const&) {
      if (B < 2) { throw; }
      raft::resource::sync_stream(res);
      run(b0, B / 2);
      run(b0 + B / 2, B - B / 2);
    } catch (batch_too_large const&) {
      if (B < 2) { throw; }
      run(b0, B / 2);
      run(b0 + B / 2, B - B / 2);
    }
  };
  for (int64_t b0 = 0; b0 < nq; b0 += bs) {
    // the vectors and certificate maxima of this batch stay fixed while updates proceed
    std::shared_lock batch_view(idx.sync().rw);
    run(b0, std::min(bs, nq - b0));
  }
}

/** Rows of `width` units: dst[i] = src[order[i]] (gather) or dst[order[i]] = src[i] (scatter). */
template <typename U, bool kScatter>
__global__ void permute_rows(const U* src, const int32_t* order, int64_t n, int64_t width, U* dst)
{
  for (int64_t t = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; t < n * width;
       t += int64_t(gridDim.x) * blockDim.x) {
    const int64_t i = t / width, j = t % width, o = order[i];
    if constexpr (kScatter) {
      dst[o * width + j] = src[t];
    } else {
      dst[t] = src[o * width + j];
    }
  }
}

/** First position of every label l <= n_labels in labels sorted ascending (binary search). */
__global__ void label_starts(const int32_t* sorted, int64_t n, int n_labels, int64_t* start)
{
  const int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (l > n_labels) { return; }
  int64_t lo = 0, hi = n;
  while (lo < hi) {
    const int64_t mid = (lo + hi) / 2;
    if (sorted[mid] < l) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  start[l] = lo;
}

template <typename T>
void partition_by_labels(raft::resources const& res, index<T>* idx)
{
  cuvs::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "certex::partition_by_labels(%zu)", size_t(idx->size()));
  RAFT_EXPECTS(idx->has_labels(), "the index has no labels (set_labels)");
  RAFT_EXPECTS(!idx->local_projections(),
               "label partitions need the global projection (local_projections = false)");
  RAFT_EXPECTS(!idx->int8_rows(), "label partitions need half-precision rows (int8_rows = false)");
  std::lock_guard writer(idx->sync().writer);
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  const int64_t n     = idx->size();
  const int MP        = idx->row_length();
  RAFT_EXPECTS(n < (int64_t(1) << 31), "at most 2^31 - 1 vectors per partition");
  rmm::device_uvector<int32_t> keys(n, stream), keys_sorted(n, stream), ids(n, stream);
  auto part_ids = raft::make_device_vector<int32_t, int64_t>(res, n);
  raft::copy(keys.data(), idx->labels().data_handle(), n, stream);
  iota_from<<<blocks_for(n), 256, 0, stream>>>(ids.data(), n);
  RAFT_CUDA_TRY(cudaGetLastError());
  size_t tmp_bytes = 0;  // radix sort is stable: the ids of a label stay ascending
  RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(nullptr,
                                                tmp_bytes,
                                                keys.data(),
                                                keys_sorted.data(),
                                                ids.data(),
                                                part_ids.data_handle(),
                                                n,
                                                0,
                                                32,
                                                stream));
  rmm::device_buffer tmp(tmp_bytes, stream);
  RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(tmp.data(),
                                                tmp_bytes,
                                                keys.data(),
                                                keys_sorted.data(),
                                                ids.data(),
                                                part_ids.data_handle(),
                                                n,
                                                0,
                                                32,
                                                stream));
  int32_t max_label = -1;
  raft::copy(&max_label, keys_sorted.data() + n - 1, 1, stream);
  raft::resource::sync_stream(res);
  const int n_labels = max_label + 1;  // unlabeled vectors (-1) sort first and match no query
  rmm::device_uvector<int64_t> starts(n_labels + 1, stream);
  label_starts<<<blocks_for(n_labels + 1), 256, 0, stream>>>(
    keys_sorted.data(), n, n_labels, starts.data());
  RAFT_CUDA_TRY(cudaGetLastError());
  std::vector<int64_t> offsets(n_labels + 1);
  raft::copy(offsets.data(), starts.data(), n_labels + 1, stream);
  auto rows = raft::make_device_matrix<half, int64_t>(res, n, MP);
  // 16-byte units: a row of MP halves is a multiple of 128 bytes
  permute_rows<int4, false>
    <<<blocks_for(std::min<int64_t>(n * MP / 8, int64_t(1) << 24)), 256, 0, stream>>>(
      reinterpret_cast<const int4*>(idx->rows().data_handle()),
      part_ids.data_handle(),
      n,
      MP / 8,
      reinterpret_cast<int4*>(rows.data_handle()));
  RAFT_CUDA_TRY(cudaGetLastError());
  raft::resource::sync_stream(res);
  std::unique_lock publish(idx->sync().rw);
  idx->set_partition(std::move(rows), std::move(part_ids), std::move(offsets), n);
}

/**
 * Filtered search by label over a label partition: every batch is sorted by label, the routed
 * pass pairs each label's queries with its partitioned rows, the label filter screens the rows
 * added after the partition, and the results return to the order of the queries.
 */
template <typename T>
void search_by_labels(raft::resources const& res,
                      search_params const& params,
                      index<T> const& idx,
                      raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
                      raft::device_vector_view<const int32_t, int64_t> query_labels,
                      raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
                      raft::device_matrix_view<float, int64_t, raft::row_major> distances)
{
  const int64_t nq = queries.extent(0), d = idx.dim();
  const int k = int(neighbors.extent(1));
  cuvs::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "certex::search_by_labels(%zu, %d)", size_t(nq), k);
  check_search(idx, nq, queries.extent(1), neighbors, distances);
  RAFT_EXPECTS(idx.has_labels(), "the index has no labels (set_labels)");
  RAFT_EXPECTS(query_labels.extent(0) == nq, "one label per query");
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  const int64_t bs    = std::max(1, params.batch_size);
  std::vector<int32_t> h_lab(nq);
  raft::copy(h_lab.data(), query_labels.data_handle(), nq, stream);
  raft::resource::sync_stream(res);
  for (int64_t b0 = 0; b0 < nq; b0 += bs) {
    const int64_t B = std::min(bs, nq - b0);
    std::shared_lock batch_view(idx.sync().rw);
    Filter filter;
    filter.kind         = Filter::kLabels;
    filter.row_labels   = idx.labels().data_handle();
    filter.query_labels = query_labels.data_handle();
    filter.q0           = b0;
    if (idx.n_partitioned() == 0) {  // no partition: the label filter over every row
      search_one_batch(res,
                       params,
                       idx,
                       queries.data_handle() + b0 * d,
                       B,
                       k,
                       neighbors.data_handle() + b0 * k,
                       distances.data_handle() + b0 * k,
                       filter,
                       nullptr);
      continue;
    }
    // the queries of the batch in label order, and one routed group per label with rows
    std::vector<int32_t> order(B), sorted_lab(B);
    for (int64_t i = 0; i < B; ++i) {
      order[i] = int32_t(i);
    }
    std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
      return h_lab[b0 + a] < h_lab[b0 + b];
    });
    const auto& off = idx.label_offsets();
    std::vector<route_group> groups;
    int tiles = 0;
    for (int64_t i = 0; i < B;) {
      const int32_t l = h_lab[b0 + order[i]];
      int64_t j       = i;
      while (j < B && h_lab[b0 + order[j]] == l) {
        sorted_lab[j] = l;
        ++j;
      }
      if (l >= 0 && l + 1 < int64_t(off.size()) && off[l + 1] > off[l]) {
        route_group g;
        g.q0      = int32_t(i);
        g.nq      = int32_t(j - i);
        g.x0      = int32_t(off[l]);
        g.nx      = int32_t(off[l + 1] - off[l]);
        g.tile0   = tiles;
        g.x_tiles = (g.nx + kTileN - 1) / kTileN;
        tiles += ((g.nq + kTileM - 1) / kTileM) * g.x_tiles;
        groups.push_back(g);
      }
      i = j;
    }
    rmm::device_uvector<int32_t> d_order(B, stream), d_lab(B, stream);
    rmm::device_uvector<route_group> d_groups(std::max<size_t>(1, groups.size()), stream);
    raft::copy(d_order.data(), order.data(), B, stream);
    raft::copy(d_lab.data(), sorted_lab.data(), B, stream);
    if (!groups.empty()) {
      RAFT_CUDA_TRY(cudaMemcpyAsync(d_groups.data(),
                                    groups.data(),
                                    groups.size() * sizeof(route_group),
                                    cudaMemcpyHostToDevice,
                                    stream));
    }
    rmm::device_uvector<T> q_sorted(B * d, stream);
    rmm::device_uvector<int64_t> nb_sorted(B * k, stream);
    rmm::device_uvector<float> ds_sorted(B * k, stream);
    permute_rows<T, false>
      <<<blocks_for(std::min<int64_t>(B * d, int64_t(1) << 24)), 256, 0, stream>>>(
        queries.data_handle() + b0 * d, d_order.data(), B, d, q_sorted.data());
    RAFT_CUDA_TRY(cudaGetLastError());
    Filter sorted_filter       = filter;  // queries in label order, from the first of the batch
    sorted_filter.query_labels = d_lab.data();
    sorted_filter.q0           = 0;
    route_plan route;
    route.rows            = idx.partition_rows().data_handle();
    route.n_rows          = idx.n_partitioned();
    route.ids             = idx.partition_ids().data_handle();
    route.groups          = d_groups.data();
    route.n_groups        = int(groups.size());
    route.n_tiles         = tiles;
    route.tail            = sorted_filter;
    route.tail.row_labels = idx.labels().data_handle() + idx.n_partitioned();
    search_one_batch(res,
                     params,
                     idx,
                     q_sorted.data(),
                     B,
                     k,
                     nb_sorted.data(),
                     ds_sorted.data(),
                     sorted_filter,
                     &route);
    permute_rows<int64_t, true>
      <<<blocks_for(std::min<int64_t>(B * k, int64_t(1) << 24)), 256, 0, stream>>>(
        nb_sorted.data(), d_order.data(), B, k, neighbors.data_handle() + b0 * k);
    permute_rows<float, true>
      <<<blocks_for(std::min<int64_t>(B * k, int64_t(1) << 24)), 256, 0, stream>>>(
        ds_sorted.data(), d_order.data(), B, k, distances.data_handle() + b0 * k);
    RAFT_CUDA_TRY(cudaGetLastError());
    raft::resource::sync_stream(res);  // the host vectors of this batch go out of scope
  }
}

/**
 * Bits of a bitset (rows = 1) or bitmap over ids, rearranged by row: bit (q, row) of the result is
 * bit (q, row_ids[row]) of the input, whose rows hold n_cols bits. One thread per 32-bit word.
 */
__global__ void bits_by_row(const uint32_t* bits,
                            int64_t n_cols,
                            int64_t rows,
                            const int32_t* row_ids,
                            int64_t n,
                            uint32_t* out)
{
  const int64_t w = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (w >= (rows * n + 31) / 32) { return; }
  uint32_t word = 0;
  for (int b = 0; b < 32; ++b) {
    const int64_t t = w * 32 + b;
    if (t >= rows * n) { break; }
    const int64_t q = t / n, row = t % n;
    const int64_t i = q * n_cols + row_ids[row];
    word |= ((bits[i >> 5] >> (i & 31)) & 1u) << b;
  }
  out[w] = word;
}

/** The kernel filter of a cuVS sample filter (none, bitset over the rows, or bitmap per query). */
template <typename T>
auto to_kernel_filter(raft::resources const& res,
                      index<T> const& idx,
                      int64_t n_queries,
                      const cuvs::neighbors::filtering::base_filter& f,
                      rmm::device_uvector<uint32_t>& row_bits) -> Filter
{
  using cuvs::neighbors::filtering::FilterType;
  Filter out;
  switch (f.get_filter_type()) {
    case FilterType::None: break;
    case FilterType::Bitset: {
      const auto& b =
        dynamic_cast<const cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>&>(f);
      RAFT_EXPECTS(b.view().size() >= idx.size(), "the bitset must cover every vector");
      out.kind = Filter::kBitset;
      out.bits = b.view().data();
      break;
    }
    case FilterType::Bitmap: {
      const auto& b =
        dynamic_cast<const cuvs::neighbors::filtering::bitmap_filter<uint32_t, int64_t>&>(f);
      RAFT_EXPECTS(b.view().get_n_rows() == n_queries && b.view().get_n_cols() >= idx.size(),
                   "the bitmap must hold one row per query and one column per vector");
      out.kind   = Filter::kBitmap;
      out.bits   = b.view().data();
      out.n_cols = b.view().get_n_cols();
      break;
    }
    default: RAFT_FAIL("certex supports no filter, a bitset filter, or a bitmap filter");
  }
  if (idx.mapped() && out.kind != Filter::kNone) {  // the bits of ids, rearranged by row
    cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
    const int64_t n = idx.size(), rows = out.kind == Filter::kBitmap ? n_queries : 1;
    const int64_t words = (rows * n + 31) / 32;
    row_bits.resize(words, stream);
    bits_by_row<<<blocks_for(words), 256, 0, stream>>>(
      out.bits, out.n_cols, rows, idx.row_ids().data_handle(), n, row_bits.data());
    RAFT_CUDA_TRY(cudaGetLastError());
    out.bits   = row_bits.data();
    out.n_cols = n;
  }
  return out;
}

/** out[id_rows[first + i]] = in[i]: values of the ids first, first + 1, ... stored by row. */
__global__ void scatter_by_id(
  const int32_t* in, int64_t n, int64_t first, const int32_t* id_rows, int32_t* out)
{
  const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (i < n) { out[id_rows[first + i]] = in[i]; }
}

template <typename T>
void set_labels(raft::resources const& res,
                raft::device_vector_view<const int32_t, int64_t> labels,
                index<T>* idx,
                int64_t first)
{
  RAFT_EXPECTS(first >= 0 && first + labels.extent(0) <= idx->size(),
               "labels must address stored vectors");
  std::lock_guard writer(idx->sync().writer);
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  if (idx->mapped()) {  // labels are stored by row
    scatter_by_id<<<blocks_for(labels.extent(0)), 256, 0, stream>>>(labels.data_handle(),
                                                                    labels.extent(0),
                                                                    first,
                                                                    idx->id_rows().data_handle(),
                                                                    idx->labels().data_handle());
    RAFT_CUDA_TRY(cudaGetLastError());
  } else {
    raft::copy(idx->labels().data_handle() + first, labels.data_handle(), labels.extent(0), stream);
  }
  raft::resource::sync_stream(res);
  std::unique_lock publish(idx->sync().rw);
  idx->mark_labeled();
  if (first < idx->n_partitioned()) { idx->drop_partition(); }  // labels of sorted rows changed
  raft::resource::sync_stream(res);
}

}  // namespace detail

template <typename T>
index<T>::index(raft::resources const& res,
                cuvs::distance::DistanceType metric,
                int64_t dim,
                int width,
                int anchor_stride,
                bool host_vectors)
  : cuvs::neighbors::index(),
    metric_(metric),
    dim_(dim),
    width_(width),
    row_length_(detail::row_length_of(width)),
    anchor_stride_(anchor_stride),
    rows_(raft::make_device_matrix<half, int64_t>(res, 0, row_length_)),
    anchor_rows_(raft::make_device_matrix<half, int64_t>(res, 0, row_length_)),
    center_(raft::make_device_vector<float, int64_t>(res, dim)),
    basis_(raft::make_device_matrix<float, int64_t>(res, width, dim)),
    host_vectors_(host_vectors),
    vectors_(raft::make_device_matrix<T, int64_t>(res, 0, dim)),
    host_rows_(raft::make_pinned_matrix<T, int64_t>(res, 0, dim)),
    anchor_vectors_(raft::make_device_matrix<T, int64_t>(res, 0, dim)),
    removed_(raft::make_device_vector<uint8_t, int64_t>(res, 0)),
    labels_(raft::make_device_vector<int32_t, int64_t>(res, 0)),
    partition_rows_(raft::make_device_matrix<half, int64_t>(res, 0, row_length_)),
    partition_ids_(raft::make_device_vector<int32_t, int64_t>(res, 0)),
    list_centers_(raft::make_device_matrix<float, int64_t>(res, 0, dim)),
    row_ids_(raft::make_device_vector<int32_t, int64_t>(res, 0)),
    id_rows_(raft::make_device_vector<int32_t, int64_t>(res, 0)),
    local_centers_(raft::make_device_matrix<float, int64_t>(res, 0, dim)),
    local_bases_(raft::make_device_matrix<float, int64_t>(res, 0, dim)),
    codes_(raft::make_device_matrix<int8_t, int64_t>(res, 0, (width + 63) / 64 * 64)),
    row_scal_(raft::make_device_matrix<float, int64_t>(res, 0, 5))
{
}

template <typename T>
void index<T>::reserve(raft::resources const& res, int64_t n)
{
  if (n <= capacity_) { return; }
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  const int64_t c     = std::max(n, 2 * capacity_);
  const int64_t ca    = (c + anchor_stride_ - 1) / anchor_stride_;
  // half-precision screening rows, or int8 codes with their scalars
  auto r  = raft::make_device_matrix<half, int64_t>(res, int8_ ? 0 : c, row_length_);
  auto a  = raft::make_device_matrix<half, int64_t>(res, ca, row_length_);
  auto av = raft::make_device_matrix<T, int64_t>(res, ca, dim_);
  if (!int8_) { raft::copy(r.data_handle(), rows_.data_handle(), size_ * row_length_, stream); }
  if (int8_) {
    auto cd = raft::make_device_matrix<int8_t, int64_t>(res, c, code_length());
    auto sc = raft::make_device_matrix<float, int64_t>(res, c, 5);
    raft::copy(cd.data_handle(), codes_.data_handle(), size_ * code_length(), stream);
    raft::copy(sc.data_handle(), row_scal_.data_handle(), size_ * 5, stream);
    codes_    = std::move(cd);
    row_scal_ = std::move(sc);
  }
  raft::copy(a.data_handle(), anchor_rows_.data_handle(), n_anchors() * row_length_, stream);
  raft::copy(av.data_handle(), anchor_vectors_.data_handle(), n_anchors() * dim_, stream);
  if (host_vectors_) {
    auto h = raft::make_pinned_matrix<T, int64_t>(res, c, dim_);
    raft::copy(h.data_handle(), host_rows_.data_handle(), size_ * dim_, stream);
    raft::resource::sync_stream(res);
    host_rows_ = std::move(h);
  } else {
    auto v = raft::make_device_matrix<T, int64_t>(res, c, dim_);
    raft::copy(v.data_handle(), vectors_.data_handle(), size_ * dim_, stream);
    vectors_ = std::move(v);
  }
  auto rm = raft::make_device_vector<uint8_t, int64_t>(res, (c + 3) / 4 * 4);
  RAFT_CUDA_TRY(cudaMemsetAsync(rm.data_handle(), 0, rm.size(), stream));
  raft::copy(
    rm.data_handle(), removed_.data_handle(), std::min<int64_t>(size_, removed_.size()), stream);
  auto lb = raft::make_device_vector<int32_t, int64_t>(res, c);
  RAFT_CUDA_TRY(cudaMemsetAsync(lb.data_handle(), 0xff, c * sizeof(int32_t), stream));  // -1
  raft::copy(
    lb.data_handle(), labels_.data_handle(), std::min<int64_t>(size_, labels_.size()), stream);
  if (row_ids_.size() > 0) {  // rows beyond the built ones are their own ids
    auto ri = raft::make_device_vector<int32_t, int64_t>(res, c);
    auto ir = raft::make_device_vector<int32_t, int64_t>(res, c);
    detail::iota_from<<<detail::blocks_for(c), 256, 0, stream>>>(ri.data_handle(), c);
    detail::iota_from<<<detail::blocks_for(c), 256, 0, stream>>>(ir.data_handle(), c);
    RAFT_CUDA_TRY(cudaGetLastError());
    raft::copy(ri.data_handle(), row_ids_.data_handle(), row_ids_.size(), stream);
    raft::copy(ir.data_handle(), id_rows_.data_handle(), id_rows_.size(), stream);
    row_ids_ = std::move(ri);
    id_rows_ = std::move(ir);
  }
  rows_           = std::move(r);
  anchor_rows_    = std::move(a);
  anchor_vectors_ = std::move(av);
  removed_        = std::move(rm);
  labels_         = std::move(lb);
  capacity_       = c;
}

#define CUVS_INST_CERTEX(T)                                                                     \
  template struct index<T>;                                                                     \
  auto build(raft::resources const& res,                                                        \
             index_params const& params,                                                        \
             raft::device_matrix_view<const T, int64_t, raft::row_major> dataset) -> index<T>   \
  {                                                                                             \
    return detail::build<T>(res, params, dataset);                                              \
  }                                                                                             \
  void extend(raft::resources const& res,                                                       \
              raft::device_matrix_view<const T, int64_t, raft::row_major> new_vectors,          \
              index<T>* idx)                                                                    \
  {                                                                                             \
    detail::extend<T>(res, new_vectors, idx);                                                   \
  }                                                                                             \
  auto build(raft::resources const& res,                                                        \
             index_params const& params,                                                        \
             raft::host_matrix_view<const T, int64_t, raft::row_major> dataset) -> index<T>     \
  {                                                                                             \
    return detail::build<T>(res, params, dataset);                                              \
  }                                                                                             \
  void extend(raft::resources const& res,                                                       \
              raft::host_matrix_view<const T, int64_t, raft::row_major> new_vectors,            \
              index<T>* idx)                                                                    \
  {                                                                                             \
    detail::extend<T>(res, new_vectors, idx);                                                   \
  }                                                                                             \
  void remove(raft::resources const& res,                                                       \
              raft::device_vector_view<const int64_t, int64_t> ids,                             \
              index<T>* idx)                                                                    \
  {                                                                                             \
    detail::remove<T>(res, ids, idx);                                                           \
  }                                                                                             \
  void set_labels(raft::resources const& res,                                                   \
                  raft::device_vector_view<const int32_t, int64_t> labels,                      \
                  index<T>* idx,                                                                \
                  int64_t first)                                                                \
  {                                                                                             \
    detail::set_labels<T>(res, labels, idx, first);                                             \
  }                                                                                             \
  void search(raft::resources const& res,                                                       \
              search_params const& params,                                                      \
              index<T> const& idx,                                                              \
              raft::device_matrix_view<const T, int64_t, raft::row_major> queries,              \
              raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,            \
              raft::device_matrix_view<float, int64_t, raft::row_major> distances,              \
              const cuvs::neighbors::filtering::base_filter& sample_filter)                     \
  {                                                                                             \
    rmm::device_uvector<uint32_t> row_bits(0, raft::resource::get_cuda_stream(res));            \
    detail::search<T>(                                                                          \
      res,                                                                                      \
      params,                                                                                   \
      idx,                                                                                      \
      queries,                                                                                  \
      neighbors,                                                                                \
      distances,                                                                                \
      detail::to_kernel_filter(res, idx, queries.extent(0), sample_filter, row_bits));          \
  }                                                                                             \
  void search(raft::resources const& res,                                                       \
              search_params const& params,                                                      \
              index<T> const& idx,                                                              \
              raft::device_matrix_view<const T, int64_t, raft::row_major> queries,              \
              raft::device_vector_view<const int32_t, int64_t> query_labels,                    \
              raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,            \
              raft::device_matrix_view<float, int64_t, raft::row_major> distances)              \
  {                                                                                             \
    detail::search_by_labels<T>(res, params, idx, queries, query_labels, neighbors, distances); \
  }                                                                                             \
  void partition_by_labels(raft::resources const& res, index<T>* idx)                           \
  {                                                                                             \
    detail::partition_by_labels<T>(res, idx);                                                   \
  }

CUVS_INST_CERTEX(float)
CUVS_INST_CERTEX(half)
CUVS_INST_CERTEX(int8_t)
CUVS_INST_CERTEX(uint8_t)

#undef CUVS_INST_CERTEX

}  // namespace cuvs::neighbors::certex
