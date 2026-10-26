/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// int8 screening rows of a CertEx index (index_params::int8_rows; the certified pass is in
// screen_int8.cu).
//
// Every lifted vector l(x) = [z_x; r_x] keeps its projection coordinates as int8 codes with one
// scale per row, z_x = s_x (c_x + e_x) with s_x = max_j |z_x,j| / 127 and |e_x,j| <= 1/2, and
// a_x / 2 = |l(x)|^2 / 2 (0 for inner products), s_x, |e_x|, |c_x|, and r_x in fp32. For a query
// encoded the same way, Cauchy-Schwarz gives
//
//   <z_q, z_x> <= s_q s_x (<c_q, c_x> + |e_x| (|c_q| + |e_q|) + |e_q| |c_x|),
//
// so that a_q + a_x - 2 (that bound + r_q r_x) is a lower bound on the lifted distance of CertEx.
// The pass evaluates it in fp32; the margin eta(q) = 64 u (sqrt(a_q) + sqrt(max a))^2 with
// u = 2^-24 covers every rounding of that evaluation (at most 16 roundings per term, terms summing
// to at most 2.6 (|l(q)| + |l(x)|)^2), and the anchors keep half-precision rows for the threshold.
//
// This header belongs to certex.cu, which includes it after the helpers that it uses.

namespace cuvs::neighbors::certex::detail {

constexpr double kEpilogueUlps = 64.0;

/** int8 codes and scalars of n lifted rows, one warp per row; raises *max_a (float bits). */
__global__ void encode_int8_rows(const float* z,
                                 const float* r,
                                 const float* a,
                                 int m,
                                 int64_t n,
                                 int code_len,
                                 bool with_half_norm,
                                 int8_t* codes,
                                 float* scal,
                                 int* max_a)
{
  const int64_t row = (blockIdx.x * int64_t(blockDim.x) + threadIdx.x) / 32;
  const int lane    = threadIdx.x & 31;
  if (row >= n) { return; }
  float mx = 0;
  for (int j = lane; j < m; j += 32) {
    mx = fmaxf(mx, fabsf(z[row * m + j]));
  }
  for (int off = 16; off; off >>= 1) {
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
  }
  const float s = fmaxf(mx, 1e-30f) / 127.0f;
  float ee = 0, cc = 0;
  for (int j = lane; j < code_len; j += 32) {
    int8_t c = 0;
    if (j < m) {
      const float v  = z[row * m + j] / s;
      const float rv = fminf(fmaxf(rintf(v), -127.0f), 127.0f);
      ee += (v - rv) * (v - rv);
      cc += rv * rv;
      c = int8_t(rv);
    }
    codes[row * code_len + j] = c;
  }
  for (int off = 16; off; off >>= 1) {
    ee += __shfl_xor_sync(0xffffffffu, ee, off);
    cc += __shfl_xor_sync(0xffffffffu, cc, off);
  }
  if (lane == 0) {
    // the norms rounded up, so that the bound stays an upper bound on <z_q, z_x>
    scal[row * 5 + 0] = with_half_norm ? a[row] / 2 : 0.0f;
    scal[row * 5 + 1] = s;
    scal[row * 5 + 2] = sqrtf(ee) * (1.0f + 0x1p-10f);
    scal[row * 5 + 3] = sqrtf(cc) * (1.0f + 0x1p-10f);
    scal[row * 5 + 4] = r[row];
    atomicMax(max_a, __float_as_int(a[row]));
  }
}

/**
 * int8 query codes and scalars: [h_q, s_q, |e_q|, |c_q|, r_q] with h_q = (a_q - U'(q)) / 2 and
 * U'(q) = U(q) + eta(q); lb_base[q] = U(q) - slack, so that seg_v + lb_base[q] bounds the score of
 * a kept pair from below (seg_v = the fp32 bound - U'(q)).
 */
__global__ void encode_int8_queries(const float* z,
                                    const float* r,
                                    const float* a,
                                    int m,
                                    int64_t n,
                                    int code_len,
                                    const double* u,
                                    double max_a,
                                    double slack,
                                    int8_t* codes,
                                    float* scal,
                                    double* lb_base)
{
  const int64_t row = (blockIdx.x * int64_t(blockDim.x) + threadIdx.x) / 32;
  const int lane    = threadIdx.x & 31;
  if (row >= n) { return; }
  float mx = 0;
  for (int j = lane; j < m; j += 32) {
    mx = fmaxf(mx, fabsf(z[row * m + j]));
  }
  for (int off = 16; off; off >>= 1) {
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
  }
  const float s = fmaxf(mx, 1e-30f) / 127.0f;
  float ee = 0, cc = 0;
  for (int j = lane; j < code_len; j += 32) {
    int8_t c = 0;
    if (j < m) {
      const float v  = z[row * m + j] / s;
      const float rv = fminf(fmaxf(rintf(v), -127.0f), 127.0f);
      ee += (v - rv) * (v - rv);
      cc += rv * rv;
      c = int8_t(rv);
    }
    codes[row * code_len + j] = c;
  }
  for (int off = 16; off; off >>= 1) {
    ee += __shfl_xor_sync(0xffffffffu, ee, off);
    cc += __shfl_xor_sync(0xffffffffu, cc, off);
  }
  if (lane == 0) {
    const double aq = a[row];
    const double eta =
      kEpilogueUlps * std::ldexp(1.0, -24) * (sqrt(aq) + sqrt(max_a)) * (sqrt(aq) + sqrt(max_a));
    const double up   = u[row] + eta;
    scal[row * 5 + 0] = isfinite(up) ? float((aq - up) / 2) : -FLT_MAX;
    scal[row * 5 + 1] = s;
    scal[row * 5 + 2] = sqrtf(ee) * (1.0f + 0x1p-10f);
    scal[row * 5 + 3] = sqrtf(cc) * (1.0f + 0x1p-10f);
    scal[row * 5 + 4] = r[row];
    lb_base[row]      = up - eta - slack;
  }
}

/** a / 2 of removed vectors becomes +inf, so that their rows never pass. */
__global__ void remove_int8_rows(
  const int64_t* ids, const int32_t* id_rows, int64_t n_ids, int64_t n_vectors, float* scal)
{
  const int64_t t = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (t >= n_ids) { return; }
  const int64_t id = ids[t];
  if (id < 0 || id >= n_vectors) { return; }
  scal[(id_rows ? id_rows[id] : id) * 5] = INFINITY;
}

/**
 * Append the rows of a source as int8 rows g0, g0 + 1, ...: store the vectors, encode codes and
 * scalars, raise max_a, and encode the half-precision rows of the anchors for the threshold.
 */
template <typename T, typename Src>
auto append_rows_int8(raft::resources const& res, index<T>& idx, Src& src, int64_t g0) -> double
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  const int64_t d     = idx.dim();
  const int m = idx.width(), CL = idx.code_length(), MP = idx.row_length();
  const bool cosine         = idx.metric() == DistanceType::CosineExpanded;
  const bool with_half_norm = idx.metric() != DistanceType::InnerProduct;
  const int max_init        = std::bit_cast<int>(float(idx.max_a));
  rmm::device_scalar<int> max_bits(stream);
  max_bits.set_value_async(max_init, stream);
  for_lifted_chunks(
    res,
    idx,
    src,
    src.n,
    cosine,
    [&](int64_t i, int64_t cn, const T* rows, const float*, float* z, float* r, float* a) {
      encode_int8_rows<<<blocks_for(cn * 32), 256, 0, stream>>>(
        z,
        r,
        a,
        m,
        cn,
        CL,
        with_half_norm,
        idx.codes().data_handle() + (g0 + i) * CL,
        idx.row_scalars().data_handle() + (g0 + i) * 5,
        max_bits.data());
      RAFT_CUDA_TRY(cudaGetLastError());
      raft::copy(idx.vectors_ptr() + (g0 + i) * d, rows, cn * d, stream);
      gather_anchor_vectors<T><<<blocks_for(cn * d / idx.anchor_stride() + d), 256, 0, stream>>>(
        rows, d, g0 + i, cn, idx.anchor_stride(), idx.anchor_vectors().data_handle());
      RAFT_CUDA_TRY(cudaGetLastError());
      raft::resource::sync_stream(res);  // the source may reuse its staging buffer
    });
  const double max_a = std::max(idx.max_a, double(std::bit_cast<float>(max_bits.value(stream))));
  // the anchors of the new rows in half precision with the global scale
  const int s           = idx.anchor_stride();
  const int64_t first_a = (g0 + s - 1) / s, last_a = (g0 + src.n + s - 1) / s;
  if (last_a <= first_a) { return max_a; }
  rmm::device_buffer mx(sizeof(maxima_bits), stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(mx.data(), 0, sizeof(maxima_bits), stream));
  device_source<T> anchors{idx.anchor_vectors().data_handle() + first_a * d, last_a - first_a, d};
  const double g = gamma_of(MP);
  for_lifted_chunks(
    res,
    idx,
    anchors,
    last_a - first_a,
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
        with_half_norm,
        idx.anchor_rows().data_handle() + (first_a + i) * MP,
        static_cast<maxima_bits*>(mx.data()));
      RAFT_CUDA_TRY(cudaGetLastError());
    });
  raft::resource::sync_stream(res);
  return max_a;
}

}  // namespace cuvs::neighbors::certex::detail
