/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Device kernels of the CertEx search around the tensor-core passes (kernels.hpp): the encoding of
// the screening rows with their rounding certificate, the exact scores of anchors and survivors in
// double precision, and the ordered verification.
//
// Notation (CertEx, Section 3). The lifted vector l(x) = [V (x - c); |(I - V^T V)(x - c)|] has
// L = m + 1 coordinates and squared norm a(x). Rows are scaled by 1 / alpha so that |l(x)| / alpha
// <= 64 and stored in half precision with four certificate coordinates:
//   row of x: [fl(l(x)/alpha), up(|l|), up(err(x) + gamma n16), -h(x), -l(x)],  h + l = a / 2
//   row of q: [fl(l(q)/alpha), up(err(q)), up(|fl(l(q))|),      1 - eps, 1]
// with up(v) = fl((1 + 2^-10) v). A database row x survives iff its fp32 tensor-core product with
// the query row is at least theta(q) = (a(q) / alpha^2 - U / alpha^2) / 2 - margin(q).

#include "kernels.hpp"

#include <cuvs/core/cuda_fp16.hpp>  // __half with default visibility, as in cuVS

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace cuvs::neighbors::certex::detail {

constexpr double kTol      = 1e-4;             // relative slack of the threshold U
constexpr float kUp        = 1.0f + 0x1p-10f;  // factor of up(v)
constexpr float kDeletedLo = -60000.0f;        // low certificate coordinate of a removed row

/**
 * Exact score of a pair: squared distance (l2), |q|^2 - 2 <q, x> (inner), or the squared distance
 * of the unit vectors, 2 - 2 <q, x> / (|q| |x|) (cosine), computed from the vectors as given.
 */
enum class score_kind { l2, inner, cosine };

template <typename T>
__device__ __forceinline__ float to_f(T v)
{
  return static_cast<float>(v);
}
template <>
__device__ __forceinline__ float to_f<half>(half v)
{
  return __half2float(v);
}

__device__ __forceinline__ half up16(float v) { return __float2half_rn(v * kUp); }

/** Centered float copy of rows [i0, i1) of a T matrix: out[r, :] = x[i0 + r, :] - c. */
template <typename T>
__global__ void center_rows(
  const T* x, int64_t dim, int64_t i0, int64_t i1, const float* c, float* out)
{
  const int64_t n = (i1 - i0) * dim;
  for (int64_t t = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; t < n;
       t += int64_t(gridDim.x) * blockDim.x) {
    const int64_t r = t / dim, j = t - r * dim;
    out[t] = to_f(x[(i0 + r) * dim + j]) - c[j];
  }
}

/**
 * Residual norm and squared lifted norm of every row, one warp per row:
 * r = sqrt(max(0, |p|^2 - |z|^2)), a = |z|^2 + r^2, where p is the centered row and z = V p.
 */
__global__ void lift_norms(const float* p,
                           int64_t dim,
                           const float* z,
                           int m,
                           int64_t n,
                           float* r_out,
                           float* a_out,
                           float* a_max)
{
  const int64_t row = (blockIdx.x * int64_t(blockDim.x) + threadIdx.x) / 32;
  const int lane    = threadIdx.x & 31;
  if (row >= n) { return; }
  float pp = 0, zz = 0;
  for (int64_t j = lane; j < dim; j += 32) {
    const float v = p[row * dim + j];
    pp += v * v;
  }
  for (int j = lane; j < m; j += 32) {
    const float v = z[row * m + j];
    zz += v * v;
  }
  for (int off = 16; off; off >>= 1) {
    pp += __shfl_xor_sync(0xffffffffu, pp, off);
    zz += __shfl_xor_sync(0xffffffffu, zz, off);
  }
  if (lane == 0) {
    const float r = sqrtf(fmaxf(pp - zz, 0.0f));
    const float a = zz + r * r;
    if (r_out) { r_out[row] = r; }
    if (a_out) { a_out[row] = a; }
    if (a_max) { atomicMax(reinterpret_cast<int*>(a_max), __float_as_int(a)); }  // a >= 0
  }
}

/** Certificate maxima of the encoded rows, as float bit patterns of non-negative values. */
struct maxima_bits {
  int lifted, norm, error, high, low;
};

/**
 * Screening rows of database vectors, one warp per row (see the notation above). For inner
 * products (with_half_norm = false), the two coordinates of -a/2 stay zero.
 */
__global__ void encode_rows(const float* z,
                            const float* r,
                            const float* a,
                            int m,
                            int64_t n,
                            int row_len,
                            float inv_alpha,
                            float gamma,
                            bool with_half_norm,
                            half* rows,
                            maxima_bits* mx)
{
  const int64_t row = (blockIdx.x * int64_t(blockDim.x) + threadIdx.x) / 32;
  const int lane    = threadIdx.x & 31;
  if (row >= n) { return; }
  const int L = m + 1;
  half* out   = rows + row * row_len;
  float ex = 0, nx = 0, n16 = 0;
  for (int j = lane; j < row_len; j += 32) {
    if (j < L) {
      const float v  = (j < m ? z[row * m + j] : r[row]) * inv_alpha;
      const half h   = __float2half_rn(v);
      const float hv = __half2float(h);
      ex += (v - hv) * (v - hv);
      nx += v * v;
      n16 += hv * hv;
      out[j] = h;
    } else if (j >= L + 4) {
      out[j] = __float2half_rn(0.0f);
    }
  }
  for (int off = 16; off; off >>= 1) {
    ex += __shfl_xor_sync(0xffffffffu, ex, off);
    nx += __shfl_xor_sync(0xffffffffu, nx, off);
    n16 += __shfl_xor_sync(0xffffffffu, n16, off);
  }
  if (lane == 0) {
    ex                 = sqrtf(ex);
    nx                 = sqrtf(nx);
    const float hn     = sqrtf(n16);  // norm of the half-precision lifted coordinates
    n16                = fmaxf(nx, hn);
    const half c_norm  = up16(nx);
    const half c_error = up16(ex + gamma * n16 * kUp);  // accumulation error of <q16, x16>, folded
    half c_high = __float2half_rn(0.0f), c_low = __float2half_rn(0.0f);
    if (with_half_norm) {
      const float ah = a[row] * inv_alpha * inv_alpha;
      const half h   = __float2half_rn(ah / 2);
      c_high         = __hneg(h);
      c_low          = __hneg(__float2half_rn(ah / 2 - __half2float(h)));
    }
    out[L]     = c_norm;
    out[L + 1] = c_error;
    out[L + 2] = c_high;
    out[L + 3] = c_low;
    // the norm of the lifted half-precision coordinates, as the maximum M_x of the margin
    atomicMax(&mx->lifted, __float_as_int(hn));
    atomicMax(&mx->norm, __float_as_int(__half2float(c_norm)));
    atomicMax(&mx->error, __float_as_int(__half2float(c_error)));
    atomicMax(&mx->high, __float_as_int(-__half2float(c_high)));
    atomicMax(&mx->low, __float_as_int(fabsf(__half2float(c_low))));
  }
}

/** Copy every stride-th row of rows [i0, i1) into the anchor rows (anchor s is row s * stride). */
__global__ void gather_anchor_rows(const half* rows,
                                   int row_len,
                                   int64_t first_anchor,
                                   int64_t last_anchor,
                                   int stride,
                                   half* anchor_rows)
{
  const int64_t n = (last_anchor - first_anchor) * row_len;
  for (int64_t t = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; t < n;
       t += int64_t(gridDim.x) * blockDim.x) {
    const int64_t s = first_anchor + t / row_len, j = t % row_len;
    anchor_rows[s * row_len + j] = rows[s * stride * row_len + j];
  }
}

/**
 * Query rows, one warp per query: the row, a(q) / alpha^2, and the margin
 * gamma (up(err(q)) M_norm + up(|fl(l(q))|) M_error + M_low).
 */
__global__ void encode_queries(const float* z,
                               const float* r,
                               const float* a,
                               int m,
                               int64_t n,
                               int row_len,
                               float inv_alpha,
                               float one_minus_eps,
                               float gamma,
                               float m_norm,
                               float m_error,
                               float m_low,
                               half* rows,
                               double* aqh,
                               float* margin)
{
  const int64_t row = (blockIdx.x * int64_t(blockDim.x) + threadIdx.x) / 32;
  const int lane    = threadIdx.x & 31;
  if (row >= n) { return; }
  const int L = m + 1;
  half* out   = rows + row * row_len;
  float dq = 0, n16 = 0;
  for (int j = lane; j < row_len; j += 32) {
    if (j < L) {
      const float v  = (j < m ? z[row * m + j] : r[row]) * inv_alpha;
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
    aqh[row]          = double(a[row]) * inv_alpha * inv_alpha;
    margin[row] = gamma * (__half2float(c_err) * m_norm + __half2float(c_norm) * m_error + m_low);
  }
}

/** Exact score of query q against database row x in double precision, accumulated by one warp. */
template <typename T, score_kind S>
__device__ double warp_score(const float* q, const T* x, int64_t dim, int lane)
{
  double acc = 0, qq = 0, xx = 0;
  for (int64_t j = lane; j < dim; j += 32) {
    const double qv = q[j], xv = to_f(x[j]);
    if constexpr (S == score_kind::l2) {
      acc += (qv - xv) * (qv - xv);
    } else {
      acc += qv * xv;
      qq += qv * qv;
      if constexpr (S == score_kind::cosine) { xx += xv * xv; }
    }
  }
  for (int off = 16; off; off >>= 1) {
    acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if constexpr (S != score_kind::l2) { qq += __shfl_xor_sync(0xffffffffu, qq, off); }
    if constexpr (S == score_kind::cosine) { xx += __shfl_xor_sync(0xffffffffu, xx, off); }
  }
  if constexpr (S == score_kind::l2) {
    return acc;
  } else if constexpr (S == score_kind::inner) {
    return qq - 2.0 * acc;
  } else {
    return 2.0 - 2.0 * acc / sqrt(fmax(qq * xx, 1e-300));
  }
}

/** Exact scores of the anchors of every query, one warp per (query, anchor) pair. */
template <typename T, score_kind S>
__global__ void anchor_scores(const float* queries,
                              const T* vectors,
                              int64_t dim,
                              const int32_t* anchor_ids,
                              int n_queries,
                              int per_query,
                              int stride,
                              int64_t n_vectors,
                              const uint8_t* removed,
                              Filter filter,
                              double* scores)
{
  const int64_t pair = (blockIdx.x * int64_t(blockDim.x) + threadIdx.x) / 32;
  const int lane     = threadIdx.x & 31;
  if (pair >= int64_t(n_queries) * per_query) { return; }
  const int64_t q = pair / per_query;
  const int64_t a = anchor_ids[pair];  // anchor a is vector a * stride; `vectors` holds the anchors
  const int64_t x = a * stride;
  double s        = INFINITY;
  // a removed anchor, or one that the filter rejects for this query, never sets a threshold
  if (x < n_vectors && !removed[x] && filter.pass(q, x)) {
    s = warp_score<T, S>(queries + q * dim, vectors + a * dim, dim, lane);
  }
  if (lane == 0) { scores[pair] = s; }
}

/**
 * Threshold of every query from its anchors, one block per query: U is the k-th smallest anchor
 * score with a relative (l2, cosine) or absolute (inner) slack, or the bound u_all that every
 * vector satisfies when fewer than k anchors exist. theta = down((aqh - U / alpha^2) / 2 - margin).
 * For cosine, U also carries cos_slack, the largest gap between the exact score and the squared
 * distance of the float32-normalized vectors that the screening rows lift (see cosine_slack()).
 */
template <score_kind S>
__global__ void anchor_threshold(const double* scores,
                                 int per_query,
                                 int k,
                                 const double* aqh,
                                 const float* margin,
                                 const double* q_norm,
                                 double alpha,
                                 double x_bound,
                                 double slack_ip,
                                 double cos_slack,
                                 float* theta,
                                 double* u_out)
{
  extern __shared__ double sv[];  // per_query scores, padded with +inf to a power of two
  const int q = blockIdx.x;
  int p2      = 1;
  while (p2 < per_query) {
    p2 <<= 1;
  }
  for (int j = threadIdx.x; j < p2; j += blockDim.x) {
    sv[j] = j < per_query ? scores[int64_t(q) * per_query + j] : INFINITY;
  }
  __syncthreads();
  for (int size = 2; size <= p2; size <<= 1) {  // bitonic sort, ascending
    for (int stride = size / 2; stride > 0; stride >>= 1) {
      for (int j = threadIdx.x; j < p2; j += blockDim.x) {
        const int o = j ^ stride;
        if (o > j) {
          const bool up = (j & size) == 0;
          if ((sv[j] > sv[o]) == up) {
            const double t = sv[j];
            sv[j]          = sv[o];
            sv[o]          = t;
          }
        }
      }
      __syncthreads();
    }
  }
  __shared__ double kth;
  if (threadIdx.x == 0) { kth = (k <= per_query && isfinite(sv[k - 1])) ? sv[k - 1] : INFINITY; }
  __syncthreads();
  if (threadIdx.x == 0) {
    const double qn = q_norm[q];
    double u_all;
    if constexpr (S != score_kind::inner) {
      u_all = (qn + x_bound) * (qn + x_bound) * (1 + kTol) + 1.0;
    } else {
      u_all = (qn * qn + 2 * qn * x_bound) * (1 + kTol) + 1.0;
    }
    double u = kth;
    if (isfinite(u)) {
      u = S == score_kind::inner ? u + 2 * kTol * qn * slack_ip : u * (1 + kTol) + cos_slack;
    }
    if (!isfinite(u)) { u = u_all; }
    const double t = (aqh[q] - u / (alpha * alpha)) / 2 - double(margin[q]);
    theta[q]       = __double2float_rd(t);
    if (u_out) { u_out[q] = u; }
  }
}

/** Lower U of every query to cap[q] and recompute theta = down((aqh - U / alpha^2) / 2 - margin).
 */
__global__ void cap_threshold(double* u,
                              const double* cap,
                              const double* aqh,
                              const float* margin,
                              double alpha,
                              int64_t B,
                              float* theta)
{
  const int64_t q = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (q >= B || !(cap[q] < u[q])) { return; }
  u[q]     = cap[q];
  theta[q] = __double2float_rd((aqh[q] - u[q] / (alpha * alpha)) / 2 - double(margin[q]));
}

/** Remove vectors by id: the low certificate coordinate of the row (and of its anchor row) becomes
 *  the most negative half-precision value, and the removal flag is set; counts the newly removed
 *  ids. id_rows maps ids to rows (null: rows are ids). */
__global__ void remove_rows(const int64_t* ids,
                            const int32_t* id_rows,
                            int64_t n_ids,
                            int64_t n_vectors,
                            half* rows,
                            half* anchor_rows,
                            int row_len,
                            int L,
                            int stride,
                            uint8_t* removed,
                            unsigned long long* newly)
{
  const int64_t t = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (t >= n_ids) { return; }
  const int64_t id = ids[t];
  if (id < 0 || id >= n_vectors) { return; }
  const int64_t x = id_rows ? id_rows[id] : id;                           // the row of the vector
  if (rows) { rows[x * row_len + L + 3] = __float2half_rn(kDeletedLo); }  // no rows: int8
  if (x % stride == 0) {
    anchor_rows[(x / stride) * row_len + L + 3] = __float2half_rn(kDeletedLo);
  }
  const unsigned int old =
    atomicOr(reinterpret_cast<unsigned int*>(removed + (x & ~int64_t(3))), 1u << (8 * (x & 3)));
  if (!((old >> (8 * (x & 3))) & 0xffu)) { atomicAdd(newly, 1ull); }
}

/** Order-preserving map of a float onto unsigned 32-bit integers. */
__device__ __forceinline__ uint32_t float_key(float f)
{
  const uint32_t b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ __forceinline__ float key_float(uint32_t k)
{
  return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}

/**
 * The query rows that a group of segments of the certified pass screened, with the constants of
 * their lower bounds: q_rows, aqh, and margin are indexed by the query index of a pair. Query row
 * i belongs to list i / period of a set of lists with their own scale and certificate maxima
 * (period > 0, arrays list_*), or to one projection (period = 0, the scalars). Its query in the
 * batch is q_map[i], i % period, or i. Rows of the pairs are ids[x] (ids non-null) or base + x.
 */
struct key_source {
  const half* q_rows;
  const double* aqh;
  const float* margin;
  double alpha, m_nu, m_high;
  const double* list_alpha  = nullptr;
  const double* list_m_nu   = nullptr;
  const double* list_m_high = nullptr;
  int64_t period            = 0;
  const int32_t* q_map      = nullptr;
  const int32_t* ids        = nullptr;
  int32_t base              = 0;
  // int8 rows: the pass wrote the fp32 bound minus U'(q), and lb_base[q] restores the bound
  const double* lb_base = nullptr;
};

/**
 * Certified lower bound and sort key of every survivor, one block per segment of the certified
 * pass: lb = (aqh - 2 (v + err)) alpha^2 - slack with err = gamma (n_q M_nu + M_high) + margin(q),
 * rounded down; key = (query << 32) | ordered(lb). The survivors of all segments land
 * contiguously. slack is the cosine slack (0 for the other metrics). The first first_segments
 * segments come from `first`, the others from `rest`.
 */
__global__ void survivor_keys(const int32_t* seg_q,
                              const int32_t* seg_x,
                              const float* seg_v,
                              const uint32_t* seg_count,
                              const uint32_t* seg_offset,
                              uint32_t seg_cap,
                              int row_len,
                              int L,
                              double gamma,
                              double slack,
                              int first_segments,
                              key_source first,
                              key_source rest,
                              uint64_t* keys,
                              int32_t* ids)
{
  const int64_t seg   = blockIdx.x;
  const uint32_t c    = seg_count[seg];
  const uint32_t o    = seg_offset[seg];
  const key_source& k = seg < first_segments ? first : rest;
  for (uint32_t t = threadIdx.x; t < c; t += blockDim.x) {
    const int64_t src = seg * int64_t(seg_cap) + t;
    const int32_t q   = seg_q[src];
    const int64_t j   = k.period > 0 ? q / k.period : 0;
    const double al   = k.period > 0 ? k.list_alpha[j] : k.alpha;
    const double mnu  = k.period > 0 ? k.list_m_nu[j] : k.m_nu;
    const double mh   = k.period > 0 ? k.list_m_high[j] : k.m_high;
    double lb;
    if (k.lb_base) {
      lb = double(seg_v[src]) + k.lb_base[q];
    } else {
      const double nq  = __half2float(k.q_rows[int64_t(q) * row_len + L + 1]);
      const double err = gamma * (nq * mnu + mh) + double(k.margin[q]);
      lb               = (k.aqh[q] - 2.0 * (double(seg_v[src]) + err)) * al * al - slack;
    }
    lb               = lb - 1e-9 * fabs(lb) - 1e-30;  // rounded down, also when negative
    const int32_t qb = k.q_map ? k.q_map[q] : (k.period > 0 ? int32_t(q % k.period) : q);
    keys[o + t]      = (uint64_t(uint32_t(qb)) << 32) | float_key(__double2float_rd(lb));
    ids[o + t]       = k.ids ? k.ids[seg_x[src]] : k.base + seg_x[src];
  }
}

/** First survivor of every query in the sorted keys (binary search), with start[n_queries] = n. */
__global__ void query_starts(const uint64_t* keys, int64_t n, int n_queries, int64_t* start)
{
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q > n_queries) { return; }
  int64_t lo = 0, hi = n;
  while (lo < hi) {
    const int64_t mid = (lo + hi) / 2;
    if ((keys[mid] >> 32) < uint64_t(q)) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  start[q] = lo;
}

/**
 * Ordered verification, one warp per query: the survivors of a query arrive in increasing order of
 * their certified lower bounds and are verified in chunks of 32, one survivor per lane, until the
 * next lower bound exceeds the k-th exact score. Every survivor that could enter the top k is
 * verified, so that the result is exact. The current top k lives in registers: slot j (j < k) is
 * entry j / 32 of lane j % 32, so that a warp holds up to 32 KPL scores in ascending order. With a
 * limit, only the first `limit` survivors of a query are verified, which yields k exact scores of
 * distinct vectors (an upper bound on the k-th score) but not the exact top k.
 */
template <typename T, score_kind S, int KPL>
__global__ void ordered_verify(const uint64_t* keys,
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
                               unsigned long long* verified = nullptr,
                               int64_t limit                = INT64_MAX,
                               Filter filter                = {})
{
  const int q    = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x & 31;
  if (q >= n_queries) { return; }
  // limit: verify at most the first `limit` survivors (a threshold probe needs only an upper bound)
  const int64_t b = start[q], n_q = start[q + 1] - b;
  const int64_t e = b + (n_q < limit ? n_q : limit);
  const float* qv = queries + int64_t(q) * dim;
  double best[KPL];
  int64_t bid[KPL];
#pragma unroll
  for (int t = 0; t < KPL; ++t) {
    best[t] = INFINITY;
    bid[t]  = -1;
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
  for (int64_t c = b; c < e; c += 32) {
    const double bound = kth();
    const int64_t i    = c + lane;
    const bool valid   = i < e;
    const float lb     = valid ? key_float(uint32_t(keys[i])) : INFINITY;
    if (__shfl_sync(0xffffffffu, lb, 0) > bound) { break; }  // no remaining survivor can enter
    int32_t x = valid ? ids[i] : -1;
    // a removed row may survive only under a threshold below its certificate (a query far outside
    // the collection); it is skipped here, so that no search returns it
    const bool need = valid && double(lb) <= bound && !removed[x] && filter.pass(q, x);
    // the warp scores the needed survivors one by one (coalesced over the dimensions)
    double s     = INFINITY;
    unsigned act = __ballot_sync(0xffffffffu, need);
    if (verified && lane == 0 && act) { atomicAdd(verified, (unsigned long long)__popc(act)); }
    while (act) {
      const int src = __ffs(act) - 1;
      act &= act - 1;
      const int32_t xs = __shfl_sync(0xffffffffu, x, src);
      const double v   = warp_score<T, S>(qv, vectors + int64_t(xs) * dim, dim, lane);
      if (lane == src) { s = v; }
    }
    // merge the 32 scores into the sorted top k
    for (int src = 0; src < 32; ++src) {
      const double v   = __shfl_sync(0xffffffffu, s, src);
      const int32_t xs = __shfl_sync(0xffffffffu, x, src);
      if (!(v < kth())) { continue; }
      int pos = 0;
#pragma unroll
      for (int t = 0; t < KPL; ++t) {
        pos += __popc(__ballot_sync(0xffffffffu, t * 32 + lane < k && best[t] <= v));
      }
#pragma unroll
      for (int t = KPL - 1; t >= 0; --t) {  // shift the slots above pos up by one, top entry first
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
          if (slot == pos) {
            best[t] = v;
            bid[t]  = xs;
          } else if (slot > pos) {
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
}

}  // namespace cuvs::neighbors::certex::detail
