/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// The cost model of CertEx searches and the tuner of the width and the anchor stride
// (certex_tune.hpp). The choice changes only the speed: every configuration is exact.

#include <cuvs/core/cuda_fp16.hpp>  // first: __half with default visibility, as in cuVS

#include <cuvs/neighbors/certex_tune.hpp>

#include "detail/kernels.hpp"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <vector>

namespace cuvs::neighbors::certex {

namespace detail {

/** Pseudo-random half rows with entries of standard deviation 1/8 (a hash of the index). */
__global__ void random_rows(half* out, int64_t n, uint32_t seed)
{
  for (int64_t t = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; t < n;
       t += int64_t(gridDim.x) * blockDim.x) {
    uint32_t h = uint32_t(t) * 2654435761u ^ seed;
    h ^= h >> 16;
    h *= 2246822519u;
    h ^= h >> 13;
    h *= 3266489917u;
    h ^= h >> 16;
    // the sum of four uniform bytes, centered: close to a Gaussian of standard deviation 1/8
    const float s = float((h & 255) + ((h >> 8) & 255) + ((h >> 16) & 255) + (h >> 24)) - 510.0f;
    out[t]        = __float2half_rn(s / 148.0f / 8.0f);
  }
}

/** Seconds of the fastest of `reps` runs of f on the stream of res. */
template <typename F>
auto time_of(raft::resources const& res, int reps, F&& f) -> double
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  f();
  raft::resource::sync_stream(res);
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    cudaEvent_t a, b;
    RAFT_CUDA_TRY(cudaEventCreate(&a));
    RAFT_CUDA_TRY(cudaEventCreate(&b));
    RAFT_CUDA_TRY(cudaEventRecord(a, stream));
    f();
    RAFT_CUDA_TRY(cudaEventRecord(b, stream));
    RAFT_CUDA_TRY(cudaEventSynchronize(b));
    float ms = 0;
    RAFT_CUDA_TRY(cudaEventElapsedTime(&ms, a, b));
    best = std::min(best, ms * 1e-3);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
  }
  return best;
}

/** Threshold that keeps about `keep` of N products of rows of L halves with entries of sd 1/8. */
auto gaussian_threshold(int64_t N, int L, double keep) -> float
{
  const double sd = std::sqrt(L / 4096.0);
  double lo = 0, hi = 40;
  while (hi - lo > 1e-6) {
    const double z                                             = (lo + hi) / 2;
    (0.5 * std::erfc(z / std::sqrt(2.0)) * N > keep ? lo : hi) = z;
  }
  return float(lo * sd);
}

auto calibrate(raft::resources const& res) -> cost_model
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  const int B         = 2048;
  const int64_t N = 16000000, NS = N / 16;
  cost_model c;
  double t_pass[2], t_anchor[2], t_surv = 0, s_surv = 0;
  const int Ls[2] = {64, 128};
  for (int w = 0; w < 2; ++w) {
    const int L = Ls[w];
    auto x      = raft::make_device_vector<half, int64_t>(res, N * L);
    auto q      = raft::make_device_vector<half, int64_t>(res, int64_t(B) * L);
    random_rows<<<4096, 256, 0, stream>>>(x.data_handle(), N * L, 17u);
    random_rows<<<256, 256, 0, stream>>>(q.data_handle(), int64_t(B) * L, 91u);
    RAFT_CUDA_TRY(cudaGetLastError());
    const int nseg     = screen_segments(B, N, L);
    const uint32_t cap = uint32_t(int64_t(B) * 6000 / nseg) + 4096;
    auto th            = raft::make_device_vector<float, int64_t>(res, B);
    auto sq            = raft::make_device_vector<int32_t, int64_t>(res, int64_t(cap) * nseg);
    auto sx            = raft::make_device_vector<int32_t, int64_t>(res, int64_t(cap) * nseg);
    auto sv            = raft::make_device_vector<float, int64_t>(res, int64_t(cap) * nseg);
    auto cnt           = raft::make_device_vector<uint32_t, int64_t>(res, nseg);
    auto set_th        = [&](float v) {
      std::vector<float> h(B, v);
      raft::copy(th.data_handle(), h.data(), B, stream);
      raft::resource::sync_stream(res);
    };
    auto pass = [&] {
      screen(q.data_handle(),
             B,
             x.data_handle(),
             N,
             L,
             th.data_handle(),
             sq.data_handle(),
             sx.data_handle(),
             sv.data_handle(),
             cnt.data_handle(),
             cap,
             Filter{},
             stream);
    };
    set_th(INFINITY);  // the products and the per-pair epilogue
    t_pass[w] = time_of(res, 5, pass);
    if (L == 128) {  // the placement of survivors: about 3,000 per query
      set_th(gaussian_threshold(N, L, 3000));
      t_surv = time_of(res, 5, pass);
      std::vector<uint32_t> h(nseg);
      raft::copy(h.data(), cnt.data_handle(), nseg, stream);
      raft::resource::sync_stream(res);
      for (auto v : h) {
        s_surv += v;
      }
      s_surv /= B;
    }
    const int G = int((NS + kTileN - 1) / kTileN);
    auto bv     = raft::make_device_vector<float, int64_t>(res, int64_t(B) * G);
    auto bi     = raft::make_device_vector<int32_t, int64_t>(res, int64_t(B) * G);
    t_anchor[w] = time_of(res, 5, [&] {
      anchor_groups(q.data_handle(),
                    B,
                    x.data_handle(),
                    int(NS),
                    L,
                    bv.data_handle(),
                    bi.data_handle(),
                    Filter{},
                    1,
                    stream);
    });
  }
  c.a_mac  = std::max(0.0, (t_pass[1] - t_pass[0]) / (double(B) * N * 64));
  c.a_pair = std::max(0.0, (t_pass[0] - c.a_mac * double(B) * N * 64) / (double(B) * N));
  // the pass over large collections, where rows of 64 halves take the warp-specialized kernel
  {
    const int64_t NL = c.large_rows;
    double t_large[2];
    for (int w = 0; w < 2; ++w) {
      const int L = Ls[w];
      auto x      = raft::make_device_vector<half, int64_t>(res, NL * L);
      auto q      = raft::make_device_vector<half, int64_t>(res, int64_t(B) * L);
      random_rows<<<4096, 256, 0, stream>>>(x.data_handle(), NL * L, 17u);
      random_rows<<<256, 256, 0, stream>>>(q.data_handle(), int64_t(B) * L, 91u);
      RAFT_CUDA_TRY(cudaGetLastError());
      const int nseg     = screen_segments(B, NL, L);
      const uint32_t cap = 4096;
      auto th            = raft::make_device_vector<float, int64_t>(res, B);
      std::vector<float> h(B, INFINITY);
      raft::copy(th.data_handle(), h.data(), B, stream);
      auto sq    = raft::make_device_vector<int32_t, int64_t>(res, int64_t(cap) * nseg);
      auto sx    = raft::make_device_vector<int32_t, int64_t>(res, int64_t(cap) * nseg);
      auto sv    = raft::make_device_vector<float, int64_t>(res, int64_t(cap) * nseg);
      auto cnt   = raft::make_device_vector<uint32_t, int64_t>(res, nseg);
      t_large[w] = time_of(res, 5, [&] {
        screen(q.data_handle(),
               B,
               x.data_handle(),
               NL,
               L,
               th.data_handle(),
               sq.data_handle(),
               sx.data_handle(),
               sv.data_handle(),
               cnt.data_handle(),
               cap,
               Filter{},
               stream);
      });
    }
    c.a_mac_large = std::max(0.0, (t_large[1] - t_large[0]) / (double(B) * NL * 64));
    c.a_pair_large =
      std::max(0.0, (t_large[0] - c.a_mac_large * double(B) * NL * 64) / (double(B) * NL));
  }
  c.a_surv  = s_surv > 0 ? std::max(0.0, (t_surv - t_pass[1]) / (B * s_surv)) : 0.0;
  c.a_amac  = std::max(0.0, (t_anchor[1] - t_anchor[0]) / (double(B) * NS * 64));
  c.a_apair = std::max(0.0, (t_anchor[0] - c.a_amac * double(B) * NS * 64) / (double(B) * NS));
  return c;
}

inline auto row_length_for(int m) -> int { return m + 5 <= 64 ? 64 : 128 * ((m + 5 + 127) / 128); }

/** Predicted seconds of one batch of B queries (without the verification constants: see fit). */
inline auto screening_time(cost_model const& c, double B, double N, int L, int s, double S)
  -> double
{
  const bool large    = N >= double(c.large_rows) && (c.a_mac_large > 0 || c.a_pair_large > 0);
  const double a_mac  = large ? c.a_mac_large : c.a_mac;
  const double a_pair = large ? c.a_pair_large : c.a_pair;
  return c.a_amac * B * (N / s) * L + c.a_apair * B * (N / s) + a_mac * B * N * L + a_pair * B * N +
         c.a_surv * B * S;
}

template <typename T>
auto tune(raft::resources const& res,
          index_params const& base,
          raft::host_matrix_view<const T, int64_t, raft::row_major> dataset,
          raft::host_matrix_view<const T, int64_t, raft::row_major> queries,
          int k,
          tune_params const& p) -> tune_result
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  const int64_t n = dataset.extent(0), d = dataset.extent(1);
  const int64_t nq = std::min<int64_t>(p.n_queries, queries.extent(0));
  tune_result out;
  out.model               = calibrate(res);
  std::vector<int> widths = p.widths;
  if (widths.empty()) {
    for (int m : {59, 123, 251}) {
      if (m < d) { widths.push_back(m); }
    }
    if (widths.empty()) { widths.push_back(int(d)); }
  }
  const int64_t n1 = std::min(p.small_prefix, n / 4 > 0 ? n / 4 : n);
  const int64_t n2 = std::min(p.large_prefix, n);
  auto dq          = raft::make_device_matrix<T, int64_t>(res, nq, d);
  raft::copy(dq.data_handle(), queries.data_handle(), nq * d, stream);
  auto nb = raft::make_device_matrix<int64_t, int64_t>(res, nq, k);
  auto ds = raft::make_device_matrix<float, int64_t>(res, nq, k);
  // the prefix searches: survivors, verified rows, and times per batch of every candidate
  struct run {
    int c;
    double N, S, F, t;
  };
  std::vector<run> runs;
  std::vector<tune_candidate> cands;
  for (int m : widths) {
    for (int s : p.strides) {
      cands.push_back({m, s, 0, 0, 0, 0, false});
      for (int64_t np : {n1, n2}) {
        index_params ip  = base;
        ip.width         = m;
        ip.anchor_stride = s;
        auto idx         = build(
          res, ip, raft::make_host_matrix_view<const T, int64_t>(dataset.data_handle(), np, d));
        search_params sp;
        sp.batch_size    = int(nq);
        auto search_once = [&] {
          search(res, sp, idx, raft::make_const_mdspan(dq.view()), nb.view(), ds.view());
        };
        search_once();
        detail::search_stats st;
        detail::collect_stats(&st);
        const double t = time_of(res, 3, search_once);
        detail::collect_stats(nullptr);
        const double runs_per_stats = 4;  // the warm-up and three timed searches counted
        runs.push_back({int(cands.size() - 1),
                        double(np),
                        st.survivors / (runs_per_stats * nq),
                        st.verified / (runs_per_stats * nq),
                        t});
      }
    }
  }
  // verification constants: the measured time beyond the screening model, per survivor, per
  // verified row, and per batch, by least squares over all prefix runs (non-negative: a negative
  // per-batch term is dropped and the other two refitted)
  auto fit = [&](bool with_batch) {
    double A[3][3] = {}, y[3] = {};
    for (auto const& r : runs) {
      const int L = row_length_for(cands[r.c].width);
      const double rest =
        r.t - screening_time(out.model, nq, r.N, L, cands[r.c].anchor_stride, r.S);
      const double v[3] = {nq * r.S, nq * r.F, with_batch ? 1.0 : 0.0};
      for (int i = 0; i < 3; ++i) {
        y[i] += v[i] * rest;
        for (int j = 0; j < 3; ++j) {
          A[i][j] += v[i] * v[j];
        }
      }
    }
    if (!with_batch) { A[2][2] = 1; }
    // Cramer's rule on the 3 x 3 normal equations
    auto det3 = [](double M[3][3]) {
      return M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
             M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
             M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
    };
    const double D = det3(A);
    std::array<double, 3> x{0, 0, 0};
    if (D == 0) { return x; }
    for (int c = 0; c < 3; ++c) {
      double M[3][3];
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          M[i][j] = j == c ? y[i] : A[i][j];
        }
      }
      x[c] = det3(M) / D;
    }
    return x;
  };
  auto x = fit(true);
  if (x[2] < 0) { x = fit(false); }
  out.model.a_sort  = std::max(0.0, x[0]);
  out.model.a_fetch = std::max(0.0, x[1]);
  out.model.a_batch = std::max(0.0, x[2]);
  // the prediction at the full size: survivors and verified rows by a power law in N
  size_t free_b = 0, total_b = 0;
  RAFT_CUDA_TRY(cudaMemGetInfo(&free_b, &total_b));
  const double budget = p.memory_budget > 0 ? p.memory_budget : double(free_b) * 0.9;
  const double N      = double(n);
  for (size_t c = 0; c < cands.size(); ++c) {
    const run& a = runs[2 * c];
    const run& b = runs[2 * c + 1];
    auto law     = [&](double y1, double y2) {
      if (y1 <= 0 || y2 <= 0 || b.N <= a.N) { return y2; }
      return y2 * std::pow(N / b.N, std::log(y2 / y1) / std::log(b.N / a.N));
    };
    auto& cd       = cands[c];
    cd.survivors   = law(a.S, b.S);
    cd.verified    = law(a.F, b.F);
    const int L    = row_length_for(cd.width);
    double t_total = 0;
    for (int64_t s0 = 0; s0 < 10000; s0 += p.batch_size) {  // 10,000 queries in batches
      const double Bq = std::min<int64_t>(p.batch_size, 10000 - s0);
      t_total += screening_time(out.model, Bq, N, L, cd.anchor_stride, cd.survivors) +
                 out.model.a_sort * Bq * cd.survivors + out.model.a_fetch * Bq * cd.verified +
                 out.model.a_batch;
    }
    cd.predicted_qps = 10000.0 / t_total;
    cd.device_bytes  = N * 2.0 * L * (1.0 + 1.0 / cd.anchor_stride) +
                      (base.host_vectors ? 0.0 : N * double(d) * sizeof(T));
    cd.fits = cd.device_bytes <= budget;
    if (cd.fits && cd.predicted_qps > out.predicted_qps) {
      out.predicted_qps = cd.predicted_qps;
      out.width         = cd.width;
      out.anchor_stride = cd.anchor_stride;
    }
  }
  out.candidates = std::move(cands);
  return out;
}

}  // namespace detail

auto calibrate(raft::resources const& res) -> cost_model { return detail::calibrate(res); }

#define CUVS_INST_CERTEX_TUNE(T)                                               \
  auto tune(raft::resources const& res,                                        \
            index_params const& base,                                          \
            raft::host_matrix_view<const T, int64_t, raft::row_major> dataset, \
            raft::host_matrix_view<const T, int64_t, raft::row_major> queries, \
            int k,                                                             \
            tune_params const& params) -> tune_result                          \
  {                                                                            \
    return detail::tune<T>(res, base, dataset, queries, k, params);            \
  }

CUVS_INST_CERTEX_TUNE(float)
CUVS_INST_CERTEX_TUNE(half)
CUVS_INST_CERTEX_TUNE(int8_t)
CUVS_INST_CERTEX_TUNE(uint8_t)

#undef CUVS_INST_CERTEX_TUNE

}  // namespace cuvs::neighbors::certex
