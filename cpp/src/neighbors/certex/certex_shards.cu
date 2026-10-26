/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// CertEx over several GPUs: contiguous shards, one CertEx index per device, a threshold shared by
// all shards, and a merge of the shard answers by exact distance (certex_shards.hpp).

#include <cuvs/core/cuda_fp16.hpp>  // first: __half with default visibility, as in cuVS

#include <cuvs/neighbors/certex_shards.hpp>

#include "detail/kernels.hpp"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

namespace cuvs::neighbors::certex {

namespace detail {

/** Run f(s) for every shard s in its own thread on its device; rethrow the first failure. */
template <typename F>
void for_each_shard(std::vector<int> const& devices, F&& f)
{
  std::vector<std::exception_ptr> errors(devices.size());
  std::vector<std::thread> threads;
  for (size_t s = 0; s < devices.size(); ++s) {
    threads.emplace_back([&, s] {
      try {
        RAFT_CUDA_TRY(cudaSetDevice(devices[s]));
        f(s);
      } catch (...) {
        errors[s] = std::current_exception();
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  for (auto& e : errors) {
    if (e) { std::rethrow_exception(e); }
  }
}

template <typename T>
auto build_sharded(std::vector<int> const& devices,
                   index_params const& params,
                   raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)
  -> sharded_index<T>
{
  const int64_t n = dataset.extent(0), d = dataset.extent(1);
  const int64_t S = int64_t(devices.size());
  RAFT_EXPECTS(S >= 1, "at least one device");
  RAFT_EXPECTS(n >= S, "fewer vectors than shards");
  sharded_index<T> out;
  out.devices = devices;
  out.resources.resize(S);
  out.shards.resize(S);
  for (int64_t s = 0; s < S; ++s) {
    out.first_id.push_back(n * s / S);
  }
  for_each_shard(devices, [&](size_t s) {
    const int64_t a = out.first_id[s], b = s + 1 < size_t(S) ? out.first_id[s + 1] : n;
    out.resources[s] = std::make_unique<raft::device_resources>();
    out.shards[s]    = std::make_unique<index<T>>(build(
      *out.resources[s],
      params,
      raft::make_host_matrix_view<const T, int64_t>(dataset.data_handle() + a * d, b - a, d)));
    raft::resource::sync_stream(*out.resources[s]);
  });
  return out;
}

template <typename T>
void search_sharded(search_params const& params,
                    sharded_index<T> const& idx,
                    raft::host_matrix_view<const T, int64_t, raft::row_major> queries,
                    raft::host_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
                    raft::host_matrix_view<float, int64_t, raft::row_major> distances)
{
  const int64_t nq = queries.extent(0), d = queries.extent(1);
  const int k    = int(neighbors.extent(1));
  const size_t S = idx.shards.size();
  RAFT_EXPECTS(neighbors.extent(0) == nq && distances.extent(0) == nq && distances.extent(1) == k,
               "output shapes do not match the queries");
  for (auto const& sh : idx.shards) {
    RAFT_EXPECTS(sh->size() >= k, "every shard must hold at least k vectors");
  }
  // round 1: the threshold of every query on every shard
  std::vector<std::vector<double>> u(S, std::vector<double>(nq));
  std::vector<std::vector<int64_t>> ids(S, std::vector<int64_t>(nq * k));
  std::vector<std::vector<float>> dist(S, std::vector<float>(nq * k));
  std::vector<double> cap(nq, std::numeric_limits<double>::infinity());
  auto run = [&](size_t s, bool first_round) {
    auto& res           = *idx.resources[s];
    cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
    auto dq             = raft::make_device_matrix<T, int64_t>(res, nq, d);
    auto nb             = raft::make_device_matrix<int64_t, int64_t>(res, nq, k);
    auto ds             = raft::make_device_matrix<float, int64_t>(res, nq, k);
    auto du             = raft::make_device_vector<double, int64_t>(res, nq);
    raft::copy(dq.data_handle(), queries.data_handle(), nq * d, stream);
    detail::threshold_hooks hooks;
    if (first_round) {
      hooks.u_out = du.data_handle();
    } else {
      raft::copy(du.data_handle(), cap.data(), nq, stream);
      hooks.u_cap = du.data_handle();
    }
    detail::set_threshold_hooks(&hooks);
    try {
      search(res, params, *idx.shards[s], raft::make_const_mdspan(dq.view()), nb.view(), ds.view());
    } catch (...) {
      detail::set_threshold_hooks(nullptr);
      throw;
    }
    detail::set_threshold_hooks(nullptr);
    if (first_round) {
      raft::copy(u[s].data(), du.data_handle(), nq, stream);
    } else {
      raft::copy(ids[s].data(), nb.data_handle(), nq * k, stream);
      raft::copy(dist[s].data(), ds.data_handle(), nq * k, stream);
    }
    raft::resource::sync_stream(res);
  };
  detail::for_each_shard(idx.devices, [&](size_t s) { run(s, true); });
  for (size_t s = 0; s < S; ++s) {
    for (int64_t q = 0; q < nq; ++q) {
      cap[q] = std::min(cap[q], u[s][q]);
    }
  }
  // round 2: every shard under the shared threshold, then the merge by exact distance
  detail::for_each_shard(idx.devices, [&](size_t s) { run(s, false); });
  const bool descending = idx.shards[0]->metric() == cuvs::distance::DistanceType::InnerProduct;
  std::vector<std::pair<float, int64_t>> cand;
  for (int64_t q = 0; q < nq; ++q) {
    cand.clear();
    for (size_t s = 0; s < S; ++s) {
      for (int t = 0; t < k; ++t) {
        const int64_t j = ids[s][q * k + t];
        if (j >= 0) { cand.emplace_back(dist[s][q * k + t], idx.first_id[s] + j); }
      }
    }
    std::sort(cand.begin(), cand.end(), [&](auto const& a, auto const& b) {
      if (a.first != b.first) { return descending ? a.first > b.first : a.first < b.first; }
      return a.second < b.second;
    });
    for (int t = 0; t < k; ++t) {
      const bool have                    = size_t(t) < cand.size();
      neighbors.data_handle()[q * k + t] = have ? cand[t].second : -1;
      distances.data_handle()[q * k + t] =
        have ? cand[t].first
             : (descending ? -std::numeric_limits<float>::infinity()
                           : std::numeric_limits<float>::infinity());
    }
  }
}

}  // namespace detail

#define CUVS_INST_CERTEX_SHARDS(T)                                                      \
  auto build_sharded(std::vector<int> const& devices,                                   \
                     index_params const& params,                                        \
                     raft::host_matrix_view<const T, int64_t, raft::row_major> dataset) \
    -> sharded_index<T>                                                                 \
  {                                                                                     \
    return detail::build_sharded<T>(devices, params, dataset);                          \
  }                                                                                     \
  void search(search_params const& params,                                              \
              sharded_index<T> const& idx,                                              \
              raft::host_matrix_view<const T, int64_t, raft::row_major> queries,        \
              raft::host_matrix_view<int64_t, int64_t, raft::row_major> neighbors,      \
              raft::host_matrix_view<float, int64_t, raft::row_major> distances)        \
  {                                                                                     \
    detail::search_sharded<T>(params, idx, queries, neighbors, distances);              \
  }

CUVS_INST_CERTEX_SHARDS(float)
CUVS_INST_CERTEX_SHARDS(half)
CUVS_INST_CERTEX_SHARDS(int8_t)
CUVS_INST_CERTEX_SHARDS(uint8_t)

#undef CUVS_INST_CERTEX_SHARDS

}  // namespace cuvs::neighbors::certex
