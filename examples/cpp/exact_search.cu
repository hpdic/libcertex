/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// The C++ module, with the calls of the cuVS index modules: the exact 10 nearest neighbors of 1,000
// queries among one million random vectors, checked on a sample of queries against exhaustive
// search in double precision.
//
//   cmake --build build --target certex_example_cpp && build/certex_example_cpp

#include <cuvs/neighbors/certex.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

int main()
{
  using namespace cuvs::neighbors;
  const int64_t n = 1000000, nq = 1000, d = 96;
  const int k = 10;

  std::vector<float> base(n * d), query(nq * d);
  std::mt19937 gen(0);
  std::normal_distribution<float> normal;
  for (auto& v : base)
    v = normal(gen);
  for (auto& v : query)
    v = normal(gen);

  raft::device_resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  auto dx     = raft::make_device_matrix<float, int64_t>(res, n, d);
  auto dq     = raft::make_device_matrix<float, int64_t>(res, nq, d);
  raft::copy(dx.data_handle(), base.data(), n * d, stream);
  raft::copy(dq.data_handle(), query.data(), nq * d, stream);

  certex::index_params params;  // squared Euclidean distance, automatic screening width
  auto index = certex::build(res, params, raft::make_const_mdspan(dx.view()));

  auto neighbors = raft::make_device_matrix<int64_t, int64_t>(res, nq, k);
  auto distances = raft::make_device_matrix<float, int64_t>(res, nq, k);
  certex::search_params search_params;
  certex::search(res,
                 search_params,
                 index,
                 raft::make_const_mdspan(dq.view()),
                 neighbors.view(),
                 distances.view());

  std::vector<int64_t> ids(nq * k);
  raft::copy(ids.data(), neighbors.data_handle(), nq * k, stream);
  res.sync_stream();

  int exact = 0, checked = 0;
  for (int64_t i = 0; i < nq; i += 50,
               ++checked) {  // exhaustive search in double precision on a sample
    std::vector<double> all(n), got(k);
    for (int64_t j = 0; j < n; ++j) {
      double s = 0;
      for (int64_t c = 0; c < d; ++c) {
        const double t = double(query[i * d + c]) - double(base[j * d + c]);
        s += t * t;
      }
      all[j] = s;
    }
    for (int t = 0; t < k; ++t)
      got[t] = all[ids[i * k + t]];
    std::partial_sort(all.begin(), all.begin() + k, all.end());
    std::sort(got.begin(), got.end());
    exact += std::equal(got.begin(), got.end(), all.begin());
  }
  std::printf("nearest neighbors of query 0:");
  for (int t = 0; t < k; ++t)
    std::printf(" %ld", long(ids[t]));
  std::printf("\nexact on %d of %d checked queries\n", exact, checked);
  return exact == checked ? 0 : 1;
}
