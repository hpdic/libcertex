/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// Exact search on a BIGANN prefix: build time, queries per second, and the fraction of queries
// whose sorted fp64 distances equal those of the official ground truth.
//
//   certex_bigann <bigann dir> <millions> [width] [batch] [labels]

#include <cuvs/neighbors/certex.hpp>
#include <cuvs/neighbors/certex_shards.hpp>
#include <cuvs/neighbors/certex_tune.hpp>

#include "../src/neighbors/certex/detail/kernels.hpp"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace cuvs::neighbors;

static std::vector<uint8_t> read_u8bin(const std::string& path, int64_t n, int64_t d)
{
  std::ifstream f(path, std::ios::binary);
  int32_t hdr[2];
  f.read(reinterpret_cast<char*>(hdr), 8);
  std::vector<uint8_t> x(n * d);
  f.read(reinterpret_cast<char*>(x.data()), n * d);
  return x;
}

static std::vector<uint8_t> read_bvecs(const std::string& path, int64_t& n, int64_t d)
{
  std::ifstream f(path, std::ios::binary);
  std::vector<uint8_t> out;
  int32_t dim;
  std::vector<uint8_t> row(d);
  while (f.read(reinterpret_cast<char*>(&dim), 4)) {
    f.read(reinterpret_cast<char*>(row.data()), d);
    out.insert(out.end(), row.begin(), row.end());
  }
  n = int64_t(out.size() / d);
  return out;
}

static std::vector<int32_t> read_ivecs(const std::string& path, int64_t nq, int k)
{
  std::ifstream f(path, std::ios::binary);
  std::vector<int32_t> out(nq * k);
  int32_t dim;
  for (int64_t i = 0; i < nq; ++i) {
    f.read(reinterpret_cast<char*>(&dim), 4);
    std::vector<int32_t> row(dim);
    f.read(reinterpret_cast<char*>(row.data()), 4 * int64_t(dim));
    std::copy_n(row.begin(), k, out.begin() + i * k);
  }
  return out;
}

int main(int argc, char** argv)
{
  // stream-ordered allocations (cudaMallocAsync): the temporaries of a search never synchronize the
  // device
  if (!std::getenv("CERTEX_SYNC_ALLOC")) {
    rmm::mr::set_current_device_resource(rmm::mr::cuda_async_memory_resource{});
  }
  // --host: build from host memory and keep the vectors in pinned host memory
  // --latency: one query at a time, median and 95th percentile
  // --no-partition: filtered search with the label filter over every row (no label partition)
  // --sm80: the cp.async and mma.sync kernels of compute capability 8.x, also on Hopper
  // --lists=P: P k-means lists, the nearest of which every query probes first
  // --local: local projections of the lists
  // --shards=S: S shards on the GPUs of the machine (round robin), a shared threshold
  // --tune: choose the width and the anchor stride by the cost model, then search with them
  // --int8: int8 screening rows
  std::vector<std::string> pos;
  bool on_host = false, latency = false, partition = true;
  int n_lists  = 0;
  bool local   = false;
  int n_shards = 0;
  bool tune    = false;
  bool int8    = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--host") {
      on_host = true;
    } else if (a == "--latency") {
      latency = true;
    } else if (a == "--no-partition") {
      partition = false;
    } else if (a == "--int8") {
      int8 = true;
    } else if (a == "--tune") {
      tune = true;
    } else if (a.rfind("--shards=", 0) == 0) {
      n_shards = std::atoi(a.c_str() + 9);
    } else if (a == "--local") {
      local = true;
    } else if (a.rfind("--lists=", 0) == 0) {
      n_lists = std::atoi(a.c_str() + 8);
    } else if (a == "--sm80") {
      certex::detail::set_kernel_override(certex::detail::screen_kernel::sm80);
    } else if (a == "--pingpong") {
      certex::detail::set_kernel_override(certex::detail::screen_kernel::pingpong);
    } else if (a == "--cooperative") {
      certex::detail::set_kernel_override(certex::detail::screen_kernel::cooperative);
    } else {
      pos.push_back(a);
    }
  }
  if (pos.size() < 2) {
    std::printf(
      "usage: %s <bigann dir> <millions> [width] [batch] [labels] [--host] [--latency] "
      "[--no-partition] [--sm80] [--pingpong] [--cooperative] [--lists=P] [--local] [--shards=S] "
      "[--tune] [--int8]\n",
      argv[0]);
    return 1;
  }
  const std::string dir = pos[0];
  const int64_t n = int64_t(std::atof(pos[1].c_str()) * 1e6), d = 128;
  const int width    = pos.size() > 2 ? std::atoi(pos[2].c_str()) : 0;
  const int batch    = pos.size() > 3 ? std::atoi(pos[3].c_str()) : 2048;
  const int n_labels = pos.size() > 4 ? std::atoi(pos[4].c_str()) : 0;  // random labels
  const int k        = 10;
  auto base          = read_u8bin(dir + "/base.1B.u8bin", n, d);
  int64_t nq         = 0;
  auto query         = read_bvecs(dir + "/bigann_query.bvecs", nq, d);
  auto gt = read_ivecs(dir + "/gnd/idx_" + std::to_string(n / 1000000) + "M.ivecs", nq, k);

  int chosen_width = width, chosen_stride = 16;
  if (tune) {
    raft::device_resources tres;
    certex::index_params tb;
    tb.host_vectors = on_host;
    tb.n_lists      = n_lists;
    const auto t0   = std::chrono::steady_clock::now();
    auto tr         = certex::tune(tres,
                           tb,
                           raft::make_host_matrix_view<const uint8_t, int64_t>(base.data(), n, d),
                           raft::make_host_matrix_view<const uint8_t, int64_t>(query.data(), nq, d),
                           k);
    const double ts = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    for (auto const& c : tr.candidates) {
      std::printf(
        "candidate m=%d s=%d: survivors %.0f, verified %.1f, predicted %.0f queries/s, %.1f GB%s\n",
        c.width,
        c.anchor_stride,
        c.survivors,
        c.verified,
        c.predicted_qps,
        c.device_bytes / 1e9,
        c.fits ? "" : " (does not fit)");
    }
    std::printf("tuned in %.1f s: m=%d, stride %d, predicted %.0f queries/s\n",
                ts,
                tr.width,
                tr.anchor_stride,
                tr.predicted_qps);
    chosen_width  = tr.width;
    chosen_stride = tr.anchor_stride;
  }
  if (n_shards > 0) {
    int n_dev = 0;
    cudaGetDeviceCount(&n_dev);
    std::vector<int> devices(n_shards);
    for (int s = 0; s < n_shards; ++s) {
      devices[s] = s % n_dev;
    }
    certex::index_params sp_ip;
    sp_ip.width        = width;
    sp_ip.n_lists      = n_lists;
    sp_ip.host_vectors = on_host;
    auto b0            = std::chrono::steady_clock::now();
    auto sidx          = certex::build_sharded(
      devices, sp_ip, raft::make_host_matrix_view<const uint8_t, int64_t>(base.data(), n, d));
    const double bs = std::chrono::duration<double>(std::chrono::steady_clock::now() - b0).count();
    std::vector<int64_t> snb(nq * k);
    std::vector<float> sds(nq * k);
    certex::search_params ssp;
    ssp.batch_size = batch;
    auto qv        = raft::make_host_matrix_view<const uint8_t, int64_t>(query.data(), nq, d);
    certex::search(ssp,
                   sidx,
                   qv,
                   raft::make_host_matrix_view<int64_t, int64_t>(snb.data(), nq, k),
                   raft::make_host_matrix_view<float, int64_t>(sds.data(), nq, k));  // warm
    auto s0 = std::chrono::steady_clock::now();
    certex::search(ssp,
                   sidx,
                   qv,
                   raft::make_host_matrix_view<int64_t, int64_t>(snb.data(), nq, k),
                   raft::make_host_matrix_view<float, int64_t>(sds.data(), nq, k));
    const double ss = std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();
    int64_t exact   = 0;
    for (int64_t i = 0; i < nq; ++i) {
      std::vector<int64_t> got(k), ref(k);
      for (int t = 0; t < k; ++t) {
        auto dist = [&](int64_t j) {
          int64_t s2 = 0;
          for (int64_t c = 0; c < d; ++c) {
            const int64_t a2 = int64_t(query[i * d + c]) - int64_t(base[j * d + c]);
            s2 += a2 * a2;
          }
          return s2;
        };
        got[t] = snb[i * k + t] >= 0 ? dist(snb[i * k + t]) : INT64_MAX;
        ref[t] = dist(gt[i * k + t]);
      }
      std::sort(got.begin(), got.end());
      std::sort(ref.begin(), ref.end());
      exact += got == ref;
    }
    std::printf(
      "BIGANN-%ldM width=%d %d shards on %d GPUs: build %.1f s, %.0f queries/s, exact %ld/%ld\n",
      long(n / 1000000),
      width,
      n_shards,
      std::min(n_shards, n_dev),
      bs,
      nq / ss,
      long(exact),
      long(nq));
    return 0;
  }
  raft::device_resources res;
  cudaStream_t stream = raft::resource::get_cuda_stream(res).get();
  auto dx             = raft::make_device_matrix<uint8_t, int64_t>(res, on_host ? 0 : n, d);
  auto dq             = raft::make_device_matrix<uint8_t, int64_t>(res, nq, d);
  if (!on_host) { raft::copy(dx.data_handle(), base.data(), n * d, stream); }
  raft::copy(dq.data_handle(), query.data(), nq * d, stream);
  res.sync_stream();

  certex::index_params ip;
  ip.width             = chosen_width;
  ip.anchor_stride     = chosen_stride;
  ip.int8_rows         = int8;
  ip.n_lists           = n_lists;
  ip.local_projections = local;
  auto t0              = std::chrono::steady_clock::now();
  ip.host_vectors      = on_host;
  auto idx             = on_host
                           ? certex::build(
                   res, ip, raft::make_host_matrix_view<const uint8_t, int64_t>(base.data(), n, d))
                           : certex::build(res, ip, raft::make_const_mdspan(dx.view()));
  res.sync_stream();
  const double build_s =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  if (n_labels > 0) {  // filtered search by random labels; exactness is covered by the tests
    std::vector<int32_t> lab(n), qlab(nq);
    uint64_t r = 88172645463325252ull;
    auto next  = [&] {
      r ^= r << 13;
      r ^= r >> 7;
      r ^= r << 17;
      return r;
    };
    for (auto& v : lab)
      v = int32_t(next() % n_labels);
    for (auto& v : qlab)
      v = int32_t(next() % n_labels);
    auto dlab  = raft::make_device_vector<int32_t, int64_t>(res, n);
    auto dqlab = raft::make_device_vector<int32_t, int64_t>(res, nq);
    raft::copy(dlab.data_handle(), lab.data(), n, stream);
    raft::copy(dqlab.data_handle(), qlab.data(), nq, stream);
    certex::set_labels(res, raft::make_const_mdspan(dlab.view()), &idx);
    const auto p0 = std::chrono::steady_clock::now();
    if (partition) { certex::partition_by_labels(res, &idx); }
    const double ps = std::chrono::duration<double>(std::chrono::steady_clock::now() - p0).count();
    auto fnb        = raft::make_device_matrix<int64_t, int64_t>(res, nq, k);
    auto fds        = raft::make_device_matrix<float, int64_t>(res, nq, k);
    certex::search_params fsp;
    fsp.batch_size = batch;
    auto run       = [&] {
      certex::search(res,
                     fsp,
                     idx,
                     raft::make_const_mdspan(dq.view()),
                     raft::make_const_mdspan(dqlab.view()),
                     fnb.view(),
                     fds.view());
      res.sync_stream();
    };
    run();
    auto f0 = std::chrono::steady_clock::now();
    run();
    const double fs = std::chrono::duration<double>(std::chrono::steady_clock::now() - f0).count();
    // exactness against exhaustive search over the label of every 50th query
    std::vector<int64_t> got(nq * k);
    raft::copy(got.data(), fnb.data_handle(), nq * k, stream);
    res.sync_stream();
    auto dist = [&](int64_t i, int64_t j) {
      int64_t s = 0;
      for (int64_t t = 0; t < d; ++t) {
        const int64_t a = int64_t(query[i * d + t]) - int64_t(base[j * d + t]);
        s += a * a;
      }
      return s;
    };
    int64_t checked = 0, exact = 0;
    for (int64_t i = 0; i < nq; i += 50, ++checked) {
      std::vector<int64_t> all;
      for (int64_t j = 0; j < n; ++j) {
        if (lab[j] == qlab[i]) { all.push_back(dist(i, j)); }
      }
      const size_t want = std::min<size_t>(k, all.size());
      std::partial_sort(all.begin(), all.begin() + want, all.end());
      std::vector<int64_t> mine;
      bool ok = true;
      for (int t = 0; t < k; ++t) {
        const int64_t j = got[i * k + t];
        if (j >= 0) {
          ok &= lab[j] == qlab[i];
          mine.push_back(dist(i, j));
        }
      }
      std::sort(mine.begin(), mine.end());
      ok &= mine.size() == want && std::equal(mine.begin(), mine.end(), all.begin());
      exact += ok;
    }
    std::printf(
      "BIGANN-%ldM width=%d labels=%d %s: %.0f filtered queries/s (partition %.2f s), exact "
      "%ld/%ld sampled\n",
      long(n / 1000000),
      idx.width(),
      n_labels,
      partition ? "partitioned" : "label filter",
      nq / fs,
      ps,
      long(exact),
      long(checked));
    return 0;
  }
  if (latency) {
    auto one_nb = raft::make_device_matrix<int64_t, int64_t>(res, 1, k);
    auto one_ds = raft::make_device_matrix<float, int64_t>(res, 1, k);
    std::vector<double> ms;
    for (int64_t i = 0; i < 220; ++i) {
      auto qv =
        raft::make_device_matrix_view<const uint8_t, int64_t>(dq.data_handle() + i * d, 1, d);
      auto t1 = std::chrono::steady_clock::now();
      certex::search(res, certex::search_params{}, idx, qv, one_nb.view(), one_ds.view());
      res.sync_stream();
      if (i >= 20) {  // warm-up
        ms.push_back(
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count());
      }
    }
    std::sort(ms.begin(), ms.end());
    std::printf("BIGANN-%ldM width=%d single query: p50 %.2f ms, p95 %.2f ms\n",
                long(n / 1000000),
                idx.width(),
                ms[ms.size() / 2],
                ms[ms.size() * 95 / 100]);
    return 0;
  }
  auto nb = raft::make_device_matrix<int64_t, int64_t>(res, nq, k);
  auto ds = raft::make_device_matrix<float, int64_t>(res, nq, k);
  certex::search_params sp;
  sp.batch_size = batch;
  certex::detail::search_stats st;  // survivors and verified rows, from the warm-up search
  certex::detail::collect_stats(&st);
  certex::search(res, sp, idx, raft::make_const_mdspan(dq.view()), nb.view(), ds.view());  // warm
  res.sync_stream();
  certex::detail::collect_stats(nullptr);
  double best = 1e30;
  for (int rep = 0; rep < 3; ++rep) {
    t0 = std::chrono::steady_clock::now();
    certex::search(res, sp, idx, raft::make_const_mdspan(dq.view()), nb.view(), ds.view());
    res.sync_stream();
    best =
      std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }
  std::vector<int64_t> h_nb(nq * k);
  raft::copy(h_nb.data(), nb.data_handle(), nq * k, stream);
  res.sync_stream();
  auto dist = [&](int64_t i, int64_t j) {
    double s = 0;
    for (int64_t t = 0; t < d; ++t) {
      const double a = query[i * d + t], b = base[j * d + t];
      s += (a - b) * (a - b);
    }
    return s;
  };
  int64_t exact = 0;
  for (int64_t i = 0; i < nq; ++i) {
    std::vector<double> got(k), ref(k);
    for (int t = 0; t < k; ++t) {
      got[t] = h_nb[i * k + t] >= 0 ? dist(i, h_nb[i * k + t]) : 1e300;
      ref[t] = dist(i, gt[i * k + t]);
    }
    std::sort(got.begin(), got.end());
    std::sort(ref.begin(), ref.end());
    exact += got == ref;
  }
  size_t free_b = 0, total_b = 0;
  cudaMemGetInfo(&free_b, &total_b);
  std::printf("survivors per query %.0f, verified per query %.1f\n",
              double(st.survivors) / std::max<int64_t>(1, st.queries),
              double(st.verified) / std::max<int64_t>(1, st.queries));
  std::printf("device memory in use %.1f GB (%s vectors)\n",
              (total_b - free_b) / 1e9,
              on_host ? "host" : "device");
  std::printf("BIGANN-%ldM width=%d row=%d batch=%d: build %.1f s, %.0f queries/s, exact %ld/%ld\n",
              long(n / 1000000),
              idx.width(),
              idx.row_length(),
              batch,
              build_s,
              nq / best,
              long(exact),
              long(nq));
  return exact == nq ? 0 : 1;
}
