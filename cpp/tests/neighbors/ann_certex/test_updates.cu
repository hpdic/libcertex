/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// Removals, filters (labels, bitset, bitmap), serialization, and searches concurrent with
// extend: every result must stay exact.

#include "../ann_certex.cuh"

#include <cuvs/core/bitmap.hpp>
#include <cuvs/core/bitset.hpp>

#include <rmm/cuda_stream.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

namespace cuvs::neighbors::certex {

namespace {

constexpr int64_t kN = 30000, kD = 96, kQ = 200;
constexpr int kK = 10;

struct Data {
  std::vector<float> x, q;
  Data()
  {
    std::mt19937 rng(7);
    x = make_rows<float>(kN, kD, false, rng);
    q = make_rows<float>(kQ, kD, false, rng);
  }
};

auto data() -> const Data&
{
  static const Data d;
  return d;
}

class AnnCertexUpdates : public ::testing::Test {
 protected:
  AnnCertexUpdates()
    : stream_(raft::resource::get_cuda_stream(res_).get()),
      dx_(raft::make_device_matrix<float, int64_t>(res_, kN, kD)),
      dq_(raft::make_device_matrix<float, int64_t>(res_, kQ, kD)),
      nb_(raft::make_device_matrix<int64_t, int64_t>(res_, kQ, kK)),
      ds_(raft::make_device_matrix<float, int64_t>(res_, kQ, kK))
  {
    raft::copy(dx_.data_handle(), data().x.data(), kN * kD, stream_);
    raft::copy(dq_.data_handle(), data().q.data(), kQ * kD, stream_);
  }

  auto queries() { return raft::make_const_mdspan(dq_.view()); }

  auto ids() -> std::vector<int64_t>
  {
    std::vector<int64_t> h(kQ * kK);
    raft::copy(h.data(), nb_.data_handle(), kQ * kK, stream_);
    raft::resource::sync_stream(res_);
    return h;
  }

  template <typename Keep>
  auto inexact(Keep keep) -> int64_t
  {
    return count_inexact(data().x, data().q, kN, kQ, kD, kK, DistanceType::L2Expanded, ids(), keep);
  }

  template <typename V>
  auto to_device(const std::vector<V>& h)
  {
    auto v = raft::make_device_vector<V, int64_t>(res_, int64_t(h.size()));
    raft::copy(v.data_handle(), h.data(), h.size(), stream_);
    return v;
  }

  raft::device_resources res_;
  cudaStream_t stream_;
  raft::device_matrix<float, int64_t> dx_, dq_;
  raft::device_matrix<int64_t, int64_t> nb_;
  raft::device_matrix<float, int64_t> ds_;
};

}  // namespace

TEST_F(AnnCertexUpdates, Remove)
{
  auto idx = build(res_, index_params{}, raft::make_const_mdspan(dx_.view()));
  std::vector<uint8_t> gone(kN, 0);
  std::vector<int64_t> rm;
  std::mt19937 rng(11);
  for (int64_t j = 0; j < kN; ++j) {
    if (rng() % 10 == 0) { rm.push_back(j); }
  }
  for (int64_t i = 0; i < kQ; ++i) {  // also the nearest neighbor of every query
    int64_t best = 0;
    double bd    = INFINITY;
    for (int64_t j = 0; j < kN; ++j) {
      const double s = exact_score(data().x, data().q, i, j, kD, DistanceType::L2Expanded);
      if (s < bd) {
        bd   = s;
        best = j;
      }
    }
    rm.push_back(best);
  }
  for (auto j : rm) {
    gone[j] = 1;
  }
  rm.push_back(rm.front());  // a repeated id has no further effect
  auto drm = to_device(rm);
  remove(res_, raft::make_const_mdspan(drm.view()), &idx);
  search(res_, search_params{}, idx, queries(), nb_.view(), ds_.view());
  EXPECT_EQ(inexact([&](int64_t, int64_t j) { return !gone[j]; }), 0);
  int64_t n_gone = 0;
  for (auto g : gone) {
    n_gone += g;
  }
  EXPECT_EQ(idx.n_removed(), n_gone);
}

TEST_F(AnnCertexUpdates, Labels)
{
  auto idx = build(res_, index_params{}, raft::make_const_mdspan(dx_.view()));
  std::mt19937 rng(5);
  for (int L : {5, 50, 500, 5000}) {  // 5000 labels: most hold fewer than k vectors
    std::vector<int32_t> lab(kN), qlab(kQ);
    for (auto& v : lab) {
      v = int32_t(rng() % L);
    }
    for (auto& v : qlab) {
      v = int32_t(rng() % L);
    }
    auto dlab  = to_device(lab);
    auto dqlab = to_device(qlab);
    set_labels(res_, raft::make_const_mdspan(dlab.view()), &idx);
    search(res_,
           search_params{},
           idx,
           queries(),
           raft::make_const_mdspan(dqlab.view()),
           nb_.view(),
           ds_.view());
    EXPECT_EQ(inexact([&](int64_t i, int64_t j) { return lab[j] == qlab[i]; }), 0)
      << L << " labels";
  }
}

TEST_F(AnnCertexUpdates, LabelsAfterUpdates)
{
  const int64_t n0 = kN / 4;
  auto idx         = build(res_,
                   index_params{},
                   raft::make_device_matrix_view<const float, int64_t>(dx_.data_handle(), n0, kD));
  extend(
    res_,
    raft::make_device_matrix_view<const float, int64_t>(dx_.data_handle() + n0 * kD, kN - n0, kD),
    &idx);
  std::mt19937 rng(2);
  std::vector<int32_t> lab(kN), qlab(kQ);
  for (auto& v : lab) {
    v = int32_t(rng() % 20);
  }
  for (auto& v : qlab) {
    v = int32_t(rng() % 20);
  }
  auto dlab = to_device(lab);
  set_labels(
    res_, raft::make_device_vector_view<const int32_t, int64_t>(dlab.data_handle(), n0), &idx);
  set_labels(
    res_,
    raft::make_device_vector_view<const int32_t, int64_t>(dlab.data_handle() + n0, kN - n0),
    &idx,
    n0);
  std::vector<uint8_t> gone(kN, 0);
  std::vector<int64_t> rm;
  for (int64_t j = 0; j < kN; ++j) {
    if (rng() % 5 == 0) {
      rm.push_back(j);
      gone[j] = 1;
    }
  }
  auto drm   = to_device(rm);
  auto dqlab = to_device(qlab);
  remove(res_, raft::make_const_mdspan(drm.view()), &idx);
  search(res_,
         search_params{},
         idx,
         queries(),
         raft::make_const_mdspan(dqlab.view()),
         nb_.view(),
         ds_.view());
  EXPECT_EQ(inexact([&](int64_t i, int64_t j) { return lab[j] == qlab[i] && !gone[j]; }), 0);
}

TEST_F(AnnCertexUpdates, PartitionedLabels)
{
  // the label partition covers the first vectors; the rest arrive by extend (the tail); some
  // vectors stay unlabeled, some are removed, and some query labels hold no vector
  const int64_t n0 = kN * 2 / 3;
  std::mt19937 rng(9);
  for (int L : {1, 7, 300, 5000}) {
    std::vector<int32_t> lab(kN), qlab(kQ);
    for (auto& v : lab) {
      v = rng() % 50 == 0 ? -1 : int32_t(rng() % L);
    }
    for (auto& v : qlab) {
      v = int32_t(rng() % (L + 2));  // labels L and L + 1 hold no vector
    }
    auto dlab  = to_device(lab);
    auto dqlab = to_device(qlab);
    auto fresh =
      build(res_,
            index_params{},
            raft::make_device_matrix_view<const float, int64_t>(dx_.data_handle(), n0, kD));
    set_labels(
      res_, raft::make_device_vector_view<const int32_t, int64_t>(dlab.data_handle(), n0), &fresh);
    partition_by_labels(res_, &fresh);
    ASSERT_EQ(fresh.n_partitioned(), n0);
    extend(
      res_,
      raft::make_device_matrix_view<const float, int64_t>(dx_.data_handle() + n0 * kD, kN - n0, kD),
      &fresh);
    set_labels(
      res_,
      raft::make_device_vector_view<const int32_t, int64_t>(dlab.data_handle() + n0, kN - n0),
      &fresh,
      n0);
    ASSERT_EQ(fresh.n_partitioned(), n0);  // labels of the tail keep the partition
    std::vector<uint8_t> gone(kN, 0);
    std::vector<int64_t> rm;
    for (int64_t j = 0; j < kN; ++j) {
      if (rng() % 8 == 0) {
        rm.push_back(j);
        gone[j] = 1;
      }
    }
    auto drm = to_device(rm);
    remove(res_, raft::make_const_mdspan(drm.view()), &fresh);
    auto keep = [&](int64_t i, int64_t j) { return lab[j] == qlab[i] && !gone[j]; };
    search_params sp;
    sp.batch_size = 64;  // several batches
    search(
      res_, sp, fresh, queries(), raft::make_const_mdspan(dqlab.view()), nb_.view(), ds_.view());
    EXPECT_EQ(inexact(keep), 0) << L << " labels, partitioned";
    partition_by_labels(res_, &fresh);  // now over every vector, without a tail
    ASSERT_EQ(fresh.n_partitioned(), kN);
    search(
      res_, sp, fresh, queries(), raft::make_const_mdspan(dqlab.view()), nb_.view(), ds_.view());
    EXPECT_EQ(inexact(keep), 0) << L << " labels, partitioned without a tail";
    set_labels(res_, raft::make_const_mdspan(dlab.view()), &fresh);
    EXPECT_EQ(fresh.n_partitioned(), 0);  // relabeled partitioned vectors drop the partition
    search(
      res_, sp, fresh, queries(), raft::make_const_mdspan(dqlab.view()), nb_.view(), ds_.view());
    EXPECT_EQ(inexact(keep), 0) << L << " labels, after the partition was dropped";
  }
}

TEST_F(AnnCertexUpdates, ListsWithUpdates)
{
  // vectors in list order: ids, removals, bitsets, labels, label partitions, and files must all
  // still address the vectors by their insertion order
  const int64_t n0 = kN * 2 / 3;
  index_params ip;
  ip.n_lists = 40;
  auto idx =
    build(res_, ip, raft::make_device_matrix_view<const float, int64_t>(dx_.data_handle(), n0, kD));
  ASSERT_EQ(idx.n_lists(), 40);
  extend(
    res_,
    raft::make_device_matrix_view<const float, int64_t>(dx_.data_handle() + n0 * kD, kN - n0, kD),
    &idx);
  search(res_, search_params{}, idx, queries(), nb_.view(), ds_.view());
  EXPECT_EQ(inexact([](int64_t, int64_t) { return true; }), 0) << "lists and a tail";
  std::mt19937 rng(17);
  std::vector<uint8_t> gone(kN, 0);
  std::vector<int64_t> rm;
  for (int64_t j = 0; j < kN; ++j) {
    if (rng() % 6 == 0) {
      rm.push_back(j);
      gone[j] = 1;
    }
  }
  auto drm = to_device(rm);
  remove(res_, raft::make_const_mdspan(drm.view()), &idx);
  search(res_, search_params{}, idx, queries(), nb_.view(), ds_.view());
  EXPECT_EQ(inexact([&](int64_t, int64_t j) { return !gone[j]; }), 0) << "removals by id";
  std::vector<uint32_t> bits((kN + 31) / 32, 0);
  for (int64_t j = 0; j < kN; ++j) {
    if (j % 4 != 1) { bits[j / 32] |= 1u << (j % 32); }
  }
  auto dbits = to_device(bits);
  cuvs::core::bitset_view<uint32_t, int64_t> view(dbits.data_handle(), kN);
  search(res_,
         search_params{},
         idx,
         queries(),
         nb_.view(),
         ds_.view(),
         cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(view));
  EXPECT_EQ(inexact([&](int64_t, int64_t j) { return !gone[j] && j % 4 != 1; }), 0)
    << "a bitset over ids";
  std::vector<int32_t> lab(kN), qlab(kQ);
  for (auto& v : lab) {
    v = int32_t(rng() % 25);
  }
  for (auto& v : qlab) {
    v = int32_t(rng() % 25);
  }
  auto dlab  = to_device(lab);
  auto dqlab = to_device(qlab);
  set_labels(res_, raft::make_const_mdspan(dlab.view()), &idx);
  auto keep = [&](int64_t i, int64_t j) { return lab[j] == qlab[i] && !gone[j]; };
  search(res_,
         search_params{},
         idx,
         queries(),
         raft::make_const_mdspan(dqlab.view()),
         nb_.view(),
         ds_.view());
  EXPECT_EQ(inexact(keep), 0) << "labels by id";
  partition_by_labels(res_, &idx);
  search(res_,
         search_params{},
         idx,
         queries(),
         raft::make_const_mdspan(dqlab.view()),
         nb_.view(),
         ds_.view());
  EXPECT_EQ(inexact(keep), 0) << "label partition";
  const char* tmp        = std::getenv("TMPDIR");
  const std::string path = std::string(tmp ? tmp : "/tmp") + "/certex_gtest_lists.certex";
  serialize(res_, path, idx);
  index<float> back(res_);
  deserialize(res_, path, &back);
  std::remove(path.c_str());
  ASSERT_EQ(back.n_lists(), 40);
  search(res_, search_params{}, back, queries(), nb_.view(), ds_.view());
  EXPECT_EQ(inexact([&](int64_t, int64_t j) { return !gone[j]; }), 0) << "a loaded index";
  extend(res_,
         raft::make_device_matrix_view<const float, int64_t>(dx_.data_handle(), 1000, kD),
         &back);  // ids kN, kN + 1, ... after a load
  ASSERT_EQ(back.size(), kN + 1000);
}

TEST_F(AnnCertexUpdates, Int8RowsWithUpdates)
{
  // int8 rows: extend, remove, a bitset, labels, and a file stay exact
  index_params ip;
  ip.int8_rows     = true;
  const int64_t n0 = kN / 2;
  auto idx =
    build(res_, ip, raft::make_device_matrix_view<const float, int64_t>(dx_.data_handle(), n0, kD));
  extend(
    res_,
    raft::make_device_matrix_view<const float, int64_t>(dx_.data_handle() + n0 * kD, kN - n0, kD),
    &idx);
  std::mt19937 rng(23);
  std::vector<uint8_t> gone(kN, 0);
  std::vector<int64_t> rm;
  for (int64_t j = 0; j < kN; ++j) {
    if (rng() % 7 == 0) {
      rm.push_back(j);
      gone[j] = 1;
    }
  }
  for (int64_t i = 0; i < kQ; ++i) {  // also the nearest neighbor of every query
    int64_t best = 0;
    double bd    = INFINITY;
    for (int64_t j = 0; j < kN; ++j) {
      const double s = exact_score(data().x, data().q, i, j, kD, DistanceType::L2Expanded);
      if (s < bd) {
        bd   = s;
        best = j;
      }
    }
    rm.push_back(best);
    gone[best] = 1;
  }
  auto drm = to_device(rm);
  remove(res_, raft::make_const_mdspan(drm.view()), &idx);
  search(res_, search_params{}, idx, queries(), nb_.view(), ds_.view());
  EXPECT_EQ(inexact([&](int64_t, int64_t j) { return !gone[j]; }), 0) << "removals";
  std::vector<uint32_t> bits((kN + 31) / 32, 0);
  for (int64_t j = 0; j < kN; ++j) {
    if (j % 3 == 0) { bits[j / 32] |= 1u << (j % 32); }
  }
  auto dbits = to_device(bits);
  cuvs::core::bitset_view<uint32_t, int64_t> view(dbits.data_handle(), kN);
  search(res_,
         search_params{},
         idx,
         queries(),
         nb_.view(),
         ds_.view(),
         cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(view));
  EXPECT_EQ(inexact([&](int64_t, int64_t j) { return !gone[j] && j % 3 == 0; }), 0) << "bitset";
  const char* tmp        = std::getenv("TMPDIR");
  const std::string path = std::string(tmp ? tmp : "/tmp") + "/certex_gtest_int8.certex";
  serialize(res_, path, idx);
  index<float> back(res_);
  deserialize(res_, path, &back);
  std::remove(path.c_str());
  ASSERT_TRUE(back.int8_rows());
  search(res_, search_params{}, back, queries(), nb_.view(), ds_.view());
  EXPECT_EQ(inexact([&](int64_t, int64_t j) { return !gone[j]; }), 0) << "a loaded index";
}

TEST_F(AnnCertexUpdates, Bitset)
{
  auto idx = build(res_, index_params{}, raft::make_const_mdspan(dx_.view()));
  std::mt19937 rng(3);
  std::vector<uint32_t> bits((kN + 31) / 32, 0);
  std::vector<uint8_t> kept(kN);
  for (int64_t j = 0; j < kN; ++j) {
    kept[j] = rng() % 10 < 3;
    if (kept[j]) { bits[j / 32] |= 1u << (j % 32); }
  }
  auto dbits = to_device(bits);
  cuvs::core::bitset_view<uint32_t, int64_t> view(dbits.data_handle(), kN);
  search(res_,
         search_params{},
         idx,
         queries(),
         nb_.view(),
         ds_.view(),
         cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(view));
  EXPECT_EQ(inexact([&](int64_t, int64_t j) { return kept[j] != 0; }), 0);
}

TEST_F(AnnCertexUpdates, Bitmap)
{
  auto idx = build(res_, index_params{}, raft::make_const_mdspan(dx_.view()));
  std::mt19937 rng(4);
  std::vector<uint32_t> bits((kQ * kN + 31) / 32, 0);
  std::vector<uint8_t> kept(kQ * kN);
  for (int64_t t = 0; t < kQ * kN; ++t) {
    kept[t] = rng() % 10 == 0;
    if (kept[t]) { bits[t / 32] |= 1u << (t % 32); }
  }
  auto dbits = to_device(bits);
  cuvs::core::bitmap_view<uint32_t, int64_t> view(dbits.data_handle(), kQ, kN);
  search(res_,
         search_params{},
         idx,
         queries(),
         nb_.view(),
         ds_.view(),
         cuvs::neighbors::filtering::bitmap_filter<uint32_t, int64_t>(view));
  EXPECT_EQ(inexact([&](int64_t i, int64_t j) { return kept[i * kN + j] != 0; }), 0);
}

TEST_F(AnnCertexUpdates, Serialize)
{
  for (bool on_host : {false, true}) {
    index_params ip;
    ip.host_vectors = on_host;
    auto idx =
      build(res_, ip, raft::make_host_matrix_view<const float, int64_t>(data().x.data(), kN, kD));
    std::vector<int64_t> rm;
    for (int64_t j = 0; j < kN; j += 7) {
      rm.push_back(j);
    }
    std::vector<int32_t> lab(kN);
    for (int64_t j = 0; j < kN; ++j) {
      lab[j] = int32_t(j % 3);
    }
    auto drm  = to_device(rm);
    auto dlab = to_device(lab);
    remove(res_, raft::make_const_mdspan(drm.view()), &idx);
    set_labels(res_, raft::make_const_mdspan(dlab.view()), &idx);
    const char* tmp = std::getenv("TMPDIR");
    const std::string path =
      std::string(tmp ? tmp : "/tmp") + "/certex_gtest_" + std::to_string(int(on_host)) + ".certex";
    serialize(res_, path, idx);
    index<float> back(res_);
    deserialize(res_, path, &back);
    index<uint8_t> wrong(res_);
    EXPECT_THROW(deserialize(res_, path, &wrong), raft::exception);  // the dtype prefix
    std::remove(path.c_str());

    search(res_, search_params{}, idx, queries(), nb_.view(), ds_.view());
    const auto a = ids();
    search(res_, search_params{}, back, queries(), nb_.view(), ds_.view());
    const auto b = ids();
    EXPECT_EQ(a, b);
    EXPECT_EQ(back.size(), idx.size());
    EXPECT_EQ(back.n_removed(), idx.n_removed());
    EXPECT_TRUE(back.has_labels());
    EXPECT_EQ(back.host_vectors(), on_host);
  }
}

/**
 * One thread extends the index in steps while another searches; every result must be exact for
 * one of the collections published between the start and the end of its search.
 */
TEST(AnnCertexConcurrency, SearchDuringExtend)
{
  const auto& x = data().x;
  const auto& q = data().q;
  rmm::cuda_stream s_build, s_search;
  raft::device_resources writer_res(s_build.value()), reader_res(s_search.value());
  const int64_t n0 = kN / 3, step = (kN - n0) / 10;
  auto dx = raft::make_device_matrix<float, int64_t>(writer_res, kN, kD);
  raft::copy(dx.data_handle(), x.data(), kN * kD, s_build.value());
  auto idx = build(writer_res,
                   index_params{},
                   raft::make_device_matrix_view<const float, int64_t>(dx.data_handle(), n0, kD));
  writer_res.sync_stream();
  auto dq = raft::make_device_matrix<float, int64_t>(reader_res, kQ, kD);
  raft::copy(dq.data_handle(), q.data(), kQ * kD, s_search.value());
  reader_res.sync_stream();
  std::atomic<int64_t> published{n0};
  std::thread writer([&] {
    for (int64_t a = n0; a < kN; a += step) {
      const int64_t cnt = std::min(step, kN - a);
      extend(
        writer_res,
        raft::make_device_matrix_view<const float, int64_t>(dx.data_handle() + a * kD, cnt, kD),
        &idx);
      published = a + cnt;
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  });
  struct record {
    int64_t lo, hi;
    std::vector<int64_t> ids;
  };
  std::vector<record> recs;
  auto nb   = raft::make_device_matrix<int64_t, int64_t>(reader_res, kQ, kK);
  auto ds   = raft::make_device_matrix<float, int64_t>(reader_res, kQ, kK);
  bool done = false;
  while (!done) {
    done             = published.load() == kN;  // one more search after the last step
    const int64_t lo = published.load();
    search(
      reader_res, search_params{}, idx, raft::make_const_mdspan(dq.view()), nb.view(), ds.view());
    record r{lo, 0, std::vector<int64_t>(kQ * kK)};
    raft::copy(r.ids.data(), nb.data_handle(), kQ * kK, s_search.value());
    reader_res.sync_stream();
    r.hi = published.load();
    recs.push_back(std::move(r));
  }
  writer.join();
  ASSERT_EQ(idx.size(), kN);
  std::vector<int64_t> prefixes;
  for (int64_t a = n0; a < kN; a += step) {
    prefixes.push_back(a);
  }
  prefixes.push_back(kN);
  // exact top-k scores of every 10th query for every prefix that may have been published
  std::vector<std::vector<std::vector<double>>> best(prefixes.size());
  for (size_t pi = 0; pi < prefixes.size(); ++pi) {
    for (int64_t i = 0; i < kQ; i += 10) {
      std::vector<double> all(prefixes[pi]);
      for (int64_t j = 0; j < prefixes[pi]; ++j) {
        all[j] = exact_score(x, q, i, j, kD, DistanceType::L2Expanded);
      }
      std::partial_sort(all.begin(), all.begin() + kK, all.end());
      best[pi].emplace_back(all.begin(), all.begin() + kK);
    }
  }
  int64_t bad = 0;
  for (auto& r : recs) {
    for (int64_t i = 0, si = 0; i < kQ; i += 10, ++si) {
      std::vector<double> got(kK);
      for (int t = 0; t < kK; ++t) {
        got[t] = exact_score(x, q, i, r.ids[i * kK + t], kD, DistanceType::L2Expanded);
      }
      std::sort(got.begin(), got.end());
      bool any = false;
      for (size_t pi = 0; pi < prefixes.size() && !any; ++pi) {
        if (prefixes[pi] < r.lo || prefixes[pi] > r.hi) { continue; }
        bool ok = true;
        for (int t = 0; t < kK && ok; ++t) {
          ok = std::fabs(got[t] - best[pi][si][t]) <= 1e-9 * (1 + std::fabs(best[pi][si][t]));
        }
        any = ok;
      }
      bad += !any;
    }
  }
  EXPECT_EQ(bad, 0) << recs.size() << " searches";
}

}  // namespace cuvs::neighbors::certex
