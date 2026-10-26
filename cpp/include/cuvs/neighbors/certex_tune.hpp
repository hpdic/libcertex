/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>

#include <cuvs/neighbors/certex.hpp>

#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cstdint>
#include <vector>

namespace CUVS_EXPORT cuvs {
namespace neighbors::certex {

/**
 * @defgroup certex_cpp_tune CertEx configuration tuning
 * @{
 */

/**
 * @brief Cost model of a search on the current GPU. Every configuration returns the same exact
 * neighbors, so that the model chooses only the speed. The time of a batch of B queries over N
 * rows of L halves with anchor stride s and S survivors and F verified rows per query is
 *
 *   t = a_amac B (N / s) L + a_apair B (N / s)        (anchor stage)
 *     + a_mac B N L + a_pair B N + a_surv B S          (certified pass)
 *     + a_sort B S + a_fetch B F                       (ordering and verification)
 *
 * The pass and anchor constants come from microbenchmarks of the kernels on random rows, and the
 * verification constants from the searches on the prefixes of the dataset.
 */
struct cost_model {
  double a_mac = 0, a_pair = 0, a_surv = 0, a_amac = 0, a_apair = 0, a_sort = 0, a_fetch = 0;
  /** a_mac and a_pair of the certified pass over large_rows rows or more, where the pass of short
   * rows switches kernels (0: as below). */
  double a_mac_large = 0, a_pair_large = 0;
  /** Seconds per batch beyond the terms above (launches and synchronizations of a batch). */
  double a_batch     = 0;
  int64_t large_rows = int64_t(1) << 26;
};

/** @brief Parameters of the tuner. */
struct tune_params {
  /** Candidate widths m (0: 59, 123, and 251, below the dimension). */
  std::vector<int> widths;
  /** Candidate anchor strides. */
  std::vector<int> strides = {16, 32};
  /** Sizes of the two prefixes on which survivors and verified rows are measured. */
  int64_t small_prefix = 1000000, large_prefix = 4000000;
  /** Queries of the prefix searches. */
  int64_t n_queries = 1024;
  /** Queries per batch of the predicted search. */
  int batch_size = 2048;
  /** Device memory available to the index in bytes (0: the free memory of the device). */
  double memory_budget = 0;
};

/** @brief One configuration with its prediction. */
struct tune_candidate {
  int width, anchor_stride;
  double survivors, verified;  // per query, extrapolated to the full dataset
  double predicted_qps;
  double device_bytes;
  bool fits;
};

/** @brief The tuner's choice and every candidate. */
struct tune_result {
  int width = 0, anchor_stride = 16;
  double predicted_qps = 0;
  cost_model model;
  std::vector<tune_candidate> candidates;
};

/** @brief Calibrate the pass and anchor constants of the cost model on the current GPU. */
auto calibrate(raft::resources const& res) -> cost_model;

/**
 * @brief Choose the width and the anchor stride of an index for a dataset: the survivors and
 * verified rows per query are measured on two prefixes of the dataset with the given queries and
 * extrapolated by a power law, and the configuration with the highest predicted throughput that
 * fits the memory budget wins. The other fields of `base` (metric, host_vectors, lists) apply to
 * every candidate.
 */
auto tune(raft::resources const& res,
          index_params const& base,
          raft::host_matrix_view<const float, int64_t, raft::row_major> dataset,
          raft::host_matrix_view<const float, int64_t, raft::row_major> queries,
          int k,
          tune_params const& params = tune_params{}) -> tune_result;
auto tune(raft::resources const& res,
          index_params const& base,
          raft::host_matrix_view<const half, int64_t, raft::row_major> dataset,
          raft::host_matrix_view<const half, int64_t, raft::row_major> queries,
          int k,
          tune_params const& params = tune_params{}) -> tune_result;
auto tune(raft::resources const& res,
          index_params const& base,
          raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset,
          raft::host_matrix_view<const int8_t, int64_t, raft::row_major> queries,
          int k,
          tune_params const& params = tune_params{}) -> tune_result;
auto tune(raft::resources const& res,
          index_params const& base,
          raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset,
          raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
          int k,
          tune_params const& params = tune_params{}) -> tune_result;

/** @} */

}  // namespace neighbors::certex
}  // namespace CUVS_EXPORT cuvs
