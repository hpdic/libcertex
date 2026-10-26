/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>

#include <cuvs/neighbors/certex.hpp>

#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdspan.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace CUVS_EXPORT cuvs {
namespace neighbors::certex {

/**
 * @defgroup certex_cpp_shards CertEx over several GPUs
 * @{
 */

/**
 * @brief A CertEx index split into contiguous shards, one CertEx index per GPU.
 *
 * A search runs in two rounds. First, every shard computes the threshold U of every query from its
 * own anchors; U bounds the k-th distance over the whole collection, because any k distinct
 * vectors bound it, so that the smallest U over the shards does too. Second, every shard searches
 * under this shared threshold and returns the exact k nearest neighbors of its own vectors within
 * it, and the host merges the shard answers by exact distance. Only k pairs of id and distance per
 * shard and query cross the interconnect.
 *
 * @tparam T data type of the vectors: float, half, int8_t, or uint8_t
 */
template <typename T>
struct sharded_index {
  /** Device of every shard. */
  std::vector<int> devices;
  /** Resources of every shard, created on its device. */
  std::vector<std::unique_ptr<raft::device_resources>> resources;
  /** The index of every shard. */
  std::vector<std::unique_ptr<index<T>>> shards;
  /** Id of the first vector of every shard; shard s holds the ids first_id[s], first_id[s] + 1, ...
   */
  std::vector<int64_t> first_id;

  /** Number of vectors over all shards. */
  [[nodiscard]] auto size() const -> int64_t
  {
    int64_t n = 0;
    for (auto const& s : shards) {
      n += s->size();
    }
    return n;
  }
};

/**
 * @brief Build a sharded index: the vectors are split into one contiguous shard per device, and
 * every shard is built on its device in parallel. A device may appear more than once.
 *
 * @param[in] devices CUDA device of every shard
 * @param[in] params index parameters of every shard
 * @param[in] dataset a host matrix view [n_rows, dim]
 */
auto build_sharded(std::vector<int> const& devices,
                   index_params const& params,
                   raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> sharded_index<float>;
auto build_sharded(std::vector<int> const& devices,
                   index_params const& params,
                   raft::host_matrix_view<const half, int64_t, raft::row_major> dataset)
  -> sharded_index<half>;
auto build_sharded(std::vector<int> const& devices,
                   index_params const& params,
                   raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
  -> sharded_index<int8_t>;
auto build_sharded(std::vector<int> const& devices,
                   index_params const& params,
                   raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
  -> sharded_index<uint8_t>;

/**
 * @brief Search the exact k nearest neighbors of every query over all shards, with
 * k = neighbors.extent(1); every shard must hold at least k vectors.
 *
 * @param[in] params search parameters of every shard
 * @param[in] idx the sharded index
 * @param[in] queries a host matrix view [n_queries, dim]
 * @param[out] neighbors a host matrix view [n_queries, k] of ids
 * @param[out] distances a host matrix view [n_queries, k], as for a single index
 */
void search(search_params const& params,
            sharded_index<float> const& idx,
            raft::host_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::host_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::host_matrix_view<float, int64_t, raft::row_major> distances);
void search(search_params const& params,
            sharded_index<half> const& idx,
            raft::host_matrix_view<const half, int64_t, raft::row_major> queries,
            raft::host_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::host_matrix_view<float, int64_t, raft::row_major> distances);
void search(search_params const& params,
            sharded_index<int8_t> const& idx,
            raft::host_matrix_view<const int8_t, int64_t, raft::row_major> queries,
            raft::host_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::host_matrix_view<float, int64_t, raft::row_major> distances);
void search(search_params const& params,
            sharded_index<uint8_t> const& idx,
            raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::host_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::host_matrix_view<float, int64_t, raft::row_major> distances);

/** @} */

}  // namespace neighbors::certex
}  // namespace CUVS_EXPORT cuvs
