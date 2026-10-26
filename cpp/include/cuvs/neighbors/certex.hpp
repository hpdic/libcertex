/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/cuda_fp16.hpp>  // first: __half with default visibility, as in cuVS

#include <cuvs/core/export.hpp>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/pinned_mdarray.hpp>
#include <raft/core/resources.hpp>

#include <cstdint>
#include <istream>
#include <memory>
#include <mutex>
#include <ostream>
#include <shared_mutex>
#include <string>
#include <vector>

namespace CUVS_EXPORT cuvs {
namespace neighbors::certex {

/**
 * @defgroup certex_cpp_index_params CertEx index build parameters
 * @{
 */

/**
 * @brief Parameters of a CertEx index.
 *
 * Supported metrics: L2Expanded, InnerProduct, and CosineExpanded.
 */
struct index_params : cuvs::neighbors::index_params {
  /**
   * Number m of principal directions kept in the half-precision screening rows. 0 picks the
   * smallest tile-aligned width that captures `variance` of the spectrum. Every width returns the
   * exact neighbors; the width only changes the speed and the device memory per vector.
   */
  int width = 0;
  /** Fraction of the variance that an automatic width captures. */
  double variance = 0.95;
  /** Every anchor_stride-th vector is an anchor, whose exact distance bounds the k-th distance. */
  int anchor_stride = 16;
  /** Number of vectors that the principal directions are computed from. */
  int64_t pca_sample = 1000000;
  /**
   * Keep the full-precision vectors in pinned host memory instead of device memory. Device memory
   * then holds only the screening rows and the anchors (about 2 (m + 5) bytes per vector), and the
   * exact distances of the few survivors read their rows from host memory over PCIe.
   */
  bool host_vectors = false;
  /**
   * Number of k-means lists (partitions) of the vectors; 0 keeps the vectors in insertion order.
   * With lists, the vectors are stored in list order, and a search first screens the lists nearest
   * to every query, whose exact distances tighten its threshold. The lists change only the speed;
   * the ids stay the insertion order.
   */
  int n_lists = 0;
  /** Number of vectors that the k-means centers are trained on. */
  int64_t kmeans_sample = 1000000;
  /**
   * Give every list (n_lists > 0) its own center and principal directions, computed from its
   * vectors, so that the lower bounds of its rows follow the local shape of the collection. The
   * bound holds for every orthonormal projection, so that local projections change only the
   * speed. Vectors added later use the global projection.
   */
  bool local_projections = false;
  /**
   * Screening rows of int8 codes: every row holds its projection coordinates as int8 codes with one
   * scale per row and its lifted norm and residual norm in fp32 (m rounded up to 64 bytes plus 20
   * bytes per vector, against 2 (m + 5) bytes in half precision), and the tensor cores accumulate
   * the code products exactly in int32. The quantization error enters the bound of every pair, so
   * that int8 rows return the same exact neighbors with somewhat more survivors. Not combined with
   * lists or label partitions.
   */
  bool int8_rows = false;
};

/** @} */

/**
 * @defgroup certex_cpp_search_params CertEx search parameters
 * @{
 */

/** @brief Parameters of a CertEx search. */
struct search_params : cuvs::neighbors::search_params {
  /** Anchors per query whose exact distances set the screening threshold. */
  int anchors = 128;
  /** Queries per batch; one certified pass over the screening rows serves a whole batch. */
  int batch_size = 2048;
  /**
   * Lists nearest to a query (an index with lists) that are screened first, so that the k-th
   * exact distance among their survivors tightens the threshold; 0 skips this probe.
   */
  int probe_lists = 2;
};

/** @} */

/**
 * @defgroup certex_cpp_index CertEx index
 * @{
 */

/**
 * @brief Exact k-nearest-neighbor index by certified exclusion.
 *
 * Every vector is stored as a lifted projection onto m principal directions plus the norm of the
 * residual, rounded to half precision, with four certificate coordinates that fold the rounding
 * error of the tensor-core product into the product itself. One pass over these screening rows
 * proves that almost every vector cannot be a neighbor of a query; the survivors receive exact
 * distances in increasing order of their certified lower bounds, until no remaining survivor can
 * enter the top k. Every answer therefore equals exhaustive search in double precision.
 *
 * The index owns a copy of the vectors as given, the screening rows, the rows of the
 * anchors, and the projection.
 *
 * @tparam T data type of the vectors: float, half, int8_t, or uint8_t
 */
template <typename T>
struct index : cuvs::neighbors::index {
  /** An empty index, to be filled by deserialize(). */
  explicit index(raft::resources const& res)
    : index(res, cuvs::distance::DistanceType::L2Expanded, 0, 1, 16, false)
  {
  }

  /** Construct an empty index for the given metric and width. */
  index(raft::resources const& res,
        cuvs::distance::DistanceType metric,
        int64_t dim,
        int width,
        int anchor_stride,
        bool host_vectors = false);

  /** Number of vectors. */
  [[nodiscard]] auto size() const noexcept -> int64_t { return size_; }
  /** Dimensionality of the vectors. */
  [[nodiscard]] auto dim() const noexcept -> int64_t { return dim_; }
  /** Number m of principal directions of the screening rows. */
  [[nodiscard]] auto width() const noexcept -> int { return width_; }
  /** Halves per screening row: m + 1 lifted and 4 certificate coordinates, padded to 64 or 128n. */
  [[nodiscard]] auto row_length() const noexcept -> int { return row_length_; }
  /** Every anchor_stride()-th vector is an anchor. */
  [[nodiscard]] auto anchor_stride() const noexcept -> int { return anchor_stride_; }
  /** Distance metric of the index. */
  [[nodiscard]] auto metric() const noexcept -> cuvs::distance::DistanceType { return metric_; }
  /** Bytes of device memory per vector taken by the screening rows and the anchor rows. */
  [[nodiscard]] auto screening_bytes_per_vector() const noexcept -> double
  {
    return 2.0 * row_length_ * (1.0 + 1.0 / anchor_stride_);
  }

  /** Whether the full-precision vectors live in pinned host memory. */
  [[nodiscard]] auto host_vectors() const noexcept -> bool { return host_vectors_; }
  /**
   * The vectors used for the exact distances, [capacity, dim] row major (the first size() rows are
   * valid), in device memory or in pinned host memory that the device reads directly.
   */
  auto vectors_ptr() noexcept -> T*
  {
    return host_vectors_ ? host_rows_.data_handle() : vectors_.data_handle();
  }
  auto vectors_ptr() const noexcept -> const T*
  {
    return host_vectors_ ? host_rows_.data_handle() : vectors_.data_handle();
  }
  /** Full-precision vectors of the anchors 0, s, 2s, ..., always in device memory. */
  auto anchor_vectors() noexcept -> raft::device_matrix_view<T, int64_t, raft::row_major>
  {
    return anchor_vectors_.view();
  }
  auto anchor_vectors() const noexcept
    -> raft::device_matrix_view<const T, int64_t, raft::row_major>
  {
    return raft::make_const_mdspan(anchor_vectors_.view());
  }
  /** Half-precision screening rows with their certificates, [capacity, row_length]. */
  auto rows() noexcept -> raft::device_matrix_view<half, int64_t, raft::row_major>
  {
    return rows_.view();
  }
  auto rows() const noexcept -> raft::device_matrix_view<const half, int64_t, raft::row_major>
  {
    return raft::make_const_mdspan(rows_.view());
  }
  /** Screening rows of the anchors 0, s, 2s, ..., [anchor capacity, row_length]. */
  auto anchor_rows() noexcept -> raft::device_matrix_view<half, int64_t, raft::row_major>
  {
    return anchor_rows_.view();
  }
  auto anchor_rows() const noexcept
    -> raft::device_matrix_view<const half, int64_t, raft::row_major>
  {
    return raft::make_const_mdspan(anchor_rows_.view());
  }
  /** Number of anchors, ceil(size() / anchor_stride()). */
  [[nodiscard]] auto n_anchors() const noexcept -> int64_t
  {
    return (size_ + anchor_stride_ - 1) / anchor_stride_;
  }
  /** Center of the projection, [dim] (the origin for inner products). */
  auto center() noexcept -> raft::device_vector_view<float, int64_t> { return center_.view(); }
  auto center() const noexcept -> raft::device_vector_view<const float, int64_t>
  {
    return raft::make_const_mdspan(center_.view());
  }
  /** Orthonormal principal directions, [width, dim]. */
  auto basis() noexcept -> raft::device_matrix_view<float, int64_t, raft::row_major>
  {
    return basis_.view();
  }
  auto basis() const noexcept -> raft::device_matrix_view<const float, int64_t, raft::row_major>
  {
    return raft::make_const_mdspan(basis_.view());
  }

  /** Removal flags, [capacity]: 1 marks a removed vector, which no search returns. */
  auto removed() noexcept -> raft::device_vector_view<uint8_t, int64_t> { return removed_.view(); }
  auto removed() const noexcept -> raft::device_vector_view<const uint8_t, int64_t>
  {
    return raft::make_const_mdspan(removed_.view());
  }
  /** Number of removed vectors; ids stay stable, so that size() counts them too. */
  [[nodiscard]] auto n_removed() const noexcept -> int64_t { return n_removed_; }
  void add_removed(int64_t c) noexcept { n_removed_ += c; }

  /** Label of every vector, [capacity], for filtered search by label (-1 = no label). */
  auto labels() noexcept -> raft::device_vector_view<int32_t, int64_t> { return labels_.view(); }
  auto labels() const noexcept -> raft::device_vector_view<const int32_t, int64_t>
  {
    return raft::make_const_mdspan(labels_.view());
  }
  /** Whether labels were set, so that a search may restrict a query to its label. */
  [[nodiscard]] auto has_labels() const noexcept -> bool { return has_labels_; }
  void mark_labeled() noexcept { has_labels_ = true; }

  /** Common scale of the lifted coordinates, so that every lifted row has norm at most 64. */
  double alpha = 1.0;
  /** Database maxima of the certificate coordinates, which the search margin uses. */
  struct certificate_maxima {
    double lifted = 0, norm = 0, error = 0, high = 0, low = 0;
  } maxima;

  /**
   * Synchronization of concurrent use: searches hold `rw` shared for every batch, so that the
   * number of vectors and the certificate maxima that a batch reads stay fixed; updates (extend,
   * remove, set_labels) are serialized by `writer`, write their rows beyond the published vectors,
   * and publish under `rw` exclusive. Searches may therefore run concurrently with updates, each
   * thread with its own raft::resources.
   */
  struct sync_state {
    std::shared_mutex rw;
    std::mutex writer;
  };
  auto sync() const noexcept -> sync_state& { return *sync_; }
  /** Number of vectors that the allocated buffers hold. */
  [[nodiscard]] auto capacity() const noexcept -> int64_t { return capacity_; }

  /**
   * int8 screening rows (index_params::int8_rows): codes(), [capacity, code_length()], and
   * row_scalars(), [capacity, 5], of every vector: a / 2 (0 for inner products), the scale s, the
   * norm of the rounding error |e|, the norm of the codes |c|, and the residual norm r; max_a() is
   * the largest lifted squared norm a.
   */
  [[nodiscard]] auto int8_rows() const noexcept -> bool { return int8_; }
  [[nodiscard]] auto code_length() const noexcept -> int { return (width_ + 63) / 64 * 64; }
  auto codes() noexcept -> raft::device_matrix_view<int8_t, int64_t, raft::row_major>
  {
    return codes_.view();
  }
  auto codes() const noexcept -> raft::device_matrix_view<const int8_t, int64_t, raft::row_major>
  {
    return raft::make_const_mdspan(codes_.view());
  }
  auto row_scalars() noexcept -> raft::device_matrix_view<float, int64_t, raft::row_major>
  {
    return row_scal_.view();
  }
  auto row_scalars() const noexcept
    -> raft::device_matrix_view<const float, int64_t, raft::row_major>
  {
    return raft::make_const_mdspan(row_scal_.view());
  }
  double max_a = 0;
  /** Switch an empty index to int8 screening rows. */
  void use_int8_rows() noexcept { int8_ = true; }

  /**
   * Label partition (partition_by_labels()): a copy of the screening rows of the vectors
   * [0, n_partitioned()) sorted by label, so that the vectors of every label occupy the rows
   * [label_offsets()[l], label_offsets()[l + 1]). partition_ids() maps a row of the copy to the id
   * of its vector. Vectors added later are searched without the partition until the next call.
   */
  [[nodiscard]] auto n_partitioned() const noexcept -> int64_t { return n_partitioned_; }
  auto partition_rows() const noexcept
    -> raft::device_matrix_view<const half, int64_t, raft::row_major>
  {
    return raft::make_const_mdspan(partition_rows_.view());
  }
  auto partition_ids() const noexcept -> raft::device_vector_view<const int32_t, int64_t>
  {
    return raft::make_const_mdspan(partition_ids_.view());
  }
  /** Host copy of the label offsets, [number of labels + 1]. */
  [[nodiscard]] auto label_offsets() const noexcept -> const std::vector<int64_t>&
  {
    return label_offsets_;
  }
  /**
   * k-means lists (index_params::n_lists): the rows [list_offsets()[j], list_offsets()[j + 1]) hold
   * the vectors of list j, which covers the vectors stored when the index was built; later vectors
   * follow them in insertion order. row_ids() maps a row to the id of its vector and id_rows() an
   * id to its row; both are empty without lists, where rows and ids coincide.
   */
  [[nodiscard]] auto n_lists() const noexcept -> int
  {
    return int(list_offsets_.empty() ? 0 : list_offsets_.size() - 1);
  }
  [[nodiscard]] auto list_offsets() const noexcept -> const std::vector<int64_t>&
  {
    return list_offsets_;
  }
  auto list_centers() const noexcept
    -> raft::device_matrix_view<const float, int64_t, raft::row_major>
  {
    return raft::make_const_mdspan(list_centers_.view());
  }
  auto row_ids() noexcept -> raft::device_vector_view<int32_t, int64_t> { return row_ids_.view(); }
  auto row_ids() const noexcept -> raft::device_vector_view<const int32_t, int64_t>
  {
    return raft::make_const_mdspan(row_ids_.view());
  }
  auto id_rows() noexcept -> raft::device_vector_view<int32_t, int64_t> { return id_rows_.view(); }
  auto id_rows() const noexcept -> raft::device_vector_view<const int32_t, int64_t>
  {
    return raft::make_const_mdspan(id_rows_.view());
  }
  /** Whether rows and ids differ (an index with lists). */
  [[nodiscard]] auto mapped() const noexcept -> bool { return row_ids_.size() > 0; }
  /** Install the lists of a build: offsets, centers, and the map of rows to ids (inverted here). */
  void set_lists(std::vector<int64_t>&& offsets,
                 raft::device_matrix<float, int64_t, raft::row_major>&& centers,
                 raft::device_vector<int32_t, int64_t>&& row_ids,
                 raft::device_vector<int32_t, int64_t>&& id_rows) noexcept
  {
    list_offsets_ = std::move(offsets);
    list_centers_ = std::move(centers);
    row_ids_      = std::move(row_ids);
    id_rows_      = std::move(id_rows);
  }

  /**
   * Local projections (index_params::local_projections): list j has the center local_centers()[j],
   * the directions local_bases()[j * width(), (j + 1) * width()), the scale local_alpha()[j], and
   * the certificate maxima local_maxima()[j] of its rows.
   */
  [[nodiscard]] auto local_projections() const noexcept -> bool { return !local_alpha_.empty(); }
  auto local_centers() const noexcept
    -> raft::device_matrix_view<const float, int64_t, raft::row_major>
  {
    return raft::make_const_mdspan(local_centers_.view());
  }
  auto local_bases() const noexcept
    -> raft::device_matrix_view<const float, int64_t, raft::row_major>
  {
    return raft::make_const_mdspan(local_bases_.view());
  }
  [[nodiscard]] auto local_alpha() const noexcept -> const std::vector<double>&
  {
    return local_alpha_;
  }
  [[nodiscard]] auto local_maxima() const noexcept -> const std::vector<certificate_maxima>&
  {
    return local_maxima_;
  }
  void set_local(raft::device_matrix<float, int64_t, raft::row_major>&& centers,
                 raft::device_matrix<float, int64_t, raft::row_major>&& bases,
                 std::vector<double>&& alpha,
                 std::vector<certificate_maxima>&& maxima) noexcept
  {
    local_centers_ = std::move(centers);
    local_bases_   = std::move(bases);
    local_alpha_   = std::move(alpha);
    local_maxima_  = std::move(maxima);
  }

  /** Install a label partition (partition_by_labels()), or drop it with n = 0. */
  void set_partition(raft::device_matrix<half, int64_t, raft::row_major>&& rows,
                     raft::device_vector<int32_t, int64_t>&& ids,
                     std::vector<int64_t>&& offsets,
                     int64_t n) noexcept
  {
    partition_rows_ = std::move(rows);
    partition_ids_  = std::move(ids);
    label_offsets_  = std::move(offsets);
    n_partitioned_  = n;
  }
  void drop_partition() noexcept
  {
    label_offsets_.clear();
    n_partitioned_ = 0;
  }

  /** Grow the capacity to at least n vectors, keeping the stored rows. */
  void reserve(raft::resources const& res, int64_t n);
  /** Set the number of stored vectors after rows were appended. */
  void set_size(int64_t n) noexcept { size_ = n; }

 private:
  cuvs::distance::DistanceType metric_;
  std::shared_ptr<sync_state> sync_ = std::make_shared<sync_state>();
  int64_t size_                     = 0;
  int64_t dim_;
  int width_;
  int row_length_;
  int anchor_stride_;
  bool host_vectors_;
  raft::device_matrix<T, int64_t, raft::row_major> vectors_;
  raft::pinned_matrix<T, int64_t, raft::row_major> host_rows_;
  raft::device_matrix<T, int64_t, raft::row_major> anchor_vectors_;
  raft::device_matrix<half, int64_t, raft::row_major> rows_;
  raft::device_matrix<half, int64_t, raft::row_major> anchor_rows_;
  raft::device_vector<float, int64_t> center_;
  raft::device_matrix<float, int64_t, raft::row_major> basis_;
  raft::device_vector<uint8_t, int64_t> removed_;
  int64_t n_removed_ = 0;
  raft::device_vector<int32_t, int64_t> labels_;
  bool has_labels_ = false;
  raft::device_matrix<half, int64_t, raft::row_major> partition_rows_;
  raft::device_vector<int32_t, int64_t> partition_ids_;
  std::vector<int64_t> label_offsets_;
  int64_t n_partitioned_ = 0;
  std::vector<int64_t> list_offsets_;
  raft::device_matrix<float, int64_t, raft::row_major> list_centers_;
  raft::device_vector<int32_t, int64_t> row_ids_;
  raft::device_vector<int32_t, int64_t> id_rows_;
  raft::device_matrix<float, int64_t, raft::row_major> local_centers_;
  raft::device_matrix<float, int64_t, raft::row_major> local_bases_;
  std::vector<double> local_alpha_;
  std::vector<certificate_maxima> local_maxima_;
  int64_t capacity_ = 0;
  bool int8_        = false;
  raft::device_matrix<int8_t, int64_t, raft::row_major> codes_;
  raft::device_matrix<float, int64_t, raft::row_major> row_scal_;
};

/** @} */

/**
 * @defgroup certex_cpp_index_build CertEx index build
 * @{
 */

/**
 * @brief Build a CertEx index.
 *
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   raft::resources res;
 *   certex::index_params params;  // L2Expanded, automatic width
 *   auto index = certex::build(res, params, dataset);
 * @endcode
 *
 * @param[in] res raft resources
 * @param[in] params index parameters
 * @param[in] dataset a device matrix view [n_rows, dim]
 * @return the index
 */
auto build(raft::resources const& res,
           index_params const& params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset) -> index<float>;
auto build(raft::resources const& res,
           index_params const& params,
           raft::device_matrix_view<const half, int64_t, raft::row_major> dataset) -> index<half>;
auto build(raft::resources const& res,
           index_params const& params,
           raft::device_matrix_view<const int8_t, int64_t, raft::row_major> dataset)
  -> index<int8_t>;
auto build(raft::resources const& res,
           index_params const& params,
           raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
  -> index<uint8_t>;

/**
 * @brief Build a CertEx index from vectors in host memory, which may exceed device memory.
 *
 * The vectors are streamed to the device in chunks; with params.host_vectors, they stay in pinned
 * host memory and only the screening rows and the anchors occupy device memory.
 */
auto build(raft::resources const& res,
           index_params const& params,
           raft::host_matrix_view<const float, int64_t, raft::row_major> dataset) -> index<float>;
auto build(raft::resources const& res,
           index_params const& params,
           raft::host_matrix_view<const half, int64_t, raft::row_major> dataset) -> index<half>;
auto build(raft::resources const& res,
           index_params const& params,
           raft::host_matrix_view<const int8_t, int64_t, raft::row_major> dataset) -> index<int8_t>;
auto build(raft::resources const& res,
           index_params const& params,
           raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> dataset)
  -> index<uint8_t>;

/** @} */

/**
 * @defgroup certex_cpp_index_extend CertEx index extend
 * @{
 */

/**
 * @brief Append vectors with the projection of the index.
 *
 * The bound holds for every orthonormal projection and every center, and the certificate maxima
 * rise to cover the new rows, so that every search stays exact without a rebuild. The new vectors
 * receive the ids size(), size() + 1, ... of the index before the call.
 *
 * @param[in] res raft resources
 * @param[in] new_vectors a device matrix view [n_rows, dim]
 * @param[inout] idx the index
 */
void extend(raft::resources const& res,
            raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            index<float>* idx);
void extend(raft::resources const& res,
            raft::device_matrix_view<const half, int64_t, raft::row_major> new_vectors,
            index<half>* idx);
void extend(raft::resources const& res,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
            index<int8_t>* idx);
void extend(raft::resources const& res,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
            index<uint8_t>* idx);

/** @brief Append vectors from host memory (see the device overload). */
void extend(raft::resources const& res,
            raft::host_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            index<float>* idx);
void extend(raft::resources const& res,
            raft::host_matrix_view<const half, int64_t, raft::row_major> new_vectors,
            index<half>* idx);
void extend(raft::resources const& res,
            raft::host_matrix_view<const int8_t, int64_t, raft::row_major> new_vectors,
            index<int8_t>* idx);
void extend(raft::resources const& res,
            raft::host_matrix_view<const uint8_t, int64_t, raft::row_major> new_vectors,
            index<uint8_t>* idx);

/** @} */

/**
 * @defgroup certex_cpp_index_labels CertEx index labels
 * @{
 */

/**
 * @brief Set the labels of the vectors first, first + 1, ..., for filtered search by label.
 *
 * A search with query labels returns, for every query, the exact k nearest vectors that carry the
 * label of the query (fewer if the label holds fewer vectors). Vectors without a label set hold -1.
 *
 * @param[in] res raft resources
 * @param[in] labels a device vector view of non-negative labels
 * @param[inout] idx the index
 * @param[in] first id of the vector that receives labels[0]
 */
void set_labels(raft::resources const& res,
                raft::device_vector_view<const int32_t, int64_t> labels,
                index<float>* idx,
                int64_t first = 0);
void set_labels(raft::resources const& res,
                raft::device_vector_view<const int32_t, int64_t> labels,
                index<half>* idx,
                int64_t first = 0);
void set_labels(raft::resources const& res,
                raft::device_vector_view<const int32_t, int64_t> labels,
                index<int8_t>* idx,
                int64_t first = 0);
void set_labels(raft::resources const& res,
                raft::device_vector_view<const int32_t, int64_t> labels,
                index<uint8_t>* idx,
                int64_t first = 0);

/**
 * @brief Sort a copy of the screening rows by label, so that a filtered search by label screens
 * only the rows of the label of every query.
 *
 * A search with query labels then pairs every query with the rows of its label alone, so that its
 * work shrinks with the selectivity; without the partition, it screens every row and keeps only
 * rows of the query's label. The copy takes another 2 row_length() bytes and 4 bytes per vector.
 * Vectors added later are searched without the partition until the next call, and setting the
 * labels of partitioned vectors drops it. Both kinds of search return the same exact neighbors.
 *
 * @param[in] res raft resources
 * @param[inout] idx an index with labels
 */
void partition_by_labels(raft::resources const& res, index<float>* idx);
void partition_by_labels(raft::resources const& res, index<half>* idx);
void partition_by_labels(raft::resources const& res, index<int8_t>* idx);
void partition_by_labels(raft::resources const& res, index<uint8_t>* idx);

/** @} */

/**
 * @defgroup certex_cpp_index_remove CertEx index remove
 * @{
 */

/**
 * @brief Remove vectors by id; no later search returns them.
 *
 * A removed row meets the constant coordinate of every query row with the most negative
 * half-precision value, so that it never survives screening; a removed anchor no longer sets a
 * threshold. Ids stay stable and are never reused; removing an id twice has no further effect.
 *
 * @param[in] res raft resources
 * @param[in] ids a device vector view of ids in [0, size())
 * @param[inout] idx the index
 */
void remove(raft::resources const& res,
            raft::device_vector_view<const int64_t, int64_t> ids,
            index<float>* idx);
void remove(raft::resources const& res,
            raft::device_vector_view<const int64_t, int64_t> ids,
            index<half>* idx);
void remove(raft::resources const& res,
            raft::device_vector_view<const int64_t, int64_t> ids,
            index<int8_t>* idx);
void remove(raft::resources const& res,
            raft::device_vector_view<const int64_t, int64_t> ids,
            index<uint8_t>* idx);

/** @} */

/**
 * @defgroup certex_cpp_index_search CertEx index search
 * @{
 */

/**
 * @brief Search the exact k nearest neighbors of every query, with k = neighbors.extent(1).
 *
 * The distances are squared Euclidean distances (L2Expanded, ascending), inner products
 * (InnerProduct, descending), or cosine distances 1 - cos (CosineExpanded, ascending), as for
 * cuvs::neighbors::brute_force.
 *
 * @param[in] res raft resources
 * @param[in] params search parameters
 * @param[in] idx the index
 * @param[in] queries a device matrix view [n_queries, dim]
 * @param[out] neighbors a device matrix view [n_queries, k], k <= 2048
 * @param[out] distances a device matrix view [n_queries, k]
 * @param[in] sample_filter a cuVS bitset filter (rows kept for every query) or bitmap filter (one
 * row of bits per query); the result holds the exact k nearest vectors that the filter keeps
 */
void search(raft::resources const& res,
            search_params const& params,
            index<float> const& idx,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});
void search(raft::resources const& res,
            search_params const& params,
            index<half> const& idx,
            raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});
void search(raft::resources const& res,
            search_params const& params,
            index<int8_t> const& idx,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});
void search(raft::resources const& res,
            search_params const& params,
            index<uint8_t> const& idx,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter =
              cuvs::neighbors::filtering::none_sample_filter{});

/**
 * @brief Filtered search by label: the exact k nearest vectors of the label of every query.
 *
 * Requires labels (set_labels). Entries beyond the number of vectors of a label hold the id -1.
 *
 * @param[in] res raft resources
 * @param[in] params search parameters
 * @param[in] idx the index
 * @param[in] queries a device matrix view [n_queries, dim]
 * @param[in] query_labels a device vector view [n_queries], one label per query
 * @param[out] neighbors a device matrix view [n_queries, k], k <= 2048
 * @param[out] distances a device matrix view [n_queries, k]
 */
void search(raft::resources const& res,
            search_params const& params,
            index<float> const& idx,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_vector_view<const int32_t, int64_t> query_labels,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);
void search(raft::resources const& res,
            search_params const& params,
            index<half> const& idx,
            raft::device_matrix_view<const half, int64_t, raft::row_major> queries,
            raft::device_vector_view<const int32_t, int64_t> query_labels,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);
void search(raft::resources const& res,
            search_params const& params,
            index<int8_t> const& idx,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
            raft::device_vector_view<const int32_t, int64_t> query_labels,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);
void search(raft::resources const& res,
            search_params const& params,
            index<uint8_t> const& idx,
            raft::device_matrix_view<const uint8_t, int64_t, raft::row_major> queries,
            raft::device_vector_view<const int32_t, int64_t> query_labels,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

/** @} */

/**
 * @defgroup certex_cpp_index_serialize CertEx index serialize
 * @{
 */

/**
 * @brief Save the index, including its vectors, so that deserialize() restores it without a
 * rebuild.
 *
 * The serialization format may change between versions of the library. A file starts with the
 * NumPy descriptor of the vector type, padded to 4 bytes (as for CAGRA), so that a reader can pick
 * the index type before deserializing.
 *
 * @code{.cpp}
 *   certex::serialize(res, "index.certex", index);
 *   certex::index<float> loaded(res);
 *   certex::deserialize(res, "index.certex", &loaded);
 * @endcode
 */
void serialize(raft::resources const& res, std::ostream& os, index<float> const& idx);
void serialize(raft::resources const& res, const std::string& filename, index<float> const& idx);
void deserialize(raft::resources const& res, std::istream& is, index<float>* idx);
void deserialize(raft::resources const& res, const std::string& filename, index<float>* idx);
void serialize(raft::resources const& res, std::ostream& os, index<half> const& idx);
void serialize(raft::resources const& res, const std::string& filename, index<half> const& idx);
void deserialize(raft::resources const& res, std::istream& is, index<half>* idx);
void deserialize(raft::resources const& res, const std::string& filename, index<half>* idx);
void serialize(raft::resources const& res, std::ostream& os, index<int8_t> const& idx);
void serialize(raft::resources const& res, const std::string& filename, index<int8_t> const& idx);
void deserialize(raft::resources const& res, std::istream& is, index<int8_t>* idx);
void deserialize(raft::resources const& res, const std::string& filename, index<int8_t>* idx);
void serialize(raft::resources const& res, std::ostream& os, index<uint8_t> const& idx);
void serialize(raft::resources const& res, const std::string& filename, index<uint8_t> const& idx);
void deserialize(raft::resources const& res, std::istream& is, index<uint8_t>* idx);
void deserialize(raft::resources const& res, const std::string& filename, index<uint8_t>* idx);

/** @} */

}  // namespace neighbors::certex
}  // namespace CUVS_EXPORT cuvs
