/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <cuvs/neighbors/common.h>
#include <dlpack/dlpack.h>
#include <stdbool.h>
#include <stdint.h>

#include <cuvs/core/export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup certex_c_index_params CertEx C-API index build parameters
 * @{
 */

/**
 * @brief Parameters of a CertEx index (cuvs::neighbors::certex::index_params). Every setting
 * returns the same exact neighbors; the settings change only the speed and the device memory.
 */
struct cuvsCertexIndexParams {
  /** L2Expanded, InnerProduct, or CosineExpanded */
  cuvsDistanceType metric;
  /** Projection width m of the screening rows (0: the smallest width that explains `variance`) */
  int32_t width;
  /** Fraction of the variance that the automatic width explains */
  double variance;
  /** Every anchor_stride-th vector is an anchor of the threshold stage */
  int32_t anchor_stride;
  /** Vectors sampled for the principal directions */
  int64_t pca_sample;
  /** Keep the full-precision vectors in pinned host memory */
  bool host_vectors;
  /** k-means lists (0: none); a search first screens the nearest lists of every query */
  int32_t n_lists;
  /** Every list gets its own projection (requires n_lists > 0) */
  bool local_projections;
  /** int8 screening rows */
  bool int8_rows;
};

typedef struct cuvsCertexIndexParams* cuvsCertexIndexParams_t;

/** @brief Allocate index parameters with the defaults of the C++ API. */
CUVS_EXPORT cuvsError_t cuvsCertexIndexParamsCreate(cuvsCertexIndexParams_t* params);

/** @brief Free index parameters. */
CUVS_EXPORT cuvsError_t cuvsCertexIndexParamsDestroy(cuvsCertexIndexParams_t params);

/**
 * @}
 */

/**
 * @defgroup certex_c_search_params CertEx C-API search parameters
 * @{
 */

/** @brief Parameters of a CertEx search (cuvs::neighbors::certex::search_params). */
struct cuvsCertexSearchParams {
  /** Anchors per query whose exact distances set the threshold */
  int32_t anchors;
  /** Queries per batch */
  int32_t batch_size;
  /** Nearest lists that an index with lists screens first */
  int32_t probe_lists;
};

typedef struct cuvsCertexSearchParams* cuvsCertexSearchParams_t;

/** @brief Allocate search parameters with the defaults of the C++ API. */
CUVS_EXPORT cuvsError_t cuvsCertexSearchParamsCreate(cuvsCertexSearchParams_t* params);

/** @brief Free search parameters. */
CUVS_EXPORT cuvsError_t cuvsCertexSearchParamsDestroy(cuvsCertexSearchParams_t params);

/**
 * @}
 */

/**
 * @defgroup certex_c_index CertEx C-API index
 * @{
 */

/** @brief An opaque CertEx index of float32, float16, int8, or uint8 vectors. */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsCertexIndex;

typedef cuvsCertexIndex* cuvsCertexIndex_t;

/** @brief Allocate an empty index handle. */
CUVS_EXPORT cuvsError_t cuvsCertexIndexCreate(cuvsCertexIndex_t* index);

/** @brief Free an index handle and its index. */
CUVS_EXPORT cuvsError_t cuvsCertexIndexDestroy(cuvsCertexIndex_t index);

/** @brief Number of vectors added to the index (removed ones included). */
CUVS_EXPORT cuvsError_t cuvsCertexIndexGetSize(cuvsCertexIndex_t index, int64_t* size);

/** @brief Dimension of the vectors of the index. */
CUVS_EXPORT cuvsError_t cuvsCertexIndexGetDim(cuvsCertexIndex_t index, int64_t* dim);

/**
 * @}
 */

/**
 * @defgroup certex_c_index_build CertEx C-API build, extend, remove, labels
 * @{
 */

/**
 * @brief Build an index from a dataset [n_rows, dim] of float32, float16, int8, or uint8 in
 * device or host memory (a host dataset is streamed to the GPU in chunks).
 *
 * @code{.c}
 *   cuvsResources_t res;
 *   cuvsResourcesCreate(&res);
 *   cuvsCertexIndexParams_t params;
 *   cuvsCertexIndexParamsCreate(&params);
 *   cuvsCertexIndex_t index;
 *   cuvsCertexIndexCreate(&index);
 *   cuvsCertexBuild(res, params, &dataset, index);
 * @endcode
 */
CUVS_EXPORT cuvsError_t cuvsCertexBuild(cuvsResources_t res,
                                        cuvsCertexIndexParams_t params,
                                        DLManagedTensor* dataset,
                                        cuvsCertexIndex_t index);

/** @brief Append vectors [n, dim] (device or host, the type of the index); they receive the next
 *  consecutive ids. */
CUVS_EXPORT cuvsError_t cuvsCertexExtend(cuvsResources_t res,
                                         DLManagedTensor* new_vectors,
                                         cuvsCertexIndex_t index);

/** @brief Remove vectors by id (an int64 device vector); no later search returns them. */
CUVS_EXPORT cuvsError_t cuvsCertexRemove(cuvsResources_t res,
                                         DLManagedTensor* ids,
                                         cuvsCertexIndex_t index);

/** @brief Set the labels (an int32 device vector) of the vectors first, first + 1, ... for
 *  filtered search by label. */
CUVS_EXPORT cuvsError_t cuvsCertexSetLabels(cuvsResources_t res,
                                            DLManagedTensor* labels,
                                            int64_t first,
                                            cuvsCertexIndex_t index);

/** @brief Sort a copy of the screening rows by label, so that a search by label screens only the
 *  rows of the label of every query (index with the global projection). */
CUVS_EXPORT cuvsError_t cuvsCertexPartitionByLabels(cuvsResources_t res, cuvsCertexIndex_t index);

/**
 * @}
 */

/**
 * @defgroup certex_c_index_search CertEx C-API search
 * @{
 */

/**
 * @brief Exact k nearest neighbors of every query, k = the second extent of `neighbors`.
 *
 * queries [n_queries, dim] (device, the type of the index), neighbors [n_queries, k] int64 and
 * distances [n_queries, k] float32 (device). The distances follow cuVS brute force: squared
 * Euclidean distances, inner products, or cosine distances. `filter` is NO_FILTER, a BITSET over
 * the ids, or a BITMAP of one row of bits per query (uint32 device words); the result then holds
 * the exact k nearest vectors that the filter keeps, with id -1 where fewer exist.
 */
CUVS_EXPORT cuvsError_t cuvsCertexSearch(cuvsResources_t res,
                                         cuvsCertexSearchParams_t params,
                                         cuvsCertexIndex_t index,
                                         DLManagedTensor* queries,
                                         DLManagedTensor* neighbors,
                                         DLManagedTensor* distances,
                                         cuvsFilter filter);

/**
 * @brief Exact k nearest neighbors of every query among the vectors of its label
 * (query_labels: an int32 device vector [n_queries]); requires labels (cuvsCertexSetLabels).
 */
CUVS_EXPORT cuvsError_t cuvsCertexSearchWithLabels(cuvsResources_t res,
                                                   cuvsCertexSearchParams_t params,
                                                   cuvsCertexIndex_t index,
                                                   DLManagedTensor* queries,
                                                   DLManagedTensor* query_labels,
                                                   DLManagedTensor* neighbors,
                                                   DLManagedTensor* distances);

/**
 * @}
 */

/**
 * @defgroup certex_c_index_serialize CertEx C-API serialize
 * @{
 */

/** @brief Save the index, including its vectors, to a file. */
CUVS_EXPORT cuvsError_t cuvsCertexSerialize(cuvsResources_t res,
                                            const char* filename,
                                            cuvsCertexIndex_t index);

/** @brief Load an index saved by cuvsCertexSerialize into an empty index handle; the vector type
 *  comes from the file. */
CUVS_EXPORT cuvsError_t cuvsCertexDeserialize(cuvsResources_t res,
                                              const char* filename,
                                              cuvsCertexIndex_t index);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif
