/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// A plain C caller of the CertEx C API (compiled as C): build an index over device float32
// vectors, optionally save and reload it, and search it with or without a bitset filter.

#include <cuvs/neighbors/certex.h>

#include <stdint.h>
#include <stdio.h>

static void matrix(DLManagedTensor* t, void* data, int64_t* shape, uint8_t code, uint8_t bits)
{
  t->dl_tensor.data               = data;
  t->dl_tensor.device.device_type = kDLCUDA;
  t->dl_tensor.device.device_id   = 0;
  t->dl_tensor.ndim               = 2;
  t->dl_tensor.dtype.code         = code;
  t->dl_tensor.dtype.bits         = bits;
  t->dl_tensor.dtype.lanes        = 1;
  t->dl_tensor.shape              = shape;
  t->dl_tensor.strides            = NULL;
  t->dl_tensor.byte_offset        = 0;
  t->manager_ctx                  = NULL;
  t->deleter                      = NULL;
}

#define CHECK(call)                                               \
  do {                                                            \
    if ((call) != CUVS_SUCCESS) {                                 \
      fprintf(stderr, "%s: %s\n", #call, cuvsGetLastErrorText()); \
      return 1;                                                   \
    }                                                             \
  } while (0)

int run_certex(int64_t n_rows,
               int64_t n_queries,
               int64_t dim,
               int64_t k,
               float* dataset,
               float* queries,
               uint32_t* bitset,
               int32_t n_lists,
               int32_t local_projections,
               const char* save_to,
               int64_t* neighbors,
               float* distances)
{
  cuvsResources_t res;
  CHECK(cuvsResourcesCreate(&res));

  int64_t x_shape[2] = {n_rows, dim}, q_shape[2] = {n_queries, dim}, o_shape[2] = {n_queries, k};
  DLManagedTensor x, q, nb, ds, words;
  matrix(&x, dataset, x_shape, kDLFloat, 32);
  matrix(&q, queries, q_shape, kDLFloat, 32);
  matrix(&nb, neighbors, o_shape, kDLInt, 64);
  matrix(&ds, distances, o_shape, kDLFloat, 32);

  cuvsCertexIndexParams_t ip;
  CHECK(cuvsCertexIndexParamsCreate(&ip));
  ip->width             = 16;
  ip->n_lists           = n_lists;
  ip->local_projections = local_projections != 0;
  cuvsCertexIndex_t index;
  CHECK(cuvsCertexIndexCreate(&index));
  CHECK(cuvsCertexBuild(res, ip, &x, index));

  int64_t size = 0, d = 0;
  CHECK(cuvsCertexIndexGetSize(index, &size));
  CHECK(cuvsCertexIndexGetDim(index, &d));
  if (size != n_rows || d != dim) { return 2; }

  if (save_to != NULL) {  // reload: the search below runs on the loaded index
    CHECK(cuvsCertexSerialize(res, save_to, index));
    CHECK(cuvsCertexIndexDestroy(index));
    CHECK(cuvsCertexIndexCreate(&index));
    CHECK(cuvsCertexDeserialize(res, save_to, index));
  }

  cuvsFilter filter;
  filter.type        = NO_FILTER;
  filter.addr        = 0;
  int64_t w_shape[1] = {(n_rows + 31) / 32};
  if (bitset != NULL) {
    matrix(&words, bitset, w_shape, kDLUInt, 32);
    words.dl_tensor.ndim = 1;
    filter.type          = BITSET;
    filter.addr          = (uintptr_t)&words;
  }
  cuvsCertexSearchParams_t sp;
  CHECK(cuvsCertexSearchParamsCreate(&sp));
  CHECK(cuvsCertexSearch(res, sp, index, &q, &nb, &ds, filter));

  CHECK(cuvsCertexSearchParamsDestroy(sp));
  CHECK(cuvsCertexIndexDestroy(index));
  CHECK(cuvsCertexIndexParamsDestroy(ip));
  CHECK(cuvsResourcesDestroy(res));
  return 0;
}
