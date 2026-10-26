/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

/* The C API, with the conventions of the cuVS C API: the exact 10 nearest neighbors of 1,000
 * queries among one million random vectors, checked on a sample of queries against exhaustive
 * search in double precision.
 *
 *   cmake --build build --target certex_example_c && build/certex_example_c
 */

#include <cuvs/neighbors/certex.h>

#include <cuda_runtime_api.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(call)                                               \
  do {                                                            \
    if ((call) != CUVS_SUCCESS) {                                 \
      fprintf(stderr, "%s: %s\n", #call, cuvsGetLastErrorText()); \
      return 1;                                                   \
    }                                                             \
  } while (0)

/* A DLPack view of a row-major device matrix. */
static void matrix(DLManagedTensor* t, void* data, int64_t* shape, uint8_t code, uint8_t bits)
{
  memset(t, 0, sizeof(*t));
  t->dl_tensor.data               = data;
  t->dl_tensor.device.device_type = kDLCUDA;
  t->dl_tensor.ndim               = 2;
  t->dl_tensor.dtype.code         = code;
  t->dl_tensor.dtype.bits         = bits;
  t->dl_tensor.dtype.lanes        = 1;
  t->dl_tensor.shape              = shape;
}

static uint64_t rng = 88172645463325252ull;
static float uniform(void)
{
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  return (float)(rng >> 40) / (float)(1 << 24) - 0.5f;
}

static int cmp(const void* a, const void* b)
{
  const double x = *(const double*)a, y = *(const double*)b;
  return (x > y) - (x < y);
}

int main(void)
{
  const int64_t n = 1000000, nq = 1000, d = 96, k = 10;
  float* base  = malloc(sizeof(float) * n * d);
  float* query = malloc(sizeof(float) * nq * d);
  for (int64_t i = 0; i < n * d; ++i)
    base[i] = uniform();
  for (int64_t i = 0; i < nq * d; ++i)
    query[i] = uniform();

  float *dx, *dq, *dd;
  int64_t* dn;
  cudaMalloc((void**)&dx, sizeof(float) * n * d);
  cudaMalloc((void**)&dq, sizeof(float) * nq * d);
  cudaMalloc((void**)&dn, sizeof(int64_t) * nq * k);
  cudaMalloc((void**)&dd, sizeof(float) * nq * k);
  cudaMemcpy(dx, base, sizeof(float) * n * d, cudaMemcpyHostToDevice);
  cudaMemcpy(dq, query, sizeof(float) * nq * d, cudaMemcpyHostToDevice);

  int64_t x_shape[2] = {n, d}, q_shape[2] = {nq, d}, o_shape[2] = {nq, k};
  DLManagedTensor x, q, nb, ds;
  matrix(&x, dx, x_shape, kDLFloat, 32);
  matrix(&q, dq, q_shape, kDLFloat, 32);
  matrix(&nb, dn, o_shape, kDLInt, 64);
  matrix(&ds, dd, o_shape, kDLFloat, 32);

  cuvsResources_t res;
  CHECK(cuvsResourcesCreate(&res));
  cuvsCertexIndexParams_t params; /* squared Euclidean distance, automatic screening width */
  CHECK(cuvsCertexIndexParamsCreate(&params));
  cuvsCertexIndex_t index;
  CHECK(cuvsCertexIndexCreate(&index));
  CHECK(cuvsCertexBuild(res, params, &x, index));

  cuvsCertexSearchParams_t search_params;
  CHECK(cuvsCertexSearchParamsCreate(&search_params));
  cuvsFilter filter = {0, NO_FILTER};
  CHECK(cuvsCertexSearch(res, search_params, index, &q, &nb, &ds, filter));
  CHECK(cuvsStreamSync(res));

  int64_t* ids = malloc(sizeof(int64_t) * nq * k);
  cudaMemcpy(ids, dn, sizeof(int64_t) * nq * k, cudaMemcpyDeviceToHost);

  int exact = 0, checked = 0;
  double* all = malloc(sizeof(double) * n);
  for (int64_t i = 0; i < nq; i += 50, ++checked) { /* exhaustive search in double precision on a
                                                       sample */
    double got[10];
    for (int64_t j = 0; j < n; ++j) {
      double s = 0;
      for (int64_t c = 0; c < d; ++c) {
        const double t = (double)query[i * d + c] - (double)base[j * d + c];
        s += t * t;
      }
      all[j] = s;
    }
    for (int64_t t = 0; t < k; ++t)
      got[t] = all[ids[i * k + t]];
    qsort(all, n, sizeof(double), cmp);
    qsort(got, k, sizeof(double), cmp);
    exact += memcmp(got, all, sizeof(double) * k) == 0;
  }
  printf("nearest neighbors of query 0:");
  for (int64_t t = 0; t < k; ++t)
    printf(" %ld", (long)ids[t]);
  printf("\nexact on %d of %d checked queries\n", exact, checked);

  CHECK(cuvsCertexSearchParamsDestroy(search_params));
  CHECK(cuvsCertexIndexDestroy(index));
  CHECK(cuvsCertexIndexParamsDestroy(params));
  CHECK(cuvsResourcesDestroy(res));
  cudaFree(dx);
  cudaFree(dq);
  cudaFree(dn);
  cudaFree(dd);
  free(base);
  free(query);
  free(ids);
  free(all);
  return exact == checked ? 0 : 1;
}
