# C API

The C API follows the cuVS C API (`cuvsBruteForce*`, `cuvsCagra*`): parameter structs with `Create` and `Destroy`,
an opaque index handle, DLPack tensors for every array, a `cuvsResources_t` handle from `cuvsResourcesCreate`, and
a `cuvsError_t` result whose message comes from `cuvsGetLastErrorText`. The header is
`c/include/cuvs/neighbors/certex.h`; the library `libcertex_c` links against `libcuvs_c`.

```c
#include <cuvs/neighbors/certex.h>

cuvsResources_t res;
cuvsResourcesCreate(&res);

cuvsCertexIndexParams_t params;
cuvsCertexIndexParamsCreate(&params);   /* defaults of the C++ API: L2Expanded, automatic width */
params->n_lists = 1024;                 /* optional: k-means lists, the nearest of which every query probes */

cuvsCertexIndex_t index;
cuvsCertexIndexCreate(&index);
cuvsCertexBuild(res, params, &dataset, index);   /* float32, float16, int8, or uint8; device or host */

cuvsCertexSearchParams_t sp;
cuvsCertexSearchParamsCreate(&sp);
cuvsFilter filter = {0, NO_FILTER};              /* or a BITSET over the ids, or a BITMAP per query */
cuvsCertexSearch(res, sp, index, &queries, &neighbors, &distances, filter);

cuvsCertexSerialize(res, "index.certex", index);
cuvsCertexIndexDestroy(index);
```

The other calls are `cuvsCertexExtend` (append vectors), `cuvsCertexRemove` (an int64 device vector of ids),
`cuvsCertexSetLabels` and `cuvsCertexPartitionByLabels` (labels for filtered search), `cuvsCertexSearchWithLabels`
(one label per query), `cuvsCertexDeserialize` (the vector type comes from the file), `cuvsCertexIndexGetSize`, and
`cuvsCertexIndexGetDim`. Every search returns the exact k nearest neighbors, with the distance conventions of cuVS
brute force.

The CMake option `CERTEX_BUILD_C_API` (on by default) builds the library and the test `NEIGHBORS_CERTEX_C_TEST`,
whose driver `c/tests/neighbors/run_certex_c.c` compiles as C.

`examples/c/exact_search.c` is a complete C program on this API (device buffers from `cudaMalloc`, DLPack views,
build, search, and a check against exhaustive search in double precision); the C++ build compiles it as
`build/certex_example_c`.
