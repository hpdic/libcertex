# C++ API

The Python package wraps a CUDA C++ library, `cuvs::neighbors::certex`, which follows the conventions of the cuVS
index modules (CAGRA, IVF-Flat, brute force): parameter structs, an index type templated on the vector type, and free
functions `build`, `extend`, `search`, `serialize`, and `deserialize` over RAFT mdspan views. The header is
`cpp/include/cuvs/neighbors/certex.hpp`.

```cpp
#include <cuvs/neighbors/certex.hpp>

using namespace cuvs::neighbors;

raft::device_resources res;
certex::index_params params;                       // L2Expanded, automatic width
params.metric = cuvs::distance::DistanceType::L2Expanded;
auto index = certex::build(res, params, dataset);  // device or host matrix view [n, dim]

certex::extend(res, more_vectors, &index);         // ids index.size(), index.size() + 1, ...
certex::remove(res, ids, &index);                  // removed ids never return

certex::search_params sp;                          // 128 anchors, batches of 2,048 queries
certex::search(res, sp, index, queries, neighbors, distances);
```

The vector type is `float`, `half`, `int8_t`, or `uint8_t`; the metrics are `L2Expanded`, `InnerProduct`, and
`CosineExpanded`, with the distance conventions of `cuvs::neighbors::brute_force`. A search takes a cuVS sample filter
(`bitset_filter` over the vectors, or `bitmap_filter` with one row per query), or one label per query after
`certex::set_labels`. Searches may run concurrently with `extend`, `remove`, and `set_labels`, each thread with its
own `raft::resources`. `certex::partition_by_labels` sorts a copy of the screening rows by label, after which a
search with query labels screens only the rows of each query's label (vectors added later are screened with the
label filter until the next call).

Host-memory datasets are streamed to the GPU in chunks; with `params.host_vectors`, the full-precision vectors stay in
pinned host memory and the GPU holds only the screening rows and the anchors (about 2 (m + 5) bytes per vector).

## Example

`examples/cpp/exact_search.cu` is a complete program: it builds an index over one million random vectors, searches
1,000 queries, and checks a sample against exhaustive search in double precision. The build below compiles it as
`build/certex_example_cpp` (option `CERTEX_BUILD_EXAMPLES`, on by default).

## Building and testing

The library builds with CMake against an installed cuVS, for example the `libcuvs-cu12` wheel; `cpp/CMakeLists.txt`
lists the options. The tests use GoogleTest, as cuVS does:

```bash
cmake -S cpp -B build -G Ninja -DCMAKE_PREFIX_PATH="<libcuvs, libraft, librmm, rapids_logger, libkvikio cmake dirs>"
cmake --build build
ctest --test-dir build                 # NEIGHBORS_ANN_CERTEX_TEST and NEIGHBORS_CERTEX_KERNELS_TEST
```

`NEIGHBORS_ANN_CERTEX_TEST` checks every result against exhaustive search in double precision for the four vector
types, the three metrics, host builds and host vectors, extend, k up to 2,048, removals, labels, bitset and bitmap
filters, serialization, and searches concurrent with extend. `NEIGHBORS_CERTEX_KERNELS_TEST` checks both kernels of
the certified pass (cooperative and warp-specialized on Hopper, and the sm80 kernels, which also run on Hopper)
against fp64 products, and the suite runs the whole search once more with the sm80 kernels. The benchmarks are `certex_bigann`
(BIGANN prefixes against the official ground truth, with `--host` and `--latency`) and `certex_kernel_bench`. Built with
`-DCERTEX_NVTX=ON`, the library marks build, extend, remove, search, and the stages of every search batch with NVTX
ranges of the `cuvs` domain, which Nsight Systems shows.

## Reference documentation

`cpp/Doxyfile` builds the reference of the C++ and C headers with Doxygen (`cd cpp && doxygen Doxyfile`, output in
`cpp/build/doxygen/html`); the build reports no documentation errors.
