# Releases and compatibility

## Versions

certex follows semantic versioning. Before 1.0, a minor release (0.x) may change the interface, and every such change
is listed in `CHANGELOG.md` with the way to migrate. From 1.0 on, the public interface changes incompatibly only in a
major release: the Python package (`certex.search`, `certex.verify`, `certex.Index`, `certex.ShardedIndex`,
`certex.auto_width`, `certex.tune`, `certex.topk`, `certex.KeyCache`) and its adapters (`certex.cuvs`, `certex.faiss`,
`certex.flashinfer`), the C++ module `cuvs::neighbors::certex`, and the C API `cuvsCertex*`.

## Deprecations

A feature that will be removed first emits a `DeprecationWarning` for at least one minor release, with the replacement
named in the warning and in `CHANGELOG.md`.

## Making a release

1. Move the `[Unreleased]` entries of `CHANGELOG.md` under the new version with its date.
2. Set `__version__` in `certex/__init__.py` and `version` in `pyproject.toml` to the same value.
3. Run `pytest -m gpu` on a GPU and the CI checks; record the GPU model in the release notes.
4. Tag the release (`git tag -s vX.Y.Z`) and build the distribution (`python -m build`).

## Wheels

The manual workflow `wheel` (`.github/workflows/wheel.yml`) builds `manylinux_2_28` wheels for Python 3.11 to 3.13 in
the PyPA manylinux image with the CUDA 12.9 compiler, for compute capability 8.0 (which also runs on 8.6 and 8.9),
9.0a, 10.0 (family), and 12.0a, with PTX for 12.0, and repairs them with auditwheel; the CUDA, cuVS, RAFT, and RMM libraries stay external and
come from the NVIDIA and RAPIDS wheels that the package depends on. A wheel built on a newer distribution carries
only the platform tag of its glibc (for example `manylinux_2_38` on Ubuntu 24.04).
