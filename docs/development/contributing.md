# Contributing to certex

Thank you for helping. certex has one rule that every change must keep: every search returns the exact k nearest
neighbors. A change that makes search faster but returns a wrong neighbor for one query is a bug.

## Development setup

```bash
git clone https://github.com/hpdic/libcertex.git && cd libcertex
python -m pip install -e ".[dev]"   # compiles the CUDA library (see docs/installation.md)
pre-commit install            # ruff checks and formatting on every commit
```

A GPU of compute capability 8.0 or newer is needed for the tests (`pytest -m gpu`), which run every exactness check against
exhaustive search in double precision; `pytest -m "not gpu"` runs the checks that need no GPU.
The C++ tests (`certex_exact_test`, `certex_kernel_check`) build with CMake from `cpp/`.

## Making a change

1. Open an issue first for anything beyond a small fix, so that the design can be agreed on before the code.
2. Keep the exactness test in `tests/` green, and add a test that would fail without your change.
3. A change to the certificate (`cpp/src/neighbors/certex/`) must explain in its comments why every excluded vector
   still cannot be a neighbor; reviews of such changes check that argument first.
4. Add an entry under `[Unreleased]` in `CHANGELOG.md`.
5. Every source file starts with the copyright and license header of the existing files.

## Sign your commits (DCO)

certex uses the [Developer Certificate of Origin](https://developercertificate.org/), as Ray does. Sign every commit
with `git commit -s`, which adds a line `Signed-off-by: Your Name <your@email>` and certifies that you have the right
to submit the change under the Apache License 2.0.

## Pull requests

- One topic per pull request, with a description of what changed and how it was tested (exactness and speed).
- Performance claims come with the command that measures them (`bench/`) and the GPU they ran on.
- CI must pass: lint, tests without a GPU, and the documentation build.

## Code of conduct

Participation is governed by the [Code of Conduct](https://github.com/hpdic/libcertex/blob/main/CODE_OF_CONDUCT.md).
