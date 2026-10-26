/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// The subset of cuVS c/src/core/exceptions.hpp that the CertEx C API uses, with the same names, so
// that c/src/neighbors/certex.cpp moves into cuVS unchanged (cuVS then provides this header).

#pragma once

#include <cuvs/core/c_api.h>

#include <exception>

namespace cuvs::core {

template <typename Fn>
cuvsError_t translate_exceptions(Fn func)
{
  cuvsError_t status;
  try {
    func();
    status = CUVS_SUCCESS;
    cuvsSetLastErrorText(NULL);
  } catch (const std::exception& e) {
    cuvsSetLastErrorText(e.what());
    status = CUVS_ERROR;
  } catch (...) {
    cuvsSetLastErrorText("unknown exception");
    status = CUVS_ERROR;
  }
  return status;
}

}  // namespace cuvs::core
