/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */

// The subset of cuVS c/src/core/interop.hpp that the CertEx C API uses, with the same names, so
// that c/src/neighbors/certex.cpp moves into cuVS unchanged (cuVS then provides this header):
// dense row-major DLPack tensors to RAFT mdspans, with checks of the device, the data type, and the
// shape.

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdspan.hpp>

#include <cuvs/core/cuda_fp16.hpp>  // __half with default visibility, as in cuVS
#include <dlpack/dlpack.h>

#include <cstdint>
#include <type_traits>

namespace cuvs::core {

inline bool is_dlpack_device_compatible(DLTensor tensor)
{
  return tensor.device.device_type == kDLCUDA || tensor.device.device_type == kDLCUDAManaged ||
         tensor.device.device_type == kDLCUDAHost;
}

inline bool is_dlpack_host_compatible(DLTensor tensor)
{
  return tensor.device.device_type == kDLCPU || tensor.device.device_type == kDLCUDAHost ||
         tensor.device.device_type == kDLCUDAManaged;
}

namespace detail {

template <typename T>
DLDataTypeCode dlpack_code()
{
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half> || std::is_same_v<T, double>) {
    return kDLFloat;
  } else if constexpr (std::is_signed_v<T>) {
    return kDLInt;
  } else {
    return kDLUInt;
  }
}

template <typename T>
void check_tensor(DLTensor const& t, int ndim)
{
  RAFT_EXPECTS(t.ndim == ndim, "certex: the DLPack tensor has the wrong number of dimensions");
  RAFT_EXPECTS(
    t.dtype.code == dlpack_code<T>() && t.dtype.bits == 8 * sizeof(T) && t.dtype.lanes == 1,
    "certex: the DLPack tensor has the wrong data type");
  if (t.strides != nullptr) {  // only dense row-major tensors
    int64_t expect = 1;
    for (int i = ndim - 1; i >= 0; --i) {
      RAFT_EXPECTS(t.shape[i] == 1 || t.strides[i] == expect,
                   "certex: the DLPack tensor must be dense and row major");
      expect *= t.shape[i];
    }
  }
}

template <typename T>
T* data_of(DLTensor const& t)
{
  return reinterpret_cast<T*>(static_cast<char*>(t.data) + t.byte_offset);
}

}  // namespace detail

/** A dense row-major DLPack tensor as a RAFT mdspan (matrix or vector, device or host). */
template <typename MdspanType>
MdspanType from_dlpack(DLManagedTensor* managed_tensor)
{
  using value_type  = typename MdspanType::value_type;
  using index_type  = typename MdspanType::index_type;
  using accessor    = typename MdspanType::accessor_type;
  DLTensor const& t = managed_tensor->dl_tensor;
  if constexpr (accessor::is_device_accessible) {
    RAFT_EXPECTS(is_dlpack_device_compatible(t), "certex: expected a tensor in device memory");
  } else {
    RAFT_EXPECTS(is_dlpack_host_compatible(t), "certex: expected a tensor in host memory");
  }
  using plain = std::remove_const_t<value_type>;
  detail::check_tensor<plain>(t, int(MdspanType::rank()));
  auto* p = detail::data_of<plain>(t);
  if constexpr (MdspanType::rank() == 2) {
    return MdspanType(p, index_type(t.shape[0]), index_type(t.shape[1]));
  } else {
    return MdspanType(p, index_type(t.shape[0]));
  }
}

}  // namespace cuvs::core
