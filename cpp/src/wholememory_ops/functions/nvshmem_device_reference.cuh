/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#ifdef WITH_NVSHMEM_SUPPORT

#include "nvshmem_template.cuh"
#include "wholememory/device_reference.cuh"

namespace wholememory_ops {

template <typename DataTypeT>
class nvshmem_device_reference {
 public:
  __device__ __forceinline__ explicit nvshmem_device_reference(
    const wholememory_nvshmem_ref_t& nvshmem_ref)
    : pointer_(static_cast<DataTypeT*>(nvshmem_ref.pointer)),
      typed_stride_(nvshmem_ref.stride / sizeof(DataTypeT))
  {
    assert(nvshmem_ref.stride % sizeof(DataTypeT) == 0);
  }

  __device__ nvshmem_device_reference() = delete;

  __device__ __forceinline__ DataTypeT load(size_t index)
  {
    size_t rank = index / typed_stride_;

    return nvshmem_get<DataTypeT>(pointer_ + index - rank * typed_stride_, rank);
  }

  __device__ __forceinline__ void store(size_t index, DataTypeT val)
  {
    size_t rank = index / typed_stride_;
    return nvshmem_put<DataTypeT>(pointer_ + index - rank * typed_stride_, val, rank);
  }

  __device__ __forceinline__ DataTypeT* symmetric_address(size_t index)
  {
    size_t rank = index / typed_stride_;
    return pointer_ + index - rank * typed_stride_;
  }

  __device__ __forceinline__ size_t dest_rank(size_t index)
  {
    size_t rank = index / typed_stride_;
    return rank;
  }

 private:
  DataTypeT* pointer_;
  size_t typed_stride_;
};
}  // namespace wholememory_ops

#endif
