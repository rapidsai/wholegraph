/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include <assert.h>
#include <cstddef>

#include "global_reference.h"
namespace wholememory {

template <typename DataTypeT>
class device_reference {
 public:
  __device__ __forceinline__ explicit device_reference(const wholememory_gref_t& gref)
    : pointer_(static_cast<DataTypeT*>(gref.pointer)),
      typed_stride_(gref.stride / sizeof(DataTypeT))
  {
    assert(gref.stride % sizeof(DataTypeT) == 0);
  }
  __device__ device_reference() = delete;

  __device__ __forceinline__ DataTypeT& operator[](size_t index)
  {
    if (typed_stride_ == 0) { return pointer_[index]; }
    size_t rank = index / typed_stride_;
    return static_cast<DataTypeT**>(
      static_cast<void*>(pointer_))[rank][index - rank * typed_stride_];
  }

 private:
  DataTypeT* pointer_;
  size_t typed_stride_;
};

}  // namespace wholememory
