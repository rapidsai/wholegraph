/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <torch/script.h>

#include "cuda_env_fns.h"

namespace whole_graph {

namespace pytorch {

whole_graph::CUDAEnvFns GetCUDAEnvFns(torch::Device d);

template<typename T>
inline std::function<T *(size_t)> GetAllocatorForTensor(torch::Tensor &t,
                                                        torch::Device d,
                                                        c10::optional<c10::ScalarType> dtype,
                                                        bool need_grad) {
  std::function<T *(size_t)> fn = [=, &t](size_t elt_count) {
    auto to = torch::TensorOptions().device(d).dtype(dtype).requires_grad(need_grad);
    t = torch::empty({(long) elt_count}, to);
    T *output_ptr = static_cast<T *>(t.data_ptr());
    return output_ptr;
  };
  return fn;
}

}// namespace pytorch

}// namespace whole_graph