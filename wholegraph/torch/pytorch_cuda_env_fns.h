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

}

}