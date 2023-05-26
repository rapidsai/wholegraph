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

#include <torch/script.h>
#include <wholememory/env_func_ptrs.h>
#include <wholememory/wholememory_tensor.h>

namespace wholegraph_torch {

c10::ScalarType get_c10_scalar_type(wholememory_dtype_t wm_dtype);

wholememory_dtype_t get_wholememory_dtype(torch::ScalarType ts_dtype);

struct pytorch_memory_context {
  torch::Tensor tensor;
  torch::TensorOptions options;
  wholememory_tensor_description_t desc;
};

void set_need_grad(pytorch_memory_context* memory_context, bool require_grad);

void create_torch_memory_context_func(void** memory_context, void* /*global_context*/);

void destroy_torch_memory_context_func(void* memory_context, void* /*global_context*/);

void* torch_common_malloc_func(wholememory_tensor_description_t* tensor_description,
                               void* memory_context,
                               bool gpu_memory = true,
                               bool pinned     = false);

void torch_common_free_func(void* memory_context, void* /*global_context*/);

void get_tensor_desc_from_torch_tensor(wholememory_tensor_description_t* tensor_desc,
                                       const torch::Tensor& t);

void get_array_desc_from_torch_tensor(wholememory_array_description_t* array_desc,
                                      const torch::Tensor& t);

void get_matrix_desc_from_torch_tensor(wholememory_matrix_description_t* matrix_desc,
                                       const torch::Tensor& t);

class wrapped_torch_tensor {
 public:
  explicit wrapped_torch_tensor(const torch::Tensor& torch_tensor);
  ~wrapped_torch_tensor();
  wholememory_tensor_t get_wholememory_tensor() const;
  void unsqueeze(int dim = -1);
  void squeeze(int dim = -1);

 private:
  wholememory_tensor_t wholememory_tensor_ = nullptr;
};

void torch_tensor_check_dim_in_range(const torch::Tensor& t,
                                     int min_dim,
                                     int max_dim,
                                     const char* info);

inline void torch_tensor_check_dim(const torch::Tensor& t, int dim, const char* info)
{
  return torch_tensor_check_dim_in_range(t, dim, dim, info);
}

void torch_tensor_check_dtype(const torch::Tensor& t, torch::Dtype dtype, const char* info);

void torch_tensor_check_dtype_is_int(const torch::Tensor& t, const char* info);

// int32 or int64
void torch_tensor_check_dtype_is_index(const torch::Tensor& t, const char* info);

void torch_tensor_check_dtype_is_float(const torch::Tensor& t, const char* info);

}  // namespace wholegraph_torch
