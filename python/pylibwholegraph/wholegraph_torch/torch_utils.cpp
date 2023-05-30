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
#include "torch_utils.h"

#include <c10/cuda/CUDAFunctions.h>

namespace wholegraph_torch {

c10::ScalarType get_c10_scalar_type(wholememory_dtype_t wm_dtype)
{
  switch (wm_dtype) {
    case WHOLEMEMORY_DT_FLOAT: return c10::ScalarType::Float;
    case WHOLEMEMORY_DT_HALF: return c10::ScalarType::Half;
    case WHOLEMEMORY_DT_DOUBLE: return c10::ScalarType::Double;
    case WHOLEMEMORY_DT_BF16: return c10::ScalarType::BFloat16;
    case WHOLEMEMORY_DT_INT: return c10::ScalarType::Int;
    case WHOLEMEMORY_DT_INT64: return c10::ScalarType::Long;
    case WHOLEMEMORY_DT_INT16: return c10::ScalarType::Short;
    case WHOLEMEMORY_DT_INT8: return c10::ScalarType::Char;
    default: return c10::ScalarType::Undefined;
  }
}

wholememory_dtype_t get_wholememory_dtype(torch::ScalarType ts_dtype)
{
  switch (ts_dtype) {
    case c10::ScalarType::Float: return WHOLEMEMORY_DT_FLOAT;
    case c10::ScalarType::Half: return WHOLEMEMORY_DT_HALF;
    case c10::ScalarType::Double: return WHOLEMEMORY_DT_DOUBLE;
    case c10::ScalarType::BFloat16: return WHOLEMEMORY_DT_BF16;
    case c10::ScalarType::Int: return WHOLEMEMORY_DT_INT;
    case c10::ScalarType::Long: return WHOLEMEMORY_DT_INT64;
    case c10::ScalarType::Short: return WHOLEMEMORY_DT_INT16;
    case c10::ScalarType::Char: return WHOLEMEMORY_DT_INT8;
    default: return WHOLEMEMORY_DT_UNKNOWN;
  }
}

void set_need_grad(pytorch_memory_context* memory_context, bool require_grad)
{
  memory_context->options = memory_context->options.requires_grad(require_grad);
}

void create_torch_memory_context_func(void** memory_context, void* /*global_context*/)
{
  *memory_context = new pytorch_memory_context();
}

void destroy_torch_memory_context_func(void* memory_context, void* /*global_context*/)
{
  if (memory_context != nullptr) { delete static_cast<pytorch_memory_context*>(memory_context); }
}

void* torch_common_malloc_func(wholememory_tensor_description_t* tensor_description,
                               void* memory_context,
                               bool gpu_memory,
                               bool pinned)
{
  auto* pytorch_context = static_cast<pytorch_memory_context*>(memory_context);
  pytorch_context->desc = *tensor_description;
  std::vector<int64_t> shape(tensor_description->dim);
  for (int i = 0; i < tensor_description->dim; i++) {
    shape[i] = tensor_description->sizes[i];
  }
  pytorch_context->options =
    pytorch_context->options.dtype(get_c10_scalar_type(tensor_description->dtype));
  if (gpu_memory) {
    pytorch_context->options =
      pytorch_context->options.device(c10::Device(c10::kCUDA, c10::cuda::current_device()));
  } else {
    pytorch_context->options = pytorch_context->options.device(c10::Device(c10::kCPU));
    pytorch_context->options = pytorch_context->options.pinned_memory(pinned);
  }
  try {
    pytorch_context->tensor = torch::empty(shape, pytorch_context->options);
  } catch (c10::Error& err) {
    fprintf(stderr, "torch_common_malloc_func allocation failed. Reasion=%s", err.what());
    throw err;
  }
  return pytorch_context->tensor.data_ptr();
}

void torch_common_free_func(void* memory_context, void* /*global_context*/)
{
  static_cast<pytorch_memory_context*>(memory_context)->tensor  = torch::Tensor();
  static_cast<pytorch_memory_context*>(memory_context)->options = torch::TensorOptions();
  wholememory_initialize_tensor_desc(&static_cast<pytorch_memory_context*>(memory_context)->desc);
}

void get_tensor_desc_from_torch_tensor(wholememory_tensor_description_t* tensor_desc,
                                       const torch::Tensor& t)
{
  tensor_desc->dim   = t.dim();
  tensor_desc->dtype = get_wholememory_dtype(t.dtype().toScalarType());
  TORCH_CHECK(tensor_desc->dtype != WHOLEMEMORY_DT_UNKNOWN);
  tensor_desc->storage_offset = t.storage_offset();
  for (int i = 0; i < tensor_desc->dim; i++) {
    tensor_desc->sizes[i]   = t.size(i);
    tensor_desc->strides[i] = t.stride(i);
  }
}

void get_array_desc_from_torch_tensor(wholememory_array_description_t* array_desc,
                                      const torch::Tensor& t)
{
  TORCH_CHECK(t.dim() == 1, "get_array_desc_from_torch_tensor: should be 1-dim tensor");
  array_desc->dtype = get_wholememory_dtype(t.dtype().toScalarType());
  TORCH_CHECK(array_desc->dtype != WHOLEMEMORY_DT_UNKNOWN);
  array_desc->size           = t.size(0);
  array_desc->storage_offset = t.storage_offset();
}

void get_matrix_desc_from_torch_tensor(wholememory_matrix_description_t* matrix_desc,
                                       const torch::Tensor& t)
{
  TORCH_CHECK(t.dim() == 2, "get_matrix_desc_from_torch_tensor: should be 2-dim tensor");
  matrix_desc->dtype = get_wholememory_dtype(t.dtype().toScalarType());
  TORCH_CHECK(matrix_desc->dtype != WHOLEMEMORY_DT_UNKNOWN);
  matrix_desc->sizes[0]       = t.size(0);
  matrix_desc->sizes[1]       = t.size(1);
  matrix_desc->stride         = t.stride(0);
  matrix_desc->storage_offset = t.storage_offset();
}

wrapped_torch_tensor::wrapped_torch_tensor(const torch::Tensor& torch_tensor)
{
  wholememory_tensor_description_t tensor_description;
  get_tensor_desc_from_torch_tensor(&tensor_description, torch_tensor);
  wholememory_make_tensor_from_pointer(
    &wholememory_tensor_, torch_tensor.storage().data(), &tensor_description);
}

wrapped_torch_tensor::~wrapped_torch_tensor()
{
  wholememory_destroy_tensor(wholememory_tensor_);
  wholememory_tensor_ = nullptr;
}

wholememory_tensor_t wrapped_torch_tensor::get_wholememory_tensor() const
{
  return wholememory_tensor_;
}

void wrapped_torch_tensor::unsqueeze(int dim)
{
  auto* tensor_desc = wholememory_tensor_get_tensor_description(wholememory_tensor_);
  TORCH_CHECK(dim >= -tensor_desc->dim - 1 && dim <= tensor_desc->dim,
              "dim = ",
              dim,
              " but t.dim()=",
              tensor_desc->dim,
              ", should in range [",
              -tensor_desc->dim - 1,
              ", ",
              tensor_desc->dim,
              "]")
  if (dim < 0) { dim += tensor_desc->dim + 1; }
  TORCH_CHECK(wholememory_unsqueeze_tensor(tensor_desc, dim), "unsqueeze failed.")
}

void wrapped_torch_tensor::squeeze(int dim)
{
  auto* tensor_desc = wholememory_tensor_get_tensor_description(wholememory_tensor_);
  TORCH_CHECK(dim >= -tensor_desc->dim && dim < tensor_desc->dim,
              "dim = ",
              dim,
              " but t.dim()=",
              tensor_desc->dim,
              ", should in range [",
              -tensor_desc->dim,
              ", ",
              tensor_desc->dim,
              ")")
  if (dim < 0) { dim += tensor_desc->dim; }
  TORCH_CHECK(tensor_desc->sizes[dim] == 1, "dim size should be 1")
  TORCH_CHECK(
    dim == tensor_desc->dim - 1 || tensor_desc->strides[dim] == tensor_desc->strides[dim + 1],
    "stride should be same as next dim")
  TORCH_CHECK(wholememory_squeeze_tensor(tensor_desc, dim))
}

void torch_tensor_check_dim_in_range(const torch::Tensor& t,
                                     int min_dim,
                                     int max_dim,
                                     const char* info)
{
  TORCH_CHECK(t.dim() >= min_dim && t.dim() <= max_dim,
              std::string(info),
              " dim=",
              t.dim(),
              ", should in range [",
              min_dim,
              ", ",
              max_dim,
              "]")
}

void torch_tensor_check_dtype(const torch::Tensor& t, torch::Dtype dtype, const char* info)
{
  TORCH_CHECK(t.dtype() == dtype, std::string(info), " should be ", dtype, " but got ", t.dtype());
}

void torch_tensor_check_dtype_is_int(const torch::Tensor& t, const char* info)
{
  TORCH_CHECK(t.dtype() == torch::kInt8 || t.dtype() == torch::kInt16 ||
                t.dtype() == torch::kInt32 || t.dtype() == torch::kInt64,
              std::string(info),
              " should be integer.")
}

// int32 or int64
void torch_tensor_check_dtype_is_index(const torch::Tensor& t, const char* info)
{
  TORCH_CHECK(t.dtype() == torch::kInt32 || t.dtype() == torch::kInt64,
              std::string(info),
              " should be int32 or int64.")
}

void torch_tensor_check_dtype_is_float(const torch::Tensor& t, const char* info)
{
  TORCH_CHECK(t.dtype() == torch::kFloat16 || t.dtype() == torch::kBFloat16 ||
                t.dtype() == torch::kFloat32 || t.dtype() == torch::kFloat64,
              std::string(info),
              " should be float tensor.")
}

}  // namespace wholegraph_torch
