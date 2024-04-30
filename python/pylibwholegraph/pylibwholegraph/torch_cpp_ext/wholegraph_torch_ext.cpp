/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <torch/extension.h>
#include <torch/script.h>

#include "torch_env_func_ptrs.h"
#include "torch_utils.h"

int64_t wrapped_get_wholegraph_env_fns()
{
  return reinterpret_cast<int64_t>(static_cast<void*>(wholegraph_torch::get_pytorch_env_func()));
}

int64_t wrapped_get_stream()
{
  return reinterpret_cast<int64_t>(static_cast<void*>(wholegraph_torch::get_current_stream()));
}

int64_t wrapped_create_output_context()
{
  return reinterpret_cast<int64_t>(wholegraph_torch::create_output_context());
}

void wrapped_destroy_output_context(int64_t output_context)
{
  wholegraph_torch::destroy_output_context(reinterpret_cast<void*>(output_context));
}

void wrapped_free_context_data(int64_t output_context)
{
  wholegraph_torch::free_context_data(reinterpret_cast<void*>(output_context), nullptr);
}

torch::Tensor get_torch_tensor_from_output_context(int64_t output_context)
{
  auto* torch_output_context =
      static_cast<wholegraph_torch::pytorch_memory_context*>(reinterpret_cast<void*>(output_context));
  return torch_output_context->tensor;
}

PYBIND11_MODULE(pylibwholegraph_torch_ext, m)
{
  m.def("get_wholegraph_env_fns",
        &wrapped_get_wholegraph_env_fns,
        "Get WholeGraph Environment functions.");
  m.def("get_stream", &wrapped_get_stream, "Get current CUDA stream.");
  m.def("create_output_context", &wrapped_create_output_context, "Create output memory context.");
  m.def("destroy_output_context", &wrapped_destroy_output_context, "Destroy output memory context.");
  m.def("free_context_data", &wrapped_free_context_data, "Free data in output memory context.");
  m.def("get_tensor_from_context",
        &get_torch_tensor_from_output_context,
        "Get PyTorch Tensor from output memory context");
}
