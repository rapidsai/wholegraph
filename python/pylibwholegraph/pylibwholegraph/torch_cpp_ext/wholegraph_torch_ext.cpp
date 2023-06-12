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
  m.def("get_tensor_from_context",
        &get_torch_tensor_from_output_context,
        "Get PyTorch Tensor from output memory context");
}
