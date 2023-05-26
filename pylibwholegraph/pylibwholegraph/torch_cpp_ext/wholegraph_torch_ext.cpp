#include <torch/extension.h>

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

PYBIND11_MODULE(pylibwholegraph_torch_ext, m)
{
  m.def("get_wholegraph_env_fns",
        &wrapped_get_wholegraph_env_fns,
        "Get WholeGraph Environment functions.");
  m.def("get_stream", &wrapped_get_stream, "Get current CUDA stream.");
}
