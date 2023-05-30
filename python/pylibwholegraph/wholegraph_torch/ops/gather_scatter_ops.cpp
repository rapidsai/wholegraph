#include <torch/script.h>

#include <c10/cuda/CUDAStream.h>
#include <wholememory/wholememory_op.h>
#include <wholememory/wholememory_tensor.h>

#include "../torch_env_func_ptrs.h"
#include "../torch_utils.h"

namespace wholegraph_torch {

torch::Tensor gather(int64_t wholememory_tensor_handle,
                     const torch::Tensor& indices,
                     torch::optional<torch::ScalarType> output_type,
                     torch::optional<bool> requires_grad)
{
  torch_tensor_check_dim(indices, 1, "gather indice");
  torch_tensor_check_dtype_is_index(indices, "gather, indices");

  wrapped_torch_tensor const wrapped_indices_tensor(indices);
  auto wt                = reinterpret_cast<wholememory_tensor_t>(wholememory_tensor_handle);
  auto* p_wm_tensor_desc = wholememory_tensor_get_tensor_description(wt);
  auto* p_indices_desc =
    wholememory_tensor_get_tensor_description(wrapped_indices_tensor.get_wholememory_tensor());
  TORCH_CHECK(p_wm_tensor_desc->dim == 1 || p_wm_tensor_desc->dim == 2,
              "wholememory_tensor_handle should be 1D or 2D WholeMemory Tensor.")

  wholememory_dtype_t wm_output_type = p_wm_tensor_desc->dtype;
  if (output_type.has_value()) { wm_output_type = get_wholememory_dtype(output_type.value()); }

  wholememory_tensor_description_t output_alloc_tensor_desc;
  output_alloc_tensor_desc.dtype                                     = wm_output_type;
  output_alloc_tensor_desc.dim                                       = p_wm_tensor_desc->dim;
  output_alloc_tensor_desc.storage_offset                            = 0;
  output_alloc_tensor_desc.sizes[0]                                  = p_indices_desc->sizes[0];
  output_alloc_tensor_desc.strides[output_alloc_tensor_desc.dim - 1] = 1;
  if (p_wm_tensor_desc->dim == 2) {
    output_alloc_tensor_desc.sizes[1] = output_alloc_tensor_desc.strides[0] =
      p_wm_tensor_desc->sizes[1];
  }

  pytorch_memory_context output_context;
  if (requires_grad.has_value()) { set_need_grad(&output_context, requires_grad.value()); }
  torch_common_malloc_func(&output_alloc_tensor_desc, &output_context);

  auto output_tensor = output_context.tensor;
  wrapped_torch_tensor const wrapped_output_tensor(output_tensor);

  TORCH_CHECK(wholememory_gather(wt,
                                 wrapped_indices_tensor.get_wholememory_tensor(),
                                 wrapped_output_tensor.get_wholememory_tensor(),
                                 wholegraph_torch::get_pytorch_env_func(),
                                 wholegraph_torch::get_current_stream()) == WHOLEMEMORY_SUCCESS)
  return output_tensor;
}

void scatter(const torch::Tensor& input,
             const torch::Tensor& indices,
             int64_t wholememory_tensor_handle)
{
  torch_tensor_check_dim_in_range(input, 1, 2, "scatter input");
  torch_tensor_check_dim(indices, 1, "scatter indice");
  torch_tensor_check_dtype_is_index(indices, "scatter, indices");

  wrapped_torch_tensor const wrapped_indices_tensor(indices);
  wrapped_torch_tensor const wrapped_input_tensor(input);
  auto wt                = reinterpret_cast<wholememory_tensor_t>(wholememory_tensor_handle);
  auto* p_wm_tensor_desc = wholememory_tensor_get_tensor_description(wt);
  TORCH_CHECK(p_wm_tensor_desc->dim == input.dim(),
              "input and wholememory_tensor_hand should be same dim.")

  if (input.dim() == 2) {
    TORCH_CHECK(input.size(1) == p_wm_tensor_desc->sizes[1],
                "input and wholememory should have same embedding size but input.size(1)=%ld, "
                "wholememory.size(1)=%ld",
                input.size(1),
                p_wm_tensor_desc->sizes[1])
  }

  TORCH_CHECK(wholememory_scatter(wrapped_input_tensor.get_wholememory_tensor(),
                                  wrapped_indices_tensor.get_wholememory_tensor(),
                                  wt,
                                  wholegraph_torch::get_pytorch_env_func(),
                                  wholegraph_torch::get_current_stream()) == WHOLEMEMORY_SUCCESS)
}

}  // namespace wholegraph_torch

static auto registry = torch::RegisterOperators()
                         .op("wholegraph::gather", &wholegraph_torch::gather)
                         .op("wholegraph::scatter", &wholegraph_torch::scatter);
