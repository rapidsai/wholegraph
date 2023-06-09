#include <c10/cuda/CUDAStream.h>
#include <random>
#include <torch/script.h>
#include <wholememory/wholegraph_op.h>
#include <wholememory/wholememory_tensor.h>

#include "../torch_env_func_ptrs.h"
#include "../torch_utils.h"

using torch::autograd::variable_list;

namespace wholegraph_torch {

variable_list weighted_sample_without_replacement(int64_t csr_row_ptr_wholememory_tensor_handle,
                                                  int64_t csr_col_ptr_wholememory_tensor_handle,
                                                  int64_t csr_weight_ptr_wholememory_tensor_handle,
                                                  torch::Tensor& input_nodes,
                                                  int64_t max_sample_count,
                                                  torch::optional<int64_t> random_seed)
{
  torch_tensor_check_dim(input_nodes, 1, "weighted_sample_without_replacement, input_nodes");
  torch_tensor_check_dtype_is_index(input_nodes,
                                    "weighted_sample_without_replacement, input_nodes");
  wrapped_torch_tensor const wrapped_input_nodes_tensor(input_nodes);

  auto wm_csr_row_ptr_tensor =
    reinterpret_cast<wholememory_tensor_t>(csr_row_ptr_wholememory_tensor_handle);
  auto wm_csr_col_ptr_tensor =
    reinterpret_cast<wholememory_tensor_t>(csr_col_ptr_wholememory_tensor_handle);
  auto wm_csr_weight_ptr_tensor =
    reinterpret_cast<wholememory_tensor_t>(csr_weight_ptr_wholememory_tensor_handle);
  auto* p_wm_csr_row_ptr_tensor_desc =
    wholememory_tensor_get_tensor_description(wm_csr_row_ptr_tensor);
  auto* p_wm_csr_col_ptr_tensor_desc =
    wholememory_tensor_get_tensor_description(wm_csr_col_ptr_tensor);
  auto* p_wm_csr_weight_ptr_tensor_desc =
    wholememory_tensor_get_tensor_description(wm_csr_weight_ptr_tensor);
  auto* p_input_nodes_desc =
    wholememory_tensor_get_tensor_description(wrapped_input_nodes_tensor.get_wholememory_tensor());

  TORCH_CHECK(p_wm_csr_row_ptr_tensor_desc->dim == 1,
              "csr_row_ptr_wholememory_tensor_handle should be 1D WholeMemory Tensor.")
  TORCH_CHECK(p_wm_csr_col_ptr_tensor_desc->dim == 1,
              "csr_col_ptr_wholememory_tensor_handle should be 1D WholeMemory Tensor.")
  TORCH_CHECK(p_wm_csr_weight_ptr_tensor_desc->dim == 1,
              "csr_weight_ptr_wholememory_tensor_handle should be 1D WholeMemory Tensor.")

  TORCH_CHECK(p_wm_csr_row_ptr_tensor_desc->dtype == WHOLEMEMORY_DT_INT64,
              "csr_row_ptr_wholememory_tensor_handle should be int64 WholeMemory Tensor.")
  TORCH_CHECK(p_wm_csr_col_ptr_tensor_desc->dtype == WHOLEMEMORY_DT_INT || WHOLEMEMORY_DT_INT64,
              "csr_col_ptr_wholememory_tensor_handle should be int or int64 WholeMemory Tensor.")
  TORCH_CHECK(
    p_wm_csr_weight_ptr_tensor_desc->dtype == WHOLEMEMORY_DT_FLOAT || WHOLEMEMORY_DT_DOUBLE,
    "csr_weight_ptr_wholememory_tensor_handle should be 1D WholeMemory Tensor.")

  wholememory_tensor_description_t output_sample_offset_tensor_desc;
  output_sample_offset_tensor_desc.dtype          = WHOLEMEMORY_DT_INT;
  output_sample_offset_tensor_desc.dim            = 1;
  output_sample_offset_tensor_desc.storage_offset = 0;
  output_sample_offset_tensor_desc.sizes[0]       = p_input_nodes_desc->sizes[0] + 1;
  output_sample_offset_tensor_desc.strides[0]     = 1;

  pytorch_memory_context output_sample_offset_context, output_dest_memory_context,
    output_center_localid_memory_context, output_edge_gid_memory_context;

  torch_common_malloc_func(&output_sample_offset_tensor_desc, &output_sample_offset_context);
  auto output_sample_offset_tensor = output_sample_offset_context.tensor;
  wrapped_torch_tensor const wrapped_output_sample_offset_tensor(output_sample_offset_tensor);

  unsigned long long random_seed_value;
  if (random_seed.has_value()) {
    random_seed_value = random_seed.value();
  } else {
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    thread_local std::uniform_int_distribution<unsigned long long> distrib;
    random_seed_value = distrib(gen);
  }

  TORCH_CHECK(wholegraph_csr_weighted_sample_without_replacement(
                wm_csr_row_ptr_tensor,
                wm_csr_col_ptr_tensor,
                wm_csr_weight_ptr_tensor,
                wrapped_input_nodes_tensor.get_wholememory_tensor(),
                max_sample_count,
                wrapped_output_sample_offset_tensor.get_wholememory_tensor(),
                &output_dest_memory_context,
                &output_center_localid_memory_context,
                &output_edge_gid_memory_context,
                random_seed_value,
                wholegraph_torch::get_pytorch_env_func(),
                wholegraph_torch::get_current_stream()) == WHOLEMEMORY_SUCCESS)

  auto output_dest_tensor           = output_dest_memory_context.tensor;
  auto output_center_localid_tensor = output_center_localid_memory_context.tensor;
  auto output_edge_gid_tensor       = output_edge_gid_memory_context.tensor;

  return {output_sample_offset_tensor,
          output_dest_tensor,
          output_center_localid_tensor,
          output_edge_gid_tensor};
}

}  // namespace wholegraph_torch

static auto registry =
  torch::RegisterOperators().op("wholegraph::weighted_sample_without_replacement",
                                &wholegraph_torch::weighted_sample_without_replacement);
