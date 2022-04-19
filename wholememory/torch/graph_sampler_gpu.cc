#include <pybind11/pybind11.h>
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime_api.h>
#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>

#include "whole_chunked_pytorch_tensor.h"
#include "whole_memory_graph.h"
#include "pytorch_cuda_env_fns.h"

using torch::autograd::variable_list;

namespace whole_memory {

namespace pytorch {

variable_list UnweightedSampleWithoutReplacementCUDA(torch::Tensor input_nodes,
                                                     torch::Tensor csr_row_ptr,
                                                     torch::Tensor csr_col_ind,
                                                     int64_t max_sample_count) {
  TORCH_CHECK(input_nodes.dim() == 1, "UnweightedSampleWithoutReplacementCUDA input_nodes dim should be 1");
  TORCH_CHECK(input_nodes.dtype() == torch::kInt32 || input_nodes.dtype() == torch::kInt64,
              "UnweightedSampleWithoutReplacementCUDA input_nodes dtype should be kInt32(kInt) or kInt64(kLong)");
  TORCH_CHECK(csr_row_ptr.dim() == 1, "UnweightedSampleWithoutReplacementCUDA csr_row_ptr dim should be 1");
  TORCH_CHECK(csr_row_ptr.dtype() == torch::kInt64,
              "UnweightedSampleWithoutReplacementCUDA csr_row_ptr dtype should be kInt64(kLong)");
  TORCH_CHECK(csr_col_ind.dim() == 1, "UnweightedSampleWithoutReplacementCUDA csr_col_ind dim should be 1");
  TORCH_CHECK(csr_col_ind.dtype() == torch::kInt32 || csr_col_ind.dtype() == torch::kInt64,
              "UnweightedSampleWithoutReplacementCUDA input_nodes dtype should be kInt32(kInt) or kInt64(kLong)");
  TORCH_CHECK(input_nodes.dtype() == csr_col_ind.dtype(),
              "UnweightedSampleWithoutReplacementCUDA input_nodes and csr_col_ind should have same type");
  int64_t input_node_count = input_nodes.size(0);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  torch::Device d = input_nodes.device();
  auto to = torch::TensorOptions().device(d).dtype(torch::kInt).requires_grad(false);
  torch::Tensor sample_offset_tensor = torch::empty({(long)(input_node_count + 1)}, to);
  torch::Tensor sample_output, center_localid;
  auto sample_output_allocator = GetAllocatorForTensor<void>(sample_output, d, input_nodes.dtype().toScalarType(), false);
  auto center_localid_allocator = GetAllocatorForTensor<void>(center_localid, d, torch::kInt32, false);
  WmmpUnweightedSampleWithoutReplacement(sample_output_allocator,
                                         center_localid_allocator,
                                         sample_offset_tensor.data_ptr<int>(),
                                         csr_row_ptr.data_ptr(),
                                         csr_col_ind.data_ptr(),
                                         C10ScalarToWMType(input_nodes.dtype().toScalarType()),
                                         input_nodes.data_ptr(),
                                         input_node_count,
                                         max_sample_count,
                                         GetCUDAEnvFns(d),
                                         stream);
  return {sample_offset_tensor, sample_output, center_localid};
}

variable_list UnweightedSampleWithoutReplacementChunkedCUDA(torch::Tensor input_nodes,
                                                            int64_t pcsr_row_ptr,
                                                            int64_t pcsr_col_ind,
                                                            int64_t max_sample_count) {
  ChunkedTensor& csr_row_ptr = *((ChunkedTensor*)pcsr_row_ptr);
  ChunkedTensor& csr_col_ind = *((ChunkedTensor*)pcsr_col_ind);
  TORCH_CHECK(input_nodes.dim() == 1, "UnweightedSampleWithoutReplacementCUDA input_nodes dim should be 1");
  TORCH_CHECK(input_nodes.dtype() == torch::kInt32 || input_nodes.dtype() == torch::kInt64,
              "UnweightedSampleWithoutReplacementCUDA input_nodes dtype should be kInt32(kInt) or kInt64(kLong)");
  TORCH_CHECK(csr_row_ptr.dim() == 1, "UnweightedSampleWithoutReplacementCUDA csr_row_ptr dim should be 1");
  TORCH_CHECK(csr_row_ptr.dtype() == torch::kInt64,
              "UnweightedSampleWithoutReplacementCUDA csr_row_ptr dtype should be kInt64(kLong)");
  TORCH_CHECK(csr_col_ind.dim() == 1, "UnweightedSampleWithoutReplacementCUDA csr_col_ind dim should be 1");
  TORCH_CHECK(csr_col_ind.dtype() == torch::kInt32 || csr_col_ind.dtype() == torch::kInt64,
              "UnweightedSampleWithoutReplacementCUDA input_nodes dtype should be kInt32(kInt) or kInt64(kLong)");
  TORCH_CHECK(input_nodes.dtype() == csr_col_ind.dtype(),
              "UnweightedSampleWithoutReplacementCUDA input_nodes and csr_col_ind should have same type");
  TORCH_CHECK(csr_row_ptr.storage_offset() == 0 && csr_col_ind.storage_offset() == 0,
              "UnweightedSampleWithoutReplacementCUDA tensor should have 0 storage_offset.");
  torch::Device d = input_nodes.device();
  whole_memory::WholeChunkedMemory_t csr_row_ptr_wcmt = csr_row_ptr.GetChunkedMemory();
  whole_memory::WholeChunkedMemory_t csr_col_ind_wcmt = csr_col_ind.GetChunkedMemory();
  int64_t input_node_count = input_nodes.size(0);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto to = torch::TensorOptions().device(d).dtype(torch::kInt).requires_grad(false);
  torch::Tensor sample_offset_tensor = torch::empty({(long)(input_node_count + 1)}, to);
  torch::Tensor sample_output, center_localid;
  auto sample_output_allocator = GetAllocatorForTensor<void>(sample_output, d, input_nodes.dtype().toScalarType(), false);
  auto center_localid_allocator = GetAllocatorForTensor<void>(center_localid, d, torch::kInt32, false);
  WmmpChunkedUnweightedSampleWithoutReplacement(sample_output_allocator,
                                                center_localid_allocator,
                                                sample_offset_tensor.data_ptr<int>(),
                                                csr_row_ptr_wcmt,
                                                csr_col_ind_wcmt,
                                                C10ScalarToWMType(input_nodes.dtype().toScalarType()),
                                                input_nodes.data_ptr(),
                                                input_node_count,
                                                max_sample_count,
                                                GetCUDAEnvFns(d),
                                                stream);
  return {sample_offset_tensor, sample_output, center_localid};
}

// Input:
//      target and neighbor
// Output:
//      unique_total_output
//      neighbor_raw_to_unique_mapping
//      unique_output_neighbor_count
variable_list AppendUniqueGPU(torch::Tensor target, torch::Tensor neighbor) {
  TORCH_CHECK(target.dim() == 1, "AppendUniqueGPU target should be 1D tensor.");
  TORCH_CHECK(neighbor.dim() == 1, "AppendUniqueGPU neighbor should be 1D tensor.");
  TORCH_CHECK(target.dtype() == torch::kInt64 || target.dtype() == torch::kInt32, "AppendUniqueGPU target should be int32 or int64 tensor.");
  TORCH_CHECK(neighbor.dtype() == torch::kInt64 || neighbor.dtype() == torch::kInt32, "AppendUniqueGPU neighbor should be int32 or int64 tensor.");
  TORCH_CHECK(target.dtype() == neighbor.dtype(), "AppendUniqueGPU target should be same type as neighbor");
  int target_count = target.sizes()[0];
  int neighbor_count = neighbor.sizes()[0];
  torch::Device d = neighbor.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto cuda_fns = GetCUDAEnvFns(d);
  torch::Tensor unique_total_output_tensor;
  torch::Tensor neighbor_raw_to_unique_mapping_tensor;
  torch::Tensor unique_output_neighbor_count_tensor;
  auto unique_total_output_allocator
      = GetAllocatorForTensor<void>(unique_total_output_tensor, d, target.dtype().toScalarType(), false);
  auto neighbor_raw_to_unique_mapping_allocator
      = GetAllocatorForTensor<int>(neighbor_raw_to_unique_mapping_tensor, d, torch::kInt, false);
  auto unique_output_neighbor_count_allocator
      = GetAllocatorForTensor<int>(unique_output_neighbor_count_tensor, d, torch::kInt, false);
  whole_memory::AppendUnique(target.data_ptr(),
                             target_count,
                             neighbor.data_ptr(),
                             neighbor_count,
                             C10ScalarToWMType(target.dtype().toScalarType()),
                             unique_total_output_allocator,
                             neighbor_raw_to_unique_mapping_allocator,
                             unique_output_neighbor_count_allocator,
                             cuda_fns,
                             stream);
  return {unique_total_output_tensor, neighbor_raw_to_unique_mapping_tensor, unique_output_neighbor_count_tensor};
}

}

}

static auto registry = torch::RegisterOperators()
    .op("wholememory::unweighted_sample_without_replacement", &whole_memory::pytorch::UnweightedSampleWithoutReplacementCUDA)
    .op("wholememory::unweighted_sample_without_replacement_chunked", &whole_memory::pytorch::UnweightedSampleWithoutReplacementChunkedCUDA)
    .op("wholememory::append_unique", &whole_memory::pytorch::AppendUniqueGPU);
