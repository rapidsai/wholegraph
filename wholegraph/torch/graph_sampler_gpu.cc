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
#include <assert.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <stdint.h>
#include <stdio.h>
#include <torch/library.h>
#include <torch/script.h>

#include "pytorch_cuda_env_fns.h"
#include "whole_chunked_pytorch_tensor.h"
#include "whole_memory_graph.h"

using torch::autograd::variable_list;

namespace whole_graph {

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
  torch::Tensor sample_offset_tensor = torch::empty({(long) (input_node_count + 1)}, to);
  torch::Tensor sample_output, center_localid;
  auto sample_output_allocator =
      GetAllocatorForTensor<void>(sample_output, d, input_nodes.dtype().toScalarType(), false);
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
  ChunkedTensor &csr_row_ptr = *((ChunkedTensor *) pcsr_row_ptr);
  ChunkedTensor &csr_col_ind = *((ChunkedTensor *) pcsr_col_ind);
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
  whole_graph::WholeChunkedMemory_t csr_row_ptr_wcmt = csr_row_ptr.GetChunkedMemory();
  whole_graph::WholeChunkedMemory_t csr_col_ind_wcmt = csr_col_ind.GetChunkedMemory();
  int64_t input_node_count = input_nodes.size(0);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto to = torch::TensorOptions().device(d).dtype(torch::kInt).requires_grad(false);
  torch::Tensor sample_offset_tensor = torch::empty({(long) (input_node_count + 1)}, to);
  torch::Tensor sample_output, center_localid;
  auto sample_output_allocator =
      GetAllocatorForTensor<void>(sample_output, d, input_nodes.dtype().toScalarType(), false);
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
  TORCH_CHECK(target.dtype() == torch::kInt64 || target.dtype() == torch::kInt32,
              "AppendUniqueGPU target should be int32 or int64 tensor.");
  TORCH_CHECK(neighbor.dtype() == torch::kInt64 || neighbor.dtype() == torch::kInt32,
              "AppendUniqueGPU neighbor should be int32 or int64 tensor.");
  TORCH_CHECK(target.dtype() == neighbor.dtype(), "AppendUniqueGPU target should be same type as neighbor");
  int target_count = target.sizes()[0];
  int neighbor_count = neighbor.sizes()[0];
  torch::Device d = neighbor.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto cuda_fns = GetCUDAEnvFns(d);
  torch::Tensor unique_total_output_tensor;
  torch::Tensor neighbor_raw_to_unique_mapping_tensor;
  torch::Tensor unique_output_neighbor_count_tensor;
  auto unique_total_output_allocator =
      GetAllocatorForTensor<void>(unique_total_output_tensor, d, target.dtype().toScalarType(), false);
  auto neighbor_raw_to_unique_mapping_allocator =
      GetAllocatorForTensor<int>(neighbor_raw_to_unique_mapping_tensor, d, torch::kInt, false);
  auto unique_output_neighbor_count_allocator =
      GetAllocatorForTensor<int>(unique_output_neighbor_count_tensor, d, torch::kInt, false);
  whole_graph::AppendUnique(target.data_ptr(),
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

torch::Tensor PyTorchCreateEdgeHashSet(const torch::Tensor &src_ids, const torch::Tensor &dst_ids) {
  TORCH_CHECK(src_ids.dim() == 1, "CreateEdgeHashSet, src_ids should be 1-D tensor");
  TORCH_CHECK(dst_ids.dim() == 1, "CreateEdgeHashSet, dst_ids should be 1-D tensor");
  TORCH_CHECK(src_ids.dtype() == torch::kInt32 || src_ids.dtype() == torch::kInt64,
              "CreateEdgeHashSet, src_ids should be int32 or int64");
  TORCH_CHECK(dst_ids.dtype() == torch::kInt32 || dst_ids.dtype() == torch::kInt64,
              "CreateEdgeHashSet, dst_ids should be int32 or int64");
  TORCH_CHECK(src_ids.dtype() == dst_ids.dtype(),
              "CreateEdgeHashSet, src_ids and dst_ids should be same type.");
  TORCH_CHECK(src_ids.size(0) == dst_ids.size(0), "CreateEdgeHashSet, src_ids and dst_ids should be same length.");
  int edge_count = src_ids.size(0);
  auto hash_memory_elt_count = whole_graph::GetEdgeHashSetEltCount(edge_count);
  torch::Device d = src_ids.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto to = torch::TensorOptions().device(d).dtype(src_ids.dtype()).requires_grad(false);
  torch::Tensor hash_set_mem_tensor = torch::empty({hash_memory_elt_count}, to);
  whole_graph::WMType id_type = C10ScalarToWMType(src_ids.dtype().toScalarType());
  whole_graph::CreateEdgeHashSet(id_type,
                                 src_ids.data_ptr(),
                                 dst_ids.data_ptr(),
                                 edge_count,
                                 hash_set_mem_tensor.data_ptr(),
                                 hash_memory_elt_count,
                                 stream);
  return hash_set_mem_tensor;
}

torch::Tensor PyTorchRetrieveCOOEdges(const torch::Tensor &src_ids,
                                      const torch::Tensor &dst_ids,
                                      const torch::Tensor &hash_set_mem_tensor) {
  TORCH_CHECK(src_ids.dim() == 1, "RetrieveCOOEdges, src_ids should be 1-D tensor");
  TORCH_CHECK(dst_ids.dim() == 1, "RetrieveCOOEdges, dst_ids should be 1-D tensor");
  TORCH_CHECK(hash_set_mem_tensor.dim() == 1, "RetrieveCOOEdges, hash_set_mem_tensor should be 1-D tensor");
  TORCH_CHECK(src_ids.dtype() == torch::kInt32 || src_ids.dtype() == torch::kInt64,
              "RetrieveCOOEdges, src_ids should be int32 or int64");
  TORCH_CHECK(dst_ids.dtype() == torch::kInt32 || dst_ids.dtype() == torch::kInt64,
              "RetrieveCOOEdges, dst_ids should be int32 or int64");
  TORCH_CHECK(hash_set_mem_tensor.dtype() == torch::kInt32 || hash_set_mem_tensor.dtype() == torch::kInt64,
              "RetrieveCOOEdges, hash_set_mem_tensor should be int32 or int64");
  TORCH_CHECK(src_ids.dtype() == dst_ids.dtype(),
              "RetrieveCOOEdges, src_ids and dst_ids should be same type.");
  TORCH_CHECK(src_ids.dtype() == hash_set_mem_tensor.dtype(),
              "RetrieveCOOEdges, src_ids and hash_set_mem_tensor should be same type.");
  TORCH_CHECK(src_ids.size(0) == dst_ids.size(0), "RetrieveCOOEdges, src_ids and dst_ids should be same length.");

  int edge_count = src_ids.size(0);

  torch::Device d = src_ids.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto to = torch::TensorOptions().device(d).dtype(torch::kInt32).requires_grad(false);
  auto hash_memory_elt_count = hash_set_mem_tensor.size(0);
  torch::Tensor output_tensor = torch::empty({edge_count}, to);
  whole_graph::WMType id_type = C10ScalarToWMType(src_ids.dtype().toScalarType());
  whole_graph::RetrieveCOOEdges(id_type,
                                src_ids.data_ptr(),
                                dst_ids.data_ptr(),
                                edge_count,
                                hash_set_mem_tensor.data_ptr(),
                                hash_memory_elt_count,
                                output_tensor.data_ptr<int>(),
                                stream);
  return output_tensor;
}

variable_list PyTorchFilterCSREdges(const torch::Tensor &src_ids,
                                    const torch::Tensor &gids_offset,
                                    const torch::Tensor &dst_ids_vdata,
                                    const torch::Tensor &hash_set_mem_tensor) {
  TORCH_CHECK(src_ids.dim() == 1, "PyTorchFilterCSREdges, src_ids should be 1-D tensor");
  TORCH_CHECK(gids_offset.dim() == 1, "PyTorchFilterCSREdges, gids_offset should be 1-D tensor");
  TORCH_CHECK(dst_ids_vdata.dim() == 1, "PyTorchFilterCSREdges, dst_ids_vdata should be 1-D tensor");
  TORCH_CHECK(hash_set_mem_tensor.dim() == 1, "PyTorchFilterCSREdges, hash_set_mem_tensor should be 1-D tensor");
  TORCH_CHECK(src_ids.dtype() == torch::kInt32 || src_ids.dtype() == torch::kInt64,
              "PyTorchFilterCSREdges, src_ids should be int32 or int64");
  TORCH_CHECK(gids_offset.dtype() == torch::kInt32,
              "PyTorchFilterCSREdges, gids_offset should be int32");
  TORCH_CHECK(dst_ids_vdata.dtype() == torch::kInt32 || dst_ids_vdata.dtype() == torch::kInt64,
              "PyTorchFilterCSREdges, dst_ids_vdata should be int32 or int64");
  TORCH_CHECK(hash_set_mem_tensor.dtype() == torch::kInt32 || hash_set_mem_tensor.dtype() == torch::kInt64,
              "PyTorchFilterCSREdges, hash_set_mem_tensor should be int32 or int64");
  TORCH_CHECK(src_ids.dtype() == dst_ids_vdata.dtype(),
              "PyTorchFilterCSREdges, src_ids and dst_ids_vdata should be same type.");
  TORCH_CHECK(src_ids.dtype() == hash_set_mem_tensor.dtype(),
              "PyTorchFilterCSREdges, src_ids and hash_set_mem_tensor should be same type.");
  TORCH_CHECK(gids_offset.size(0) == src_ids.size(0) + 1,
              "PyTorchFilterCSREdges, gids_offset.size should be src_ids.size + 1.");

  int node_count = src_ids.size(0);
  int edge_count = dst_ids_vdata.size(0);

  // neighboor_gids_offset, neighboor_gids_vdata, neighboor_src_lids
  torch::Device d = src_ids.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  //auto to = torch::TensorOptions().device(d).dtype(torch::kInt32).requires_grad(false);
  torch::Tensor new_gids_offset = torch::empty_like(gids_offset);
  torch::Tensor new_dst_ids, new_src_lids;

  auto cuda_fns = GetCUDAEnvFns(d);
  auto new_dst_ids_allocator = GetAllocatorForTensor<void>(new_dst_ids, d, dst_ids_vdata.dtype().toScalarType(), false);
  auto new_src_lids_allocator = GetAllocatorForTensor<int>(new_src_lids, d, torch::kInt32, false);

  auto hash_memory_elt_count = hash_set_mem_tensor.size(0);
  whole_graph::WMType id_type = C10ScalarToWMType(src_ids.dtype().toScalarType());
  whole_graph::FilterCSREdges(id_type,
                              src_ids.data_ptr(),
                              node_count,
                              gids_offset.data_ptr<int>(),
                              dst_ids_vdata.data_ptr(),
                              edge_count,
                              hash_set_mem_tensor.data_ptr(),
                              hash_memory_elt_count,
                              new_gids_offset.data_ptr<int>(),
                              new_dst_ids_allocator,
                              new_src_lids_allocator,
                              cuda_fns,
                              stream);
  return {new_gids_offset, new_dst_ids, new_src_lids};
}

torch::Tensor PerSourceUniformNegativeSample(torch::Tensor input_nodes,
                                             torch::Tensor csr_row_ptr,
                                             torch::Tensor csr_col_ind,
                                             int64_t graph_dst_node_count,
                                             int64_t negative_sample_count = 1) {
  TORCH_CHECK(input_nodes.dim() == 1, "PerSourceUniformNegativeSample input_nodes dim should be 1");
  TORCH_CHECK(input_nodes.dtype() == torch::kInt32 || input_nodes.dtype() == torch::kInt64,
              "PerSourceUniformNegativeSample input_nodes dtype should be kInt32(kInt) or kInt64(kLong)");
  TORCH_CHECK(csr_row_ptr.dim() == 1, "PerSourceUniformNegativeSample csr_row_ptr dim should be 1");
  TORCH_CHECK(csr_row_ptr.dtype() == torch::kInt64,
              "PerSourceUniformNegativeSample csr_row_ptr dtype should be kInt64(kLong)");
  TORCH_CHECK(csr_col_ind.dim() == 1, "PerSourceUniformNegativeSample csr_col_ind dim should be 1");
  TORCH_CHECK(csr_col_ind.dtype() == torch::kInt32 || csr_col_ind.dtype() == torch::kInt64,
              "PerSourceUniformNegativeSample input_nodes dtype should be kInt32(kInt) or kInt64(kLong)");
  TORCH_CHECK(input_nodes.dtype() == csr_col_ind.dtype(),
              "PerSourceUniformNegativeSample input_nodes and csr_col_ind should have same type");
  int64_t input_node_count = input_nodes.size(0);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  torch::Device d = input_nodes.device();
  torch::Tensor negative_sample_output;
  auto sample_output_allocator =
      GetAllocatorForTensor<void>(negative_sample_output, d, input_nodes.dtype().toScalarType(), false);
  WmmpPerNodeUniformNegativeSample(sample_output_allocator,
                                   csr_row_ptr.data_ptr(),
                                   csr_col_ind.data_ptr(),
                                   C10ScalarToWMType(input_nodes.dtype().toScalarType()),
                                   input_nodes.data_ptr(),
                                   input_node_count,
                                   graph_dst_node_count,
                                   negative_sample_count,
                                   GetCUDAEnvFns(d),
                                   stream);
  return negative_sample_output;
}

torch::Tensor PerSourceUniformNegativeSampleChunked(torch::Tensor input_nodes,
                                                    int64_t pcsr_row_ptr,
                                                    int64_t pcsr_col_ind,
                                                    int64_t graph_dst_node_count,
                                                    int64_t negative_sample_count = 1) {
  ChunkedTensor &csr_row_ptr = *((ChunkedTensor *) pcsr_row_ptr);
  ChunkedTensor &csr_col_ind = *((ChunkedTensor *) pcsr_col_ind);
  TORCH_CHECK(input_nodes.dim() == 1, "PerSourceUniformNegativeSample input_nodes dim should be 1");
  TORCH_CHECK(input_nodes.dtype() == torch::kInt32 || input_nodes.dtype() == torch::kInt64,
              "PerSourceUniformNegativeSample input_nodes dtype should be kInt32(kInt) or kInt64(kLong)");
  TORCH_CHECK(csr_row_ptr.dim() == 1, "PerSourceUniformNegativeSample csr_row_ptr dim should be 1");
  TORCH_CHECK(csr_row_ptr.dtype() == torch::kInt64,
              "PerSourceUniformNegativeSample csr_row_ptr dtype should be kInt64(kLong)");
  TORCH_CHECK(csr_col_ind.dim() == 1, "PerSourceUniformNegativeSample csr_col_ind dim should be 1");
  TORCH_CHECK(csr_col_ind.dtype() == torch::kInt32 || csr_col_ind.dtype() == torch::kInt64,
              "PerSourceUniformNegativeSample input_nodes dtype should be kInt32(kInt) or kInt64(kLong)");
  TORCH_CHECK(input_nodes.dtype() == csr_col_ind.dtype(),
              "PerSourceUniformNegativeSample input_nodes and csr_col_ind should have same type");
  int64_t input_node_count = input_nodes.size(0);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  torch::Device d = input_nodes.device();
  whole_graph::WholeChunkedMemory_t csr_row_ptr_wcmt = csr_row_ptr.GetChunkedMemory();
  whole_graph::WholeChunkedMemory_t csr_col_ind_wcmt = csr_col_ind.GetChunkedMemory();
  torch::Tensor negative_sample_output;
  auto sample_output_allocator =
      GetAllocatorForTensor<void>(negative_sample_output, d, input_nodes.dtype().toScalarType(), false);
  WmmpChunkedPerNodeUniformNegativeSample(sample_output_allocator,
                                          csr_row_ptr_wcmt,
                                          csr_col_ind_wcmt,
                                          C10ScalarToWMType(input_nodes.dtype().toScalarType()),
                                          input_nodes.data_ptr(),
                                          input_node_count,
                                          graph_dst_node_count,
                                          negative_sample_count,
                                          GetCUDAEnvFns(d),
                                          stream);
  return negative_sample_output;
}

torch::Tensor GetCSRMixedSubGraphEdgeTypes(const torch::Tensor &src_mixid,
                                           const torch::Tensor &sub_graph_csr_row_ptr,
                                           const torch::Tensor &sub_graph_csr_col_mixid,
                                           const torch::Tensor &edge_type_dict,
                                           const torch::Tensor &to_typed_id,
                                           int64_t node_type_count,
                                           int64_t edge_type_count) {
  TORCH_CHECK(edge_type_count <= node_type_count * node_type_count,
              "edge_type_count should be less than node_type_count * node_type_count");
  TORCH_CHECK(src_mixid.dim() == 1, "src_mixid should be 1D tensor");
  TORCH_CHECK(src_mixid.dtype() == torch::ScalarType::Int || src_mixid.dtype() == torch::ScalarType::Long,
              "src_mixid should be 1D tensor");
  int64_t src_node_count = src_mixid.size(0);
  TORCH_CHECK(sub_graph_csr_row_ptr.dtype() == torch::ScalarType::Int, "CSR row_ptr should be Int tensor.");
  TORCH_CHECK(sub_graph_csr_row_ptr.dim() == 1, "CSR row_ptr should be 1-D tensor.");
  TORCH_CHECK(sub_graph_csr_row_ptr.size(0) == src_node_count + 1, "CSR row_ptr should be (node_count + 1, ) tensor.");
  TORCH_CHECK(sub_graph_csr_col_mixid.dim() == 1, "sub_graph_csr_col_mixid should be 1-D tensor.");
  TORCH_CHECK(sub_graph_csr_col_mixid.dtype() == torch::ScalarType::Int
                  || sub_graph_csr_col_mixid.dtype() == torch::ScalarType::Long,
              "Sub graph CSR col_mixid should be Int or Long tensor.");
  int64_t dst_node_count = sub_graph_csr_col_mixid.size(0);
  TORCH_CHECK(edge_type_dict.dtype() == torch::ScalarType::Char, "edge_type_dict should be Char tensor.");
  TORCH_CHECK(edge_type_dict.dim() == 2, "edge_type_dict should be 2-D tensor.");
  TORCH_CHECK(edge_type_dict.size(0) == node_type_count && edge_type_dict.size(1) == node_type_count,
              "edge_type_dict should be (node_type_count, node_type_count) tensor.");
  TORCH_CHECK(to_typed_id.dtype() == torch::ScalarType::Long, "to_typed_id should be Long tensor");
  TORCH_CHECK(to_typed_id.dim() == 1, "to_typed_id should be 1-D tensor");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  torch::TensorOptions options;

  options = options.dtype(edge_type_dict.dtype())
                .device(src_mixid.device())
                .requires_grad(false);
  std::vector<int64_t> size{dst_node_count};
  torch::Tensor output = torch::empty(size, options);

  whole_graph::WmmpGetCSRMixedSubGraphEdgeTypes(
      whole_graph::pytorch::C10ScalarToWMType(src_mixid.dtype().toScalarType()),
      output.data_ptr<int8_t>(),
      src_mixid.data_ptr(),
      sub_graph_csr_row_ptr.data_ptr<int>(),
      sub_graph_csr_col_mixid.data_ptr(),
      edge_type_dict.data_ptr<int8_t>(),
      to_typed_id.data_ptr<int64_t>(),
      src_node_count,
      dst_node_count,
      node_type_count,
      edge_type_count,
      stream);

  return output;
}

torch::Tensor GetCSRMixedSubGraphEdgeTypesChunked(const torch::Tensor &src_mixid,
                                                  const torch::Tensor &sub_graph_csr_row_ptr,
                                                  const torch::Tensor &sub_graph_csr_col_mixid,
                                                  const torch::Tensor &edge_type_dict,
                                                  int64_t to_typed_id_ptr,
                                                  int64_t node_type_count,
                                                  int64_t edge_type_count) {
  auto &to_typed_id = *(whole_graph::pytorch::ChunkedTensor *) (to_typed_id_ptr);
  TORCH_CHECK(edge_type_count <= node_type_count * node_type_count,
              "edge_type_count should be less than node_type_count * node_type_count");
  TORCH_CHECK(src_mixid.dim() == 1, "src_mixid should be 1D tensor");
  TORCH_CHECK(src_mixid.dtype() == torch::ScalarType::Int || src_mixid.dtype() == torch::ScalarType::Long,
              "src_mixid should be 1D tensor");
  int64_t src_node_count = src_mixid.size(0);
  TORCH_CHECK(sub_graph_csr_row_ptr.dtype() == torch::ScalarType::Int, "CSR row_ptr should be Int tensor.");
  TORCH_CHECK(sub_graph_csr_row_ptr.dim() == 1, "CSR row_ptr should be 1-D tensor.");
  TORCH_CHECK(sub_graph_csr_row_ptr.size(0) == src_node_count + 1, "CSR row_ptr should be (node_count + 1, ) tensor.");
  TORCH_CHECK(sub_graph_csr_col_mixid.dim() == 1, "sub_graph_csr_col_mixid should be 1-D tensor.");
  TORCH_CHECK(sub_graph_csr_col_mixid.dtype() == torch::ScalarType::Int
                  || sub_graph_csr_col_mixid.dtype() == torch::ScalarType::Long,
              "Sub graph CSR col_mixid should be Int or Long tensor.");
  int64_t dst_node_count = sub_graph_csr_col_mixid.size(0);
  TORCH_CHECK(edge_type_dict.dtype() == torch::ScalarType::Char, "edge_type_dict should be Char tensor.");
  TORCH_CHECK(edge_type_dict.dim() == 2, "edge_type_dict should be 2-D tensor.");
  TORCH_CHECK(edge_type_dict.size(0) == node_type_count && edge_type_dict.size(1) == node_type_count,
              "edge_type_dict should be (node_type_count, node_type_count) tensor.");
  TORCH_CHECK(to_typed_id.dtype() == torch::ScalarType::Long, "to_typed_id should be Long tensor");
  TORCH_CHECK(to_typed_id.dim() == 1, "to_typed_id should be 1-D tensor");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  torch::TensorOptions options;

  options = options.dtype(edge_type_dict.dtype())
                .device(src_mixid.device())
                .requires_grad(false);
  std::vector<int64_t> size{dst_node_count};
  torch::Tensor output = torch::empty(size, options);

  whole_graph::WmmpGetCSRMixedSubGraphEdgeTypesChunked(
      whole_graph::pytorch::C10ScalarToWMType(src_mixid.dtype().toScalarType()),
      output.data_ptr<int8_t>(),
      src_mixid.data_ptr(),
      sub_graph_csr_row_ptr.data_ptr<int>(),
      sub_graph_csr_col_mixid.data_ptr(),
      edge_type_dict.data_ptr<int8_t>(),
      to_typed_id.GetChunkedMemory(),
      src_node_count,
      dst_node_count,
      node_type_count,
      edge_type_count,
      stream);

  return output;
}

torch::Tensor GetBucketedCSRFromSortedTypedIDs(const torch::Tensor &typed_ids,
                                               int64_t node_type_count) {
  int64_t id_count = typed_ids.size(0);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(typed_ids.dtype() == torch::ScalarType::Long, "typed_ids should be Long tensor.");
  TORCH_CHECK(typed_ids.dim() == 1, "typed_ids should be 1-D tensor.");
  TORCH_CHECK(node_type_count > 0, "node_type_count should be greater than 0.");

  torch::TensorOptions options;
  options = options.dtype(torch::ScalarType::Int)
                .device(typed_ids.device())
                .requires_grad(false);
  std::vector<int64_t> size{node_type_count + 1};
  torch::Tensor output = torch::empty(size, options);
  whole_graph::WmmpGetBucketedCSRFromSortedTypedIDs(output.data_ptr<int>(),
                                                    typed_ids.data_ptr<int64_t>(),
                                                    id_count,
                                                    node_type_count,
                                                    stream);
  return output;
}

torch::Tensor PackToTypedIDs(const torch::Tensor &ids,
                             const torch::Tensor &type_ids) {
  TORCH_CHECK(type_ids.dim() == 1, "type_ids should be 1-D tensor.");
  TORCH_CHECK(ids.dim() == 1, "ids should be 1-D tensor.");
  TORCH_CHECK(ids.size(0) == type_ids.size(0), "ids and should be same length.");
  TORCH_CHECK(ids.dtype() == torch::ScalarType::Int, "ids should be Int tensor.");
  TORCH_CHECK(type_ids.dtype() == torch::ScalarType::Char, "ids should be Char tensor.");
  int64_t id_count = ids.size(0);
  torch::TensorOptions options;
  options = options.dtype(torch::ScalarType::Long)
                .device(type_ids.device())
                .requires_grad(false);
  std::vector<int64_t> size{id_count};
  torch::Tensor output = torch::empty(size, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  whole_graph::WmmpPackToTypedIDs(output.data_ptr<int64_t>(),
                                  type_ids.data_ptr<int8_t>(),
                                  ids.data_ptr<int>(),
                                  id_count,
                                  stream);
  return output;
}

variable_list UnpackTypedIDs(const torch::Tensor &typed_ids) {
  TORCH_CHECK(typed_ids.dim() == 1, "type_ids should be 1-D tensor.");
  TORCH_CHECK(typed_ids.dtype() == torch::ScalarType::Long, "typed_ids should be Long tensor.");
  int64_t id_count = typed_ids.size(0);
  torch::TensorOptions type_options, id_options;
  type_options = type_options.dtype(torch::ScalarType::Char)
                     .device(typed_ids.device())
                     .requires_grad(false);
  id_options = id_options.dtype(torch::ScalarType::Int)
                   .device(typed_ids.device())
                   .requires_grad(false);
  std::vector<int64_t> size{id_count};
  torch::Tensor output_type = torch::empty(size, type_options);
  torch::Tensor output_id = torch::empty(size, id_options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  whole_graph::WmmpUnpackTypedIDs(typed_ids.data_ptr<int64_t>(),
                                  output_type.data_ptr<int8_t>(),
                                  output_id.data_ptr<int>(),
                                  id_count,
                                  stream);
  return {output_id, output_type};
}

variable_list WeightedSampleWithoutReplacementCUDA(torch::Tensor input_nodes,
                                                   torch::Tensor csr_row_ptr,
                                                   torch::Tensor csr_col_ind,
                                                   torch::Tensor csr_weight_ptr,
                                                   int64_t max_sample_count,
                                                   torch::optional<torch::Tensor> csr_local_sorted_map_indices_ptr) {
  TORCH_CHECK(input_nodes.dim() == 1, "WeightedSampleWithoutReplacementCUDA input_nodes dim should be 1");
  TORCH_CHECK(input_nodes.dtype() == torch::kInt32 || input_nodes.dtype() == torch::kInt64,
              "WeightedSampleWithoutReplacementCUDA input_nodes dtype should be kInt32(kInt) or kInt64(kLong)");
  TORCH_CHECK(csr_row_ptr.dim() == 1, "WeightedSampleWithoutReplacementCUDA csr_row_ptr dim should be 1");
  TORCH_CHECK(csr_row_ptr.dtype() == torch::kInt64,
              "WeightedSampleWithoutReplacementCUDA csr_row_ptr dtype should be kInt64(kLong)");
  TORCH_CHECK(csr_col_ind.dim() == 1, "WeightedSampleWithoutReplacementCUDA csr_col_ind dim should be 1");
  TORCH_CHECK(csr_col_ind.dtype() == torch::kInt32 || csr_col_ind.dtype() == torch::kInt64,
              "WeightedSampleWithoutReplacementCUDA csr_col_ind dtype should be kInt32(kInt) or kInt64(kLong)");
  TORCH_CHECK(input_nodes.dtype() == csr_col_ind.dtype(),
              "WeightedSampleWithoutReplacementCUDA input_nodes and csr_col_ind should have same type");
  TORCH_CHECK(csr_weight_ptr.dim() == 1, "WeightedSampleWithoutReplacementCUDA csr_weight_ptr dim should be 1");
  TORCH_CHECK(csr_weight_ptr.dtype() == torch::kFloat32 || csr_weight_ptr.dtype() == torch::kFloat64,
              "WeightedSampleWithoutReplacementCUDA csr_weight_ptr dtype should be kFloat32(kFloat) or kFloat64(kDouble)");
  TORCH_CHECK(csr_weight_ptr.size(0) == csr_col_ind.size(0),
              "WeightedSampleWithoutReplacementCUDA csr_weight_ptr size should be equal to csr_col_ind size");
  void *csr_local_sorted_map_indices_data_ptr = nullptr;
  if (csr_local_sorted_map_indices_ptr) {
    csr_local_sorted_map_indices_data_ptr = csr_local_sorted_map_indices_ptr->data_ptr();
    TORCH_CHECK(csr_local_sorted_map_indices_ptr->dim() == 1,
                "WeightedSampleWithoutReplacementCUDA csr_weight_ptr dim should be 1");
    TORCH_CHECK(csr_local_sorted_map_indices_ptr->dtype() == torch::kInt32,
                "WeightedSampleWithoutReplacementCUDA csr_local_sorted_map_indices_ptr dtype should be torch::KInt32");
  }
  int64_t input_node_count = input_nodes.size(0);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // input_nodes.device().is_cuda()
  torch::Device d = input_nodes.device();
  auto to = torch::TensorOptions().device(d).dtype(torch::kInt).requires_grad(false);
  torch::Tensor sample_offset_tensor = torch::empty({(long) (input_node_count + 1)}, to);
  torch::Tensor sample_output, center_localid;
  auto sample_output_allocator =
      GetAllocatorForTensor<void>(sample_output, d, input_nodes.dtype().toScalarType(), false);
  auto center_localid_allocator = GetAllocatorForTensor<void>(center_localid, d, torch::kInt32, false);
  WmmpWeightedSampleWithoutReplacement(sample_output_allocator,
                                       center_localid_allocator,
                                       sample_offset_tensor.data_ptr<int>(),
                                       csr_row_ptr.data_ptr(),
                                       csr_col_ind.data_ptr(),
                                       csr_weight_ptr.data_ptr(),
                                       csr_local_sorted_map_indices_data_ptr,
                                       C10ScalarToWMType(input_nodes.dtype().toScalarType()),
                                       C10ScalarToWMType(csr_weight_ptr.dtype().toScalarType()),
                                       input_nodes.data_ptr(),
                                       input_node_count,
                                       max_sample_count,
                                       GetCUDAEnvFns(d),
                                       stream);
  return {sample_offset_tensor, sample_output, center_localid};
}

variable_list WeightedSampleWithoutReplacementChunkedCUDA(torch::Tensor input_nodes,
                                                          int64_t pcsr_row_ptr,
                                                          int64_t pcsr_col_ind,
                                                          int64_t pcsr_weight_ptr,
                                                          int64_t max_sample_count,
                                                          torch::optional<int64_t> pcsr_local_sorted_map_indices_ptr) {
  ChunkedTensor &csr_row_ptr = *((ChunkedTensor *) pcsr_row_ptr);
  ChunkedTensor &csr_col_ind = *((ChunkedTensor *) pcsr_col_ind);
  ChunkedTensor &csr_weight_ptr = *((ChunkedTensor *) pcsr_weight_ptr);

  TORCH_CHECK(input_nodes.dim() == 1, "WeightedSampleWithoutReplacementCUDA input_nodes dim should be 1");
  TORCH_CHECK(input_nodes.dtype() == torch::kInt32 || input_nodes.dtype() == torch::kInt64,
              "WeightedSampleWithoutReplacementCUDA input_nodes dtype should be kInt32(kInt) or kInt64(kLong)");
  TORCH_CHECK(csr_row_ptr.dim() == 1, "WeightedSampleWithoutReplacementCUDA csr_row_ptr dim should be 1");
  TORCH_CHECK(csr_row_ptr.dtype() == torch::kInt64,
              "WeightedSampleWithoutReplacementCUDA csr_row_ptr dtype should be kInt64(kLong)");
  TORCH_CHECK(csr_col_ind.dim() == 1, "WeightedSampleWithoutReplacementCUDA csr_col_ind dim should be 1");
  TORCH_CHECK(csr_col_ind.dtype() == torch::kInt32 || csr_col_ind.dtype() == torch::kInt64,
              "WeightedSampleWithoutReplacementCUDA input_nodes dtype should be kInt32(kInt) or kInt64(kLong)");
  TORCH_CHECK(input_nodes.dtype() == csr_col_ind.dtype(),
              "WeightedSampleWithoutReplacementCUDA input_nodes and csr_col_ind should have same type");
  TORCH_CHECK(
      csr_row_ptr.storage_offset() == 0 && csr_col_ind.storage_offset() == 0 && csr_weight_ptr.storage_offset() == 0,
      "WeightedSampleWithoutReplacementCUDA tensor should have 0 storage_offset.");
  TORCH_CHECK(csr_weight_ptr.dim() == 1, "WeightedSampleWithoutReplacementCUDA csr_weight_ptr dim should be 1");
  TORCH_CHECK(csr_weight_ptr.dtype() == torch::kFloat32 || csr_weight_ptr.dtype() == torch::kFloat64,
              "WeightedSampleWithoutReplacementCUDA csr_weight_ptr dtype should be kFloat32(kFloat) or kFloat64(kDouble)");
  TORCH_CHECK(csr_weight_ptr.size(0) == csr_col_ind.size(0),
              "WeightedSampleWithoutReplacementCUDA csr_weight_ptr size should be equal to csr_col_ind size");
  whole_graph::WholeChunkedMemory_t csr_local_sorted_map_indices_ptr_wcmt = nullptr;
  if (pcsr_local_sorted_map_indices_ptr) {
    ChunkedTensor &csr_local_sorted_map_indices_ptr = *((ChunkedTensor *) (*pcsr_local_sorted_map_indices_ptr));
    TORCH_CHECK(csr_local_sorted_map_indices_ptr.dim() == 1,
                "WeightedSampleWithoutReplacementCUDA csr_local_sorted_map_indices_ptr dim should be 1 ");
    TORCH_CHECK(csr_local_sorted_map_indices_ptr.dtype() == torch::kInt32,
                "WeightedSampleWithoutReplacementCUDA the dtype of csr_local_sorted_map_indices_ptr should be torch::KInt32");
    TORCH_CHECK(csr_local_sorted_map_indices_ptr.size(0) == csr_col_ind.size(0),
                "WeightedSampleWithoutReplacementCUDA csr_local_sorted_map_indices_ptr size should be equal to csr_col_ind size");
    csr_local_sorted_map_indices_ptr_wcmt = csr_local_sorted_map_indices_ptr.GetChunkedMemory();
  }
  torch::Device d = input_nodes.device();
  whole_graph::WholeChunkedMemory_t csr_row_ptr_wcmt = csr_row_ptr.GetChunkedMemory();
  whole_graph::WholeChunkedMemory_t csr_col_ind_wcmt = csr_col_ind.GetChunkedMemory();
  whole_graph::WholeChunkedMemory_t csr_weight_wcmt = csr_weight_ptr.GetChunkedMemory();

  int64_t input_node_count = input_nodes.size(0);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto to = torch::TensorOptions().device(d).dtype(torch::kInt).requires_grad(false);
  torch::Tensor sample_offset_tensor = torch::empty({(long) (input_node_count + 1)}, to);
  torch::Tensor sample_output, center_localid;
  auto sample_output_allocator =
      GetAllocatorForTensor<void>(sample_output, d, input_nodes.dtype().toScalarType(), false);
  auto center_localid_allocator = GetAllocatorForTensor<void>(center_localid, d, torch::kInt32, false);
  WmmpChunkedWeightedSampleWithoutReplacement(sample_output_allocator,
                                              center_localid_allocator,
                                              sample_offset_tensor.data_ptr<int>(),
                                              csr_row_ptr_wcmt,
                                              csr_col_ind_wcmt,
                                              csr_weight_wcmt,
                                              csr_local_sorted_map_indices_ptr_wcmt,
                                              C10ScalarToWMType(input_nodes.dtype().toScalarType()),
                                              C10ScalarToWMType(csr_weight_ptr.dtype().toScalarType()),
                                              input_nodes.data_ptr(),
                                              input_node_count,
                                              max_sample_count,
                                              GetCUDAEnvFns(d),
                                              stream);
  return {sample_offset_tensor, sample_output, center_localid};
}

variable_list ExtractSubGraphWithFilter(const torch::Tensor &target_gid,
                                        const torch::Tensor &filter_target_value,
                                        const torch::Tensor &edges_csr_row,
                                        const torch::Tensor &edges_csr_col,
                                        const torch::Tensor &edges_value,
                                        int64_t extract_type_int,
                                        int64_t need_value) {
  TORCH_CHECK(target_gid.dim() == 1, "ExtractSubGraphWithFilter target_gid dim should be 1");
  TORCH_CHECK(target_gid.dtype() == torch::kInt32 || target_gid.dtype() == torch::kInt64,
              "ExtractSubGraphWithFilter target_gid dtype should be kInt32(kInt) or kInt64(kLong)");
  int64_t target_count = target_gid.size(0);
  TORCH_CHECK(filter_target_value.dim() == 1, "ExtractSubGraphWithFilter target_gid dim should be 1");
  TORCH_CHECK(filter_target_value.dtype() == torch::kInt8 || filter_target_value.dtype() == torch::kInt16
                  || filter_target_value.dtype() == torch::kInt32 || filter_target_value.dtype() == torch::kInt64
                  || filter_target_value.dtype() == torch::kFloat16 || filter_target_value.dtype() == torch::kFloat32
                  || filter_target_value.dtype() == torch::kFloat64,
              "ExtractSubGraphWithFilter filter_target_value dtype should be Int or float type.");
  TORCH_CHECK(filter_target_value.size(0) == target_gid.size(0),
              "ExtractSubGraphWithFilter filter_target_value should be same length as target_gid");
  TORCH_CHECK(edges_csr_row.dim() == 1, "ExtractSubGraphWithFilter edges_csr_row dim should be 1");
  TORCH_CHECK(edges_csr_row.dtype() == torch::kInt64,
              "ExtractSubGraphWithFilter edges_csr_row dtype should be kInt64(kLong)");
  int64_t total_node_count = edges_csr_row.size(0) - 1;
  TORCH_CHECK(edges_csr_col.dim() == 1, "ExtractSubGraphWithFilter edges_csr_col dim should be 1");
  TORCH_CHECK(edges_csr_col.dtype() == torch::kInt32 || edges_csr_col.dtype() == torch::kInt64,
              "ExtractSubGraphWithFilter edges_csr_col dtype should be kInt32(kInt) or kInt64(kLong)");
  TORCH_CHECK(target_gid.dtype() == edges_csr_col.dtype(),
              "ExtractSubGraphWithFilter target_gid and edges_csr_col should have same type");
  TORCH_CHECK(edges_value.dim() == 1, "ExtractSubGraphWithFilter edges_value dim should be 1");
  TORCH_CHECK(edges_value.size(0) == edges_csr_col.size(0),
              "ExtractSubGraphWithFilter edges_value should be same length as edges_csr_col.");
  TORCH_CHECK(edges_value.dtype() == filter_target_value.dtype(),
              "ExtractSubGraphWithFilter edges_value should be same type as filter_target_value");
  TORCH_CHECK(need_value == 0 || need_value == 1, "ExtractSubGraphWithFilter need_value should be 0 or 1");
  torch::Device d = target_gid.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto to = torch::TensorOptions().device(d).dtype(torch::kLong).requires_grad(false);
  torch::Tensor subgraph_row_tensor = torch::empty({(long) (target_count + 1)}, to);
  torch::Tensor subgraph_col_tensor, subgraph_edge_value_tensor;
  auto subgraph_col_allocator =
      GetAllocatorForTensor<void>(subgraph_col_tensor, d, edges_csr_col.dtype().toScalarType(), false);
  auto subgraph_edge_value_allocator =
      GetAllocatorForTensor<void>(subgraph_edge_value_tensor, d, edges_value.dtype().toScalarType(), false);
  WmmpExtractSubGraphWithFilter(C10ScalarToWMType(edges_csr_col.dtype().toScalarType()),
                                C10ScalarToWMType(edges_value.dtype().toScalarType()),
                                extract_type_int,
                                need_value == 1,
                                target_count,
                                total_node_count,
                                subgraph_row_tensor.data_ptr<int64_t>(),
                                subgraph_col_allocator,
                                subgraph_edge_value_allocator,
                                target_gid.data_ptr(),
                                filter_target_value.data_ptr(),
                                edges_csr_row.data_ptr<int64_t>(),
                                edges_csr_col.data_ptr(),
                                edges_value.data_ptr(),
                                GetCUDAEnvFns(d),
                                stream);
  if (need_value == 1) {
    return {subgraph_row_tensor, subgraph_col_tensor, subgraph_edge_value_tensor};
  } else {
    return {subgraph_row_tensor, subgraph_col_tensor};
  }
}

variable_list ExtractSubGraphWithFilterChunked(torch::Tensor target_gid,
                                               torch::Tensor filter_target_value,
                                               int64_t p_edges_csr_row,
                                               int64_t p_edges_csr_col,
                                               int64_t p_edges_value,
                                               int64_t extract_type_int,
                                               int64_t need_value) {
  ChunkedTensor &edges_csr_row = *((ChunkedTensor *) p_edges_csr_row);
  ChunkedTensor &edges_csr_col = *((ChunkedTensor *) p_edges_csr_col);
  ChunkedTensor &edges_value = *((ChunkedTensor *) p_edges_value);
  TORCH_CHECK(target_gid.dim() == 1, "ExtractSubGraphWithFilter target_gid dim should be 1");
  TORCH_CHECK(target_gid.dtype() == torch::kInt32 || target_gid.dtype() == torch::kInt64,
              "ExtractSubGraphWithFilter target_gid dtype should be kInt32(kInt) or kInt64(kLong)");
  int64_t target_count = target_gid.size(0);
  TORCH_CHECK(filter_target_value.dim() == 1, "ExtractSubGraphWithFilter target_gid dim should be 1");
  TORCH_CHECK(filter_target_value.dtype() == torch::kInt8 || filter_target_value.dtype() == torch::kInt16
                  || filter_target_value.dtype() == torch::kInt32 || filter_target_value.dtype() == torch::kInt64
                  || filter_target_value.dtype() == torch::kFloat16 || filter_target_value.dtype() == torch::kFloat32
                  || filter_target_value.dtype() == torch::kFloat64,
              "ExtractSubGraphWithFilter filter_target_value dtype should be Int or float type.");
  TORCH_CHECK(filter_target_value.size(0) == target_gid.size(0),
              "ExtractSubGraphWithFilter filter_target_value should be same length as target_gid");
  TORCH_CHECK(edges_csr_row.dim() == 1, "ExtractSubGraphWithFilter edges_csr_row dim should be 1");
  TORCH_CHECK(edges_csr_row.dtype() == torch::kInt64,
              "ExtractSubGraphWithFilter edges_csr_row dtype should be kInt64(kLong)");
  int64_t total_node_count = edges_csr_row.size(0) - 1;
  TORCH_CHECK(edges_csr_col.dim() == 1, "ExtractSubGraphWithFilter edges_csr_col dim should be 1");
  TORCH_CHECK(edges_csr_col.dtype() == torch::kInt32 || edges_csr_col.dtype() == torch::kInt64,
              "ExtractSubGraphWithFilter edges_csr_col dtype should be kInt32(kInt) or kInt64(kLong)");
  TORCH_CHECK(target_gid.dtype() == edges_csr_col.dtype(),
              "ExtractSubGraphWithFilter target_gid and edges_csr_col should have same type");
  TORCH_CHECK(edges_value.dim() == 1, "ExtractSubGraphWithFilter edges_value dim should be 1");
  TORCH_CHECK(edges_value.size(0) == edges_csr_col.size(0),
              "ExtractSubGraphWithFilter edges_value should be same length as edges_csr_col.");
  TORCH_CHECK(edges_value.dtype() == filter_target_value.dtype(),
              "ExtractSubGraphWithFilter edges_value should be same type as filter_target_value");
  TORCH_CHECK(need_value == 0 || need_value == 1, "ExtractSubGraphWithFilter need_value should be 0 or 1");
  torch::Device d = target_gid.device();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto to = torch::TensorOptions().device(d).dtype(torch::kLong).requires_grad(false);
  torch::Tensor subgraph_row_tensor = torch::empty({(long) (target_count + 1)}, to);
  torch::Tensor subgraph_col_tensor, subgraph_edge_value_tensor;
  auto subgraph_col_allocator =
      GetAllocatorForTensor<void>(subgraph_col_tensor, d, edges_csr_col.dtype().toScalarType(), false);
  auto subgraph_edge_value_allocator =
      GetAllocatorForTensor<void>(subgraph_edge_value_tensor, d, edges_value.dtype().toScalarType(), false);
  WmmpExtractSubGraphWithFilterChunked(C10ScalarToWMType(edges_csr_col.dtype().toScalarType()),
                                       C10ScalarToWMType(edges_value.dtype().toScalarType()),
                                       extract_type_int,
                                       need_value == 1,
                                       target_count,
                                       total_node_count,
                                       subgraph_row_tensor.data_ptr<int64_t>(),
                                       subgraph_col_allocator,
                                       subgraph_edge_value_allocator,
                                       target_gid.data_ptr(),
                                       filter_target_value.data_ptr(),
                                       edges_csr_row.GetChunkedMemory(),
                                       edges_csr_col.GetChunkedMemory(),
                                       edges_value.GetChunkedMemory(),
                                       GetCUDAEnvFns(d),
                                       stream);
  if (need_value == 1) {
    return {subgraph_row_tensor, subgraph_col_tensor, subgraph_edge_value_tensor};
  } else {
    return {subgraph_row_tensor, subgraph_col_tensor};
  }
}

}// namespace pytorch

}// namespace whole_graph

static auto registry = torch::RegisterOperators()
                           .op("wholegraph::unweighted_sample_without_replacement",
                               &whole_graph::pytorch::UnweightedSampleWithoutReplacementCUDA)
                           .op("wholegraph::unweighted_sample_without_replacement_chunked",
                               &whole_graph::pytorch::UnweightedSampleWithoutReplacementChunkedCUDA)
                           .op("wholegraph::append_unique", &whole_graph::pytorch::AppendUniqueGPU)
                           .op("wholegraph::create_edge_hashset", &whole_graph::pytorch::PyTorchCreateEdgeHashSet)
                           .op("wholegraph::retrieve_coo_edges", &whole_graph::pytorch::PyTorchRetrieveCOOEdges)
                           .op("wholegraph::filter_csr_edges", &whole_graph::pytorch::PyTorchFilterCSREdges)
                           .op("wholegraph::per_source_uniform_negative_sample", &whole_graph::pytorch::PerSourceUniformNegativeSample)
                           .op("wholegraph::per_source_uniform_negative_sample_chunked",
                               &whole_graph::pytorch::PerSourceUniformNegativeSampleChunked)
                           .op("wholegraph::get_csr_mixed_sub_graph_edge_types", &whole_graph::pytorch::GetCSRMixedSubGraphEdgeTypes)
                           .op("wholegraph::get_csr_mixed_sub_graph_edge_types_chunked",
                               &whole_graph::pytorch::GetCSRMixedSubGraphEdgeTypesChunked)
                           .op("wholegraph::get_bucketed_csr_from_sorted_typed_ids",
                               &whole_graph::pytorch::GetBucketedCSRFromSortedTypedIDs)
                           .op("wholegraph::pack_to_typed_ids",
                               &whole_graph::pytorch::PackToTypedIDs)
                           .op("wholegraph::unpack_typed_ids",
                               &whole_graph::pytorch::UnpackTypedIDs)
                           .op("wholegraph::weighted_sample_without_replacement",
                               &whole_graph::pytorch::WeightedSampleWithoutReplacementCUDA)
                           .op("wholegraph::weighted_sample_without_replacement_chunked",
                               &whole_graph::pytorch::WeightedSampleWithoutReplacementChunkedCUDA)
                           .op("wholegraph::extract_subgraph_with_filter",
                               &whole_graph::pytorch::ExtractSubGraphWithFilter)
                           .op("wholegraph::extract_subgraph_with_filter_chunked",
                               &whole_graph::pytorch::ExtractSubGraphWithFilterChunked);
