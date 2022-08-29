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
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <stdint.h>
#include <stdio.h>
#include <torch/script.h>

#include "pytorch_cuda_env_fns.h"
#include "whole_chunked_pytorch_tensor.h"
#include "whole_memory_embedding.h"
#include "whole_nccl_pytorch_tensor.h"

namespace whole_graph {

namespace pytorch {

template<bool RequireGrad = false>
torch::Tensor WholeMemoryGatherCUDA(const torch::Tensor &indice,
                                    const torch::Tensor &parameter,
                                    torch::ScalarType output_scalar_type) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  caffe2::TypeMeta output_meta_type;
  output_meta_type = output_scalar_type;
  TORCH_CHECK(indice.dim() == 1, "WholeMemoryGatherCUDA indice dim should be 1");
  TORCH_CHECK(indice.dtype() == torch::kInt32 || indice.dtype() == torch::kInt64,
              "WholeMemoryGatherCUDA indice dtype should be kInt32(kInt) or kInt64(kLong)");
  TORCH_CHECK(parameter.dim() == 2 || parameter.dim() == 1, "WholeMemoryGatherCUDA parameter dim should be 1 or 2")
  TORCH_CHECK(parameter.dtype() == torch::kInt8 || parameter.dtype() == torch::kInt16 || parameter.dtype() == torch::kInt32 || parameter.dtype() == torch::kInt64 || parameter.dtype() == torch::kFloat16 || parameter.dtype() == torch::kBFloat16 || parameter.dtype() == torch::kFloat32 || parameter.dtype() == torch::kFloat64,
              "WholeMemoryGatherCUDA parameter dtype should be half float double or bfloat16");
  TORCH_CHECK(output_meta_type == torch::kInt8 || output_meta_type == torch::kInt16 || output_meta_type == torch::kInt32 || output_meta_type == torch::kInt64 || output_meta_type == torch::kFloat16 || output_meta_type == torch::kBFloat16 || output_meta_type == torch::kFloat32 || output_meta_type == torch::kFloat64,
              "WholeMemoryGatherCUDA output type should be half float double or bfloat16");
  TORCH_CHECK(indice.device() == parameter.device(), "indice and parameter should have same device.");
  int64_t indice_count = indice.size(0);
  int64_t embedding_dim = parameter.dim() == 2 ? parameter.size(1) : 1;
  int64_t embedding_stride = parameter.stride(0);
  torch::Device d = indice.device();
  c10::ScalarType output_type = output_scalar_type;
  c10::ScalarType param_type = parameter.dtype().toScalarType();
  c10::ScalarType index_type = indice.dtype().toScalarType();
  auto to = torch::TensorOptions().device(d).dtype(output_meta_type).requires_grad(RequireGrad);
  std::vector<int64_t> output_shape({indice_count, embedding_dim});
  if (parameter.dim() == 1) output_shape.pop_back();
  torch::Tensor output_tensor = torch::empty(output_shape, to);
  whole_graph::WholeMemoryGather(whole_graph::pytorch::C10ScalarToWMType(output_type),
                                 whole_graph::pytorch::C10ScalarToWMType(param_type),
                                 whole_graph::pytorch::C10ScalarToWMType(index_type),
                                 output_tensor.data_ptr(),
                                 parameter.data_ptr(),
                                 indice.data_ptr(),
                                 0,
                                 indice_count,
                                 embedding_dim,
                                 embedding_stride,
                                 embedding_dim,
                                 stream);
  return output_tensor;
}

torch::Tensor WholeMemoryGatherCUDANoGrad(const torch::Tensor &indice,
                                          const torch::Tensor &parameter,
                                          torch::ScalarType output_scalar_type) {
  return WholeMemoryGatherCUDA<false>(indice, parameter, output_scalar_type);
}

torch::Tensor WholeMemoryGatherCUDAGrad(const torch::Tensor &indice,
                                        const torch::Tensor &parameter,
                                        torch::ScalarType output_scalar_type) {
  return WholeMemoryGatherCUDA<true>(indice, parameter, output_scalar_type);
}

template<bool RequireGrad = false>
torch::Tensor WholeMemoryGatherChunkedCUDA(const torch::Tensor &indice,
                                           int64_t pparameter,
                                           torch::ScalarType output_scalar_type) {
  ChunkedTensor &parameter = *((ChunkedTensor *) pparameter);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  caffe2::TypeMeta output_meta_type;
  output_meta_type = output_scalar_type;
  TORCH_CHECK(indice.dim() == 1, "WholeMemoryGatherCUDA indice dim should be 1");
  TORCH_CHECK(indice.dtype() == torch::kInt32 || indice.dtype() == torch::kInt64,
              "WholeMemoryGatherCUDA indice dtype should be kInt32(kInt) or kInt64(kLong)");
  TORCH_CHECK(parameter.dim() == 2 || parameter.dim() == 1, "WholeMemoryGatherCUDA parameter dim should be 1 or 2")
  TORCH_CHECK(parameter.dtype() == torch::kInt8 || parameter.dtype() == torch::kInt16 || parameter.dtype() == torch::kInt32 || parameter.dtype() == torch::kInt64 || parameter.dtype() == torch::kFloat16 || parameter.dtype() == torch::kBFloat16 || parameter.dtype() == torch::kFloat32 || parameter.dtype() == torch::kFloat64,
              "WholeMemoryGatherCUDA parameter dtype should be half float double or bfloat16");
  TORCH_CHECK(output_meta_type == torch::kInt8 || output_meta_type == torch::kInt16 || output_meta_type == torch::kInt32 || output_meta_type == torch::kInt64 || output_meta_type == torch::kFloat16 || output_meta_type == torch::kBFloat16 || output_meta_type == torch::kFloat32 || output_meta_type == torch::kFloat64,
              "WholeMemoryGatherCUDA output type should be half float double or bfloat16");
  int64_t indice_count = indice.size(0);
  int64_t embedding_dim = parameter.dim() == 2 ? parameter.size(1) : 1;
  int64_t embedding_stride = parameter.stride(0);
  torch::Device d = indice.device();
  c10::ScalarType output_type = output_scalar_type;
  c10::ScalarType param_type = parameter.dtype().toScalarType();
  c10::ScalarType index_type = indice.dtype().toScalarType();
  auto to = torch::TensorOptions().device(d).dtype(output_meta_type).requires_grad(RequireGrad);
  std::vector<int64_t> output_shape({indice_count, embedding_dim});
  if (parameter.dim() == 1) output_shape.pop_back();
  torch::Tensor output_tensor = torch::empty(output_shape, to);
  whole_graph::WholeChunkedMemory_t wcmt = parameter.GetChunkedMemory();
  whole_graph::WholeMemoryChunkedGather(whole_graph::pytorch::C10ScalarToWMType(output_type),
                                        whole_graph::pytorch::C10ScalarToWMType(param_type),
                                        whole_graph::pytorch::C10ScalarToWMType(index_type),
                                        output_tensor.data_ptr(),
                                        wcmt,
                                        indice.data_ptr(),
                                        parameter.storage_offset(),
                                        indice_count,
                                        embedding_dim,
                                        embedding_stride,
                                        embedding_dim,
                                        stream);
  return output_tensor;
}

torch::Tensor WholeMemoryGatherChunkedCUDANoGrad(const torch::Tensor &indice,
                                                 int64_t pparameter,
                                                 torch::ScalarType output_scalar_type) {
  return WholeMemoryGatherChunkedCUDA<false>(indice, pparameter, output_scalar_type);
}

torch::Tensor WholeMemoryGatherChunkedCUDAGrad(const torch::Tensor &indice,
                                               int64_t pparameter,
                                               torch::ScalarType output_scalar_type) {
  return WholeMemoryGatherChunkedCUDA<true>(indice, pparameter, output_scalar_type);
}

template<bool RequireGrad = false>
torch::Tensor WholeMemoryGatherNCCLCUDA(const torch::Tensor &indice,
                                        int64_t pparameter) {
  NCCLTensor &parameter = *((NCCLTensor *) pparameter);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(indice.dim() == 1, "WholeMemoryGatherNCCLCUDA indice dim should be 1");
  TORCH_CHECK(indice.dtype() == torch::kInt32 || indice.dtype() == torch::kInt64,
              "WholeMemoryGatherNCCLCUDA indice dtype should be kInt32(kInt) or kInt64(kLong)");
  TORCH_CHECK(parameter.dim() == 2 || parameter.dim() == 1, "WholeMemoryGatherCUDA parameter dim should be 1 or 2")
  TORCH_CHECK(parameter.dtype() == torch::kInt8 || parameter.dtype() == torch::kInt16 || parameter.dtype() == torch::kInt32 || parameter.dtype() == torch::kInt64 || parameter.dtype() == torch::kFloat16 || parameter.dtype() == torch::kBFloat16 || parameter.dtype() == torch::kFloat32 || parameter.dtype() == torch::kFloat64,
              "WholeMemoryGatherNCCLCUDA parameter dtype should be half float double or bfloat16");
  int64_t indice_count = indice.size(0);
  int64_t embedding_dim = parameter.dim() == 2 ? parameter.size(1) : 1;
  int64_t embedding_stride = parameter.stride(0);
  torch::Device d = indice.device();
  c10::ScalarType param_type = parameter.dtype().toScalarType();
  c10::ScalarType index_type = indice.dtype().toScalarType();
  auto to = torch::TensorOptions().device(d).dtype(parameter.dtype()).requires_grad(RequireGrad);
  std::vector<int64_t> output_shape({indice_count, embedding_dim});
  if (parameter.dim() == 1) output_shape.pop_back();
  torch::Tensor output_tensor = torch::empty(output_shape, to);
  whole_graph::WholeNCCLMemory_t wnmt = parameter.GetNCCLMemory();
  auto cuda_fns = GetCUDAEnvFns(d);
  whole_graph::WholeMemoryNCCLGather(whole_graph::pytorch::C10ScalarToWMType(param_type),
                                     whole_graph::pytorch::C10ScalarToWMType(index_type),
                                     output_tensor.data_ptr(),
                                     wnmt,
                                     indice.data_ptr(),
                                     parameter.storage_offset(),
                                     indice_count,
                                     embedding_dim,
                                     embedding_stride,
                                     embedding_dim,
                                     cuda_fns,
                                     stream);
  return output_tensor;
}

torch::Tensor WholeMemoryGatherNCCLCUDANoGrad(const torch::Tensor &indice,
                                              int64_t pparameter) {
  return WholeMemoryGatherNCCLCUDA<false>(indice, pparameter);
}

torch::Tensor WholeMemoryGatherNCCLCUDAGrad(const torch::Tensor &indice,
                                            int64_t pparameter) {
  return WholeMemoryGatherNCCLCUDA<true>(indice, pparameter);
}

void WholeMemoryScatterCUDA(const torch::Tensor &input,
                            const torch::Tensor &indice,
                            const torch::Tensor &parameter) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(indice.dim() == 1, "WholeMemoryScatterCUDA indice dim should be 1");
  TORCH_CHECK(indice.dtype() == torch::kInt32 || indice.dtype() == torch::kInt64,
              "WholeMemoryScatterCUDA indice dtype should be kInt32(kInt) or kInt64(kLong)");
  TORCH_CHECK(parameter.dim() == 1 || parameter.dim() == 2, "WholeMemoryScatterCUDA parameter dim should be 1 or 2")
  TORCH_CHECK(
      parameter.dtype() == torch::kInt8 || parameter.dtype() == torch::kInt16 || parameter.dtype() == torch::kInt32
          || parameter.dtype() == torch::kInt64 || parameter.dtype() == torch::kFloat16
          || parameter.dtype() == torch::kBFloat16 || parameter.dtype() == torch::kFloat32
          || parameter.dtype() == torch::kFloat64,
      "WholeMemoryScatterCUDA parameter dtype not supported");
  TORCH_CHECK(input.dtype() == parameter.dtype(),
              "WholeMemoryScatterCUDA input type should be same as parameter");
  TORCH_CHECK(indice.device() == parameter.device(), "indice and parameter should have same device.");
  TORCH_CHECK(indice.device() == input.device(), "indice and input should have same device.");
  int64_t indice_count = indice.size(0);
  int64_t embedding_dim = parameter.dim() == 2 ? parameter.size(1) : 1;
  int64_t embedding_stride = parameter.stride(0);
  TORCH_CHECK(parameter.dim() == input.dim(), "embedding and input should be same dim");
  TORCH_CHECK(input.dim() == 1 || input.size(1) == embedding_dim, "input embedding dim should same as parameter");
  int64_t input_stride = input.stride(0);
  c10::ScalarType param_type = parameter.dtype().toScalarType();
  c10::ScalarType index_type = indice.dtype().toScalarType();
  whole_graph::WholeMemoryScatter(whole_graph::pytorch::C10ScalarToWMType(param_type),
                                  whole_graph::pytorch::C10ScalarToWMType(index_type),
                                  input.data_ptr(),
                                  parameter.data_ptr(),
                                  indice.data_ptr(),
                                  0,
                                  indice_count,
                                  embedding_dim,
                                  embedding_stride,
                                  input_stride,
                                  stream);
}

void WholeMemoryScatterChunkedCUDA(const torch::Tensor &input,
                                   const torch::Tensor &indice,
                                   int64_t pparameter) {
  ChunkedTensor &parameter = *((ChunkedTensor *) pparameter);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(indice.dim() == 1, "WholeMemoryScatterChunkedCUDA indice dim should be 1");
  TORCH_CHECK(indice.dtype() == torch::kInt32 || indice.dtype() == torch::kInt64,
              "WholeMemoryScatterChunkedCUDA indice dtype should be kInt32(kInt) or kInt64(kLong)");
  TORCH_CHECK(parameter.dim() == 1 || parameter.dim() == 2, "WholeMemoryScatterChunkedCUDA parameter dim should be 1 or 2")
  TORCH_CHECK(
      parameter.dtype() == torch::kInt8 || parameter.dtype() == torch::kInt16 || parameter.dtype() == torch::kInt32
          || parameter.dtype() == torch::kInt64 || parameter.dtype() == torch::kFloat16
          || parameter.dtype() == torch::kBFloat16 || parameter.dtype() == torch::kFloat32
          || parameter.dtype() == torch::kFloat64,
      "WholeMemoryScatterChunkedCUDA parameter dtype not supported");
  TORCH_CHECK(input.dtype() == parameter.dtype(),
              "WholeMemoryScatterChunkedCUDA input type should be same as parameter");
  TORCH_CHECK(indice.device() == input.device(), "indice and input should have same device.");
  int64_t indice_count = indice.size(0);
  int64_t embedding_dim = parameter.dim() == 2 ? parameter.size(1) : 1;
  int64_t embedding_stride = parameter.stride(0);
  TORCH_CHECK(parameter.dim() == input.dim(), "embedding and input should be same dim");
  TORCH_CHECK(input.dim() == 1 || input.size(1) == embedding_dim, "input embedding dim should same as parameter");
  int64_t input_stride = input.stride(0);
  c10::ScalarType param_type = parameter.dtype().toScalarType();
  c10::ScalarType index_type = indice.dtype().toScalarType();
  whole_graph::WholeChunkedMemory_t wcmt = parameter.GetChunkedMemory();
  whole_graph::WholeMemoryChunkedScatter(whole_graph::pytorch::C10ScalarToWMType(param_type),
                                         whole_graph::pytorch::C10ScalarToWMType(index_type),
                                         input.data_ptr(),
                                         wcmt,
                                         indice.data_ptr(),
                                         parameter.storage_offset(),
                                         indice_count,
                                         embedding_dim,
                                         embedding_stride,
                                         input_stride,
                                         stream);
}

void WholeMemoryScatterNCCLCUDA(const torch::Tensor &input,
                                const torch::Tensor &indice,
                                int64_t pparameter) {
  NCCLTensor &parameter = *((NCCLTensor *) pparameter);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(indice.dim() == 1, "WholeMemoryScatterNCCLCUDA indice dim should be 1");
  TORCH_CHECK(indice.dtype() == torch::kInt32 || indice.dtype() == torch::kInt64,
              "WholeMemoryScatterNCCLCUDA indice dtype should be kInt32(kInt) or kInt64(kLong)");
  TORCH_CHECK(parameter.dim() == 1 || parameter.dim() == 2, "WholeMemoryScatterNCCLCUDA parameter dim should be 1 or 2")
  TORCH_CHECK(
      parameter.dtype() == torch::kInt8 || parameter.dtype() == torch::kInt16 || parameter.dtype() == torch::kInt32
          || parameter.dtype() == torch::kInt64 || parameter.dtype() == torch::kFloat16
          || parameter.dtype() == torch::kBFloat16 || parameter.dtype() == torch::kFloat32
          || parameter.dtype() == torch::kFloat64,
      "WholeMemoryScatterNCCLCUDA parameter dtype not supported");
  TORCH_CHECK(input.dtype() == parameter.dtype(),
              "WholeMemoryScatterNCCLCUDA input type should be same as parameter");
  TORCH_CHECK(indice.device() == input.device(), "indice and input should have same device.");
  int64_t indice_count = indice.size(0);
  int64_t embedding_dim = parameter.dim() == 2 ? parameter.size(1) : 1;
  int64_t embedding_stride = parameter.stride(0);
  TORCH_CHECK(parameter.dim() == input.dim(), "embedding and input should be same dim");
  TORCH_CHECK(input.dim() == 1 || input.size(1) == embedding_dim, "input embedding dim should same as parameter");
  int64_t input_stride = input.stride(0);
  c10::ScalarType param_type = parameter.dtype().toScalarType();
  c10::ScalarType index_type = indice.dtype().toScalarType();
  torch::Device d = indice.device();
  auto cuda_fns = GetCUDAEnvFns(d);
  whole_graph::WholeNCCLMemory_t wnmt = parameter.GetNCCLMemory();
  whole_graph::WholeMemoryNCCLScatter(whole_graph::pytorch::C10ScalarToWMType(param_type),
                                      whole_graph::pytorch::C10ScalarToWMType(index_type),
                                      input.data_ptr(),
                                      wnmt,
                                      indice.data_ptr(),
                                      parameter.storage_offset(),
                                      indice_count,
                                      embedding_dim,
                                      embedding_stride,
                                      input_stride,
                                      cuda_fns,
                                      stream);
}

torch::autograd::variable_list WholeMemoryExchangeEmbeddingGrads(const torch::Tensor &sparse_indices,
                                                                 const torch::Tensor &sparse_grads,
                                                                 int64_t total_entry_count,
                                                                 int64_t comm_ptr) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto *bcomm = (whole_graph::BootstrapCommunicator *) comm_ptr;
  TORCH_CHECK(sparse_indices.dim() == 1, "WholeMemoryExchangeEmbeddingGrads indice dim should be 1");
  TORCH_CHECK(sparse_indices.dtype() == torch::kInt32 || sparse_indices.dtype() == torch::kInt64,
              "WholeMemoryExchangeEmbeddingGrads sparse_indices dtype should be kInt32(kInt) or kInt64(kLong)");
  TORCH_CHECK(sparse_grads.dim() == 2, "WholeMemoryExchangeEmbeddingGrads sparse_grads dim should be 2");
  TORCH_CHECK(sparse_indices.size(0) == sparse_grads.size(0),
              "WholeMemoryExchangeEmbeddingGrads sparse_indices and sparse_grads size 0 not equal");
  TORCH_CHECK(
      sparse_grads.dtype() == torch::kFloat16 || sparse_grads.dtype() == torch::kFloat32
          || sparse_grads.dtype() == torch::kFloat64,
      "WholeMemoryExchangeEmbeddingGrads sparse_grads dtype not supported");
  TORCH_CHECK(sparse_indices.device() == sparse_grads.device(), "sparse_indices and sparse_grads should have same device.");
  int64_t embedding_dim = sparse_grads.size(1);
  int64_t embedding_stride = sparse_grads.stride(0);
  c10::ScalarType grad_type = sparse_grads.dtype().toScalarType();
  c10::ScalarType index_type = sparse_indices.dtype().toScalarType();
  torch::Device d = sparse_indices.device();
  auto cuda_fns = GetCUDAEnvFns(d);
  torch::Tensor local_indice_tensor, local_grad_tensor;
  auto local_indice_allocator = GetAllocatorForTensor<void>(local_indice_tensor, d, index_type, false);
  auto local_grad_allocator = GetAllocatorForTensor<void>(local_grad_tensor, d, grad_type, false);
  whole_graph::WholeMemoryExchangeEmbeddingGrads(whole_graph::pytorch::C10ScalarToWMType(index_type),
                                                 whole_graph::pytorch::C10ScalarToWMType(grad_type),
                                                 local_indice_allocator,
                                                 local_grad_allocator,
                                                 sparse_indices.data_ptr(),
                                                 sparse_grads.data_ptr(),
                                                 sparse_indices.size(0),
                                                 embedding_dim,
                                                 embedding_stride,
                                                 total_entry_count,
                                                 bcomm,
                                                 cuda_fns,
                                                 stream);
  return {local_indice_tensor, local_grad_tensor.reshape({local_indice_tensor.size(0), embedding_dim})};
}

}// namespace pytorch

}// namespace whole_graph

static auto registry = torch::RegisterOperators()
                           .op("wholegraph::gather", &whole_graph::pytorch::WholeMemoryGatherCUDANoGrad)
                           .op("wholegraph::gather_need_grad", &whole_graph::pytorch::WholeMemoryGatherCUDAGrad)
                           .op("wholegraph::gather_chunked", &whole_graph::pytorch::WholeMemoryGatherChunkedCUDANoGrad)
                           .op("wholegraph::gather_chunked_need_grad", &whole_graph::pytorch::WholeMemoryGatherChunkedCUDAGrad)
                           .op("wholegraph::gather_nccl", &whole_graph::pytorch::WholeMemoryGatherNCCLCUDANoGrad)
                           .op("wholegraph::gather_nccl_need_grad", &whole_graph::pytorch::WholeMemoryGatherNCCLCUDAGrad)
                           .op("wholegraph::scatter", &whole_graph::pytorch::WholeMemoryScatterCUDA)
                           .op("wholegraph::scatter_chunked", &whole_graph::pytorch::WholeMemoryScatterChunkedCUDA)
                           .op("wholegraph::scatter_nccl", &whole_graph::pytorch::WholeMemoryScatterNCCLCUDA)
                           .op("wholegraph::exchange_embedding_grads", &whole_graph::pytorch::WholeMemoryExchangeEmbeddingGrads);
