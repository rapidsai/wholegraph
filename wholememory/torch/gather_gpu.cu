#include <pybind11/pybind11.h>
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>

#include "whole_chunked_pytorch_tensor.h"
#include "whole_memory_embedding.h"

namespace whole_memory {

namespace pytorch {

torch::Tensor WholeMemoryGatherCUDA(torch::Tensor indice, torch::Tensor parameter, torch::ScalarType output_scalar_type) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  caffe2::TypeMeta output_meta_type;
  output_meta_type = output_scalar_type;
  TORCH_CHECK(indice.dim() == 1, "WholeMemoryGatherCUDA indice dim should be 1");
  TORCH_CHECK(indice.dtype() == torch::kInt32 || indice.dtype() == torch::kInt64,
              "WholeMemoryGatherCUDA indice dtype should be kInt32(kInt) or kInt64(kLong)");
  TORCH_CHECK(parameter.dim() == 2, "WholeMemoryGatherCUDA parameter dim should be 2")
  TORCH_CHECK(parameter.dtype() == torch::kFloat16 || parameter.dtype() == torch::kBFloat16 ||
      parameter.dtype() == torch::kFloat32 || parameter.dtype() == torch::kFloat64,
              "WholeMemoryGatherCUDA parameter dtype should be half float double or bfloat16");
  TORCH_CHECK(output_meta_type == torch::kFloat16 || output_meta_type == torch::kBFloat16 ||
      output_meta_type == torch::kFloat32 || output_meta_type == torch::kFloat64,
              "WholeMemoryGatherCUDA output type should be half float double or bfloat16");
  TORCH_CHECK(indice.device() == parameter.device(), "indice and parameter should have same device.");
  int64_t indice_count = indice.size(0);
  int64_t embedding_dim = parameter.size(1);
  int64_t embedding_stride = parameter.stride(0);
  torch::Device d = indice.device();
  c10::ScalarType output_type = output_scalar_type;
  c10::ScalarType param_type = parameter.dtype().toScalarType();
  c10::ScalarType index_type = indice.dtype().toScalarType();
  auto to = torch::TensorOptions().device(d).dtype(output_meta_type).requires_grad(false);
  torch::Tensor output_tensor = torch::empty({indice_count, embedding_dim}, to);
  whole_memory::WholeMemoryGather(whole_memory::pytorch::C10ScalarToWMType(output_type),
                                  whole_memory::pytorch::C10ScalarToWMType(param_type),
                                  whole_memory::pytorch::C10ScalarToWMType(index_type),
                                  output_tensor.data_ptr(),
                                  parameter.data_ptr(),
                                  indice.data_ptr(),
                                  parameter.storage_offset(),
                                  indice_count,
                                  embedding_dim,
                                  embedding_stride,
                                  embedding_dim,
                                  stream);
  return output_tensor;
}

torch::Tensor WholeMemoryGatherChunkedCUDA(torch::Tensor indice, int64_t pparameter, torch::ScalarType output_scalar_type) {
  ChunkedTensor& parameter = *((ChunkedTensor*)pparameter);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  caffe2::TypeMeta output_meta_type;
  output_meta_type = output_scalar_type;
  TORCH_CHECK(indice.dim() == 1, "WholeMemoryGatherCUDA indice dim should be 1");
  TORCH_CHECK(indice.dtype() == torch::kInt32 || indice.dtype() == torch::kInt64,
              "WholeMemoryGatherCUDA indice dtype should be kInt32(kInt) or kInt64(kLong)");
  TORCH_CHECK(parameter.dim() == 2, "WholeMemoryGatherCUDA parameter dim should be 2")
  TORCH_CHECK(parameter.dtype() == torch::kFloat16 || parameter.dtype() == torch::kBFloat16 ||
      parameter.dtype() == torch::kFloat32 || parameter.dtype() == torch::kFloat64,
              "WholeMemoryGatherCUDA parameter dtype should be half float double or bfloat16");
  TORCH_CHECK(output_meta_type == torch::kFloat16 || output_meta_type == torch::kBFloat16 ||
      output_meta_type == torch::kFloat32 || output_meta_type == torch::kFloat64,
              "WholeMemoryGatherCUDA output type should be half float double or bfloat16");
  int64_t indice_count = indice.size(0);
  int64_t embedding_dim = parameter.size(1);
  int64_t embedding_stride = parameter.stride(0);
  torch::Device d = indice.device();
  c10::ScalarType output_type = output_scalar_type;
  c10::ScalarType param_type = parameter.dtype().toScalarType();
  c10::ScalarType index_type = indice.dtype().toScalarType();
  auto to = torch::TensorOptions().device(d).dtype(output_meta_type).requires_grad(false);
  torch::Tensor output_tensor = torch::empty({indice_count, embedding_dim}, to);
  whole_memory::WholeChunkedMemory_t wcmt = parameter.GetChunkedMemory();
  whole_memory::WholeMemoryChunkedGather(whole_memory::pytorch::C10ScalarToWMType(output_type),
                                         whole_memory::pytorch::C10ScalarToWMType(param_type),
                                         whole_memory::pytorch::C10ScalarToWMType(index_type),
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

}

}

static auto registry = torch::RegisterOperators()
    .op("wholememory::gather", &whole_memory::pytorch::WholeMemoryGatherCUDA)
    .op("wholememory::gather_chunked", &whole_memory::pytorch::WholeMemoryGatherChunkedCUDA);
