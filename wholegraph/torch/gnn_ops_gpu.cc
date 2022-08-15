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
#include <stdint.h>
#include <stdio.h>
#include <torch/script.h>

#include "../gnn_ops.h"
#include "pytorch_dtype.h"

torch::Tensor spmm_csr_noweight_forward(const torch::Tensor &csr_row_ptr,
                                        const torch::Tensor &csr_col_ind,
                                        const torch::Tensor &x,
                                        int64_t aggregator) {
  TORCH_CHECK(aggregator >= 0 && aggregator < 3, "aggregator should be [0, 1, 2]");
  TORCH_CHECK(csr_row_ptr.dtype() == torch::ScalarType::Int, "CSR row_ptr should be Int tensor.");
  TORCH_CHECK(csr_col_ind.dtype() == torch::ScalarType::Int, "CSR col_ind should be Int tensor.");
  TORCH_CHECK(x.dtype() == torch::ScalarType::Float || x.dtype() == torch::ScalarType::Half,
              "x should be half or float.");
  TORCH_CHECK(csr_row_ptr.dim() == 1, "CSR row_ptr should be 1-D tensor.");
  TORCH_CHECK(csr_col_ind.dim() == 1, "CSR col_ind should be 1-D tensor.");
  TORCH_CHECK(x.dim() == 2, "x should be 2-D tensor.");
  TORCH_CHECK(csr_row_ptr.sizes()[0] >= 1, "CSR row_ptr size should be larger than 0 tensor.");
  int64_t target_vec_count = csr_row_ptr.sizes()[0] - 1;
  int64_t embedding_dim = x.sizes()[1];
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  torch::TensorOptions options;

  options = options.dtype(x.dtype())
                .device(x.device())
                .requires_grad(x.requires_grad());
  std::vector<int64_t> size{target_vec_count, embedding_dim};
  torch::Tensor output = torch::empty(size, options);

  whole_graph::SpmmCsrNoWeightForward(whole_graph::pytorch::C10ScalarToWMType(x.dtype().toScalarType()),
                                      csr_row_ptr.data_ptr<int>(),
                                      target_vec_count,
                                      csr_col_ind.data_ptr<int>(),
                                      csr_col_ind.sizes()[0],
                                      x.data_ptr(),
                                      embedding_dim,// embedding dim
                                      x.stride(0),
                                      x.sizes()[0],// total column count
                                      output.data_ptr(),
                                      embedding_dim,
                                      aggregator,
                                      stream);

  return output;
}

torch::Tensor spmm_csr_noweight_backward(const torch::Tensor &csr_row_ptr,
                                         const torch::Tensor &csr_col_ind,
                                         const torch::Tensor &sample_dup_count,
                                         const torch::Tensor &grad_output,
                                         int64_t aggregator) {
  TORCH_CHECK(aggregator >= 0 && aggregator < 3, "aggregator should be [0, 1, 2]");
  TORCH_CHECK(csr_row_ptr.dtype() == torch::ScalarType::Int, "CSR row_ptr should be Int tensor.");
  TORCH_CHECK(csr_col_ind.dtype() == torch::ScalarType::Int, "CSR col_ind should be Int tensor.");
  TORCH_CHECK(sample_dup_count.dtype() == torch::ScalarType::Int, "sample_dup_count should be Int tensor.");
  TORCH_CHECK(grad_output.dtype() == torch::ScalarType::Float || grad_output.dtype() == torch::ScalarType::Half,
              "grad_output should be half or float.");
  TORCH_CHECK(csr_row_ptr.dim() == 1, "CSR row_ptr should be 1-D tensor.");
  TORCH_CHECK(csr_col_ind.dim() == 1, "CSR col_ind should be 1-D tensor.");
  TORCH_CHECK(grad_output.dim() == 2, "x should be 2-D tensor.");
  TORCH_CHECK(sample_dup_count.dim() == 1, "sample_dup_count should be 1-D tensor.");
  TORCH_CHECK(csr_row_ptr.sizes()[0] >= 1, "CSR row_ptr size should be larger than 0.");

  int64_t target_vec_count = csr_row_ptr.sizes()[0] - 1;
  int64_t neighboors_count = csr_col_ind.sizes()[0];
  int64_t input_count = sample_dup_count.sizes()[0];

  int64_t embedding_dim = grad_output.size(1);
  int64_t embedding_stride = grad_output.stride(0);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  torch::TensorOptions options;

  options = options.dtype(grad_output.dtype())
                .device(grad_output.device());
  std::vector<int64_t> size{input_count, embedding_dim};
  torch::Tensor grad_x = torch::empty(size, options);

  SpmmCsrNoWeightBackword(whole_graph::pytorch::C10ScalarToWMType(grad_output.dtype().toScalarType()),
                          csr_row_ptr.data_ptr<int>(),
                          csr_col_ind.data_ptr<int>(),
                          sample_dup_count.data_ptr<int>(),
                          grad_output.data_ptr(),
                          embedding_stride,
                          grad_x.data_ptr(),
                          embedding_dim,
                          target_vec_count,// total row count,
                          neighboors_count,
                          input_count,// total column count
                          embedding_dim,
                          aggregator,
                          stream);

  return grad_x;
}

static std::vector<int64_t> check_input_and_compute_output(
    const torch::Tensor &csr_row_ptr,
    const torch::Tensor &csr_col_ind,
    const torch::Tensor &x,
    const torch::Tensor &edge_weight) {
  TORCH_CHECK(csr_row_ptr.dtype() == torch::ScalarType::Int, "CSR row_ptr should be Int tensor.");
  TORCH_CHECK(csr_col_ind.dtype() == torch::ScalarType::Int, "CSR col_ind should be Int tensor.");
  TORCH_CHECK(x.dtype() == torch::ScalarType::Float || x.dtype() == torch::ScalarType::Half,
              "x should be half or float.");
  TORCH_CHECK(edge_weight.dtype() == torch::ScalarType::Float || edge_weight.dtype() == torch::ScalarType::Half,
              "edge_weight should be half or float");
  TORCH_CHECK(x.dtype() == edge_weight.dtype(), "x and edge_weight should have same type.");
  TORCH_CHECK(csr_row_ptr.dim() == 1, "CSR row_ptr should be 1-D tensor.");
  TORCH_CHECK(csr_col_ind.dim() == 1, "CSR col_ind should be 1-D tensor.");
  TORCH_CHECK(x.dim() == 3, "x should be 3-D tensor.");
  TORCH_CHECK(edge_weight.dim() == 2, "edge_weight should be 2-D tensor.");
  TORCH_CHECK(csr_row_ptr.sizes()[0] >= 1, "CSR row_ptr size should be larger than 0 tensor.");
  int64_t target_vec_count = csr_row_ptr.sizes()[0] - 1;
  int64_t total_edge_count = csr_col_ind.size(0);
  TORCH_CHECK(edge_weight.size(0) == total_edge_count, "edge_weight[0] should be total edge count.");
  int64_t num_head = edge_weight.size(1);
  TORCH_CHECK(x.size(1) == num_head, "x[1] should be num_head.");
  int64_t embedding_dim = x.size(2);
  std::vector<int64_t> size{target_vec_count, num_head, embedding_dim};
  return size;
}

torch::Tensor gspmm_csr_weighted_forward(const torch::Tensor &csr_row_ptr,
                                         const torch::Tensor &csr_col_ind,
                                         const torch::Tensor &x,
                                         const torch::Tensor &edge_weight) {
  std::vector<int64_t> size = check_input_and_compute_output(csr_row_ptr, csr_col_ind, x, edge_weight);
  TORCH_CHECK(size.size() == 3, "size should have 3 value.");
  int64_t target_vec_count = size[0];
  int64_t num_head = size[1];
  int64_t embedding_dim = size[2];
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  torch::TensorOptions options;

  options = options.dtype(x.dtype())
                .device(x.device())
                .requires_grad(x.requires_grad() || edge_weight.requires_grad());

  torch::Tensor output = torch::empty(size, options);

  // (TN, num_heads, d_out)
  gSpmmCsrWeightedForward(
      whole_graph::pytorch::C10ScalarToWMType(x.dtype().toScalarType()),
      csr_row_ptr.data_ptr<int>(),
      target_vec_count,// total row count
      csr_col_ind.data_ptr<int>(),
      csr_col_ind.size(0),
      edge_weight.data_ptr(),
      num_head,
      x.data_ptr(),
      embedding_dim,// embedding dim
      x.stride(1),
      x.stride(0),
      x.size(0),// total column count
      output.data_ptr(),
      output.stride(1),
      output.stride(0),
      stream);
  return output;
}

using torch::autograd::variable_list;

variable_list gspmm_csr_weighted_backward(const torch::Tensor &csr_row_ptr,
                                          const torch::Tensor &csr_col_ind,
                                          const torch::Tensor &x,
                                          const torch::Tensor &edge_weight,
                                          const torch::Tensor &sample_dup_count,
                                          const torch::Tensor &grad_output) {
  variable_list output_vars;
  std::vector<int64_t> size = check_input_and_compute_output(csr_row_ptr, csr_col_ind, x, edge_weight);
  TORCH_CHECK(size.size() == 3, "size should have 3 value.");
  int64_t target_vec_count = size[0];
  int64_t num_head = size[1];
  int64_t embedding_dim = size[2];
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(grad_output.dtype() == x.dtype(), "grad_output and x should have same type.");
  TORCH_CHECK(grad_output.dim() == 3 && grad_output.size(0) == size[0] && grad_output.size(1) == size[1]
                  && grad_output.size(2) == size[2],
              "grad_output not same as output size.");
  TORCH_CHECK(sample_dup_count.dim() == 1, "sample_dup_count should be 1-D tensor.");
  TORCH_CHECK(sample_dup_count.dtype() == torch::ScalarType::Int, "sample_dup_count should be Int tensor.");
  int64_t input_count = x.size(0);

  TORCH_CHECK(sample_dup_count.size(0) == input_count, "sample_dup_count should be same size as input_count.");

  TORCH_CHECK(grad_output.size(2) == grad_output.stride(1), "strided grad_output is not supported now.");

  void *grad_x_ptr = nullptr;
  void *grad_edge_weight_ptr = nullptr;
  if (x.requires_grad()) {
    torch::TensorOptions options;
    options = options.dtype(x.dtype()).device(x.device());
    torch::Tensor grad_x = torch::empty(x.sizes(), options);
    grad_x_ptr = grad_x.data_ptr();
    output_vars.push_back(grad_x);
  }
  if (edge_weight.requires_grad()) {
    torch::TensorOptions options;
    options = options.dtype(edge_weight.dtype()).device(edge_weight.device());
    torch::Tensor grad_edge_weight = torch::empty(edge_weight.sizes(), options);
    grad_edge_weight_ptr = grad_edge_weight.data_ptr();
    output_vars.push_back(grad_edge_weight);
  }
  gSpmmCsrWeightedFusedSharedMemoryBackward(
      whole_graph::pytorch::C10ScalarToWMType(grad_output.dtype().toScalarType()),
      x.requires_grad(), edge_weight.requires_grad(),
      csr_row_ptr.data_ptr<int>(),
      target_vec_count,// total row count
      csr_col_ind.data_ptr<int>(),
      csr_col_ind.size(0),
      edge_weight.data_ptr(),
      sample_dup_count.data_ptr<int>(),
      x.data_ptr(), embedding_dim, num_head,
      x.stride(1),
      x.stride(0),
      x.size(0),
      grad_output.data_ptr(),
      grad_output.stride(1),
      grad_output.stride(0),
      grad_x_ptr,
      embedding_dim,
      embedding_dim * num_head,
      grad_edge_weight_ptr,
      stream);

  return output_vars;
}

torch::Tensor spadd_gat_csr_forward(const torch::Tensor &csr_row_ptr,
                                    const torch::Tensor &csr_col_ind,
                                    const torch::Tensor &edge_weight_left,
                                    const torch::Tensor &edge_weight_right) {

  TORCH_CHECK(csr_row_ptr.dtype() == torch::ScalarType::Int, "CSR row_ptr shold be Int tensor.");
  TORCH_CHECK(csr_col_ind.dtype() == torch::ScalarType::Int, "CSR col_ind should be Int tensor.");
  TORCH_CHECK(
      edge_weight_left.dtype() == torch::ScalarType::Float || edge_weight_left.dtype() == torch::ScalarType::Half,
      "edge_weight_left shold be half or float.");
  TORCH_CHECK(
      edge_weight_right.dtype() == torch::ScalarType::Float || edge_weight_right.dtype() == torch::ScalarType::Half,
      "edge_weight_left shold be half or float.");
  TORCH_CHECK(edge_weight_left.dtype() == edge_weight_right.dtype(),
              "edge_weight_left data type should be equal to edge_weight_right data type.");
  TORCH_CHECK(csr_row_ptr.dim() == 1, "CSR row_ptr should be 1-D tensor.");
  TORCH_CHECK(csr_col_ind.dim() == 1, "CSR col_ind should be 1-D tensor.");
  TORCH_CHECK(edge_weight_left.dim() == 2, "edge_weight_left should be 2-D tensor.");
  TORCH_CHECK(edge_weight_right.dim() == 2, "edge_weight_right should be 2-D tensor.");

  int64_t target_vec_count = csr_row_ptr.sizes()[0] - 1;
  TORCH_CHECK(edge_weight_left.sizes()[0] == target_vec_count,
              "edge_weight_left rows should be equal to target count.");

  int num_head = edge_weight_left.sizes()[1];
  TORCH_CHECK(edge_weight_left.sizes()[1] == edge_weight_right.sizes()[1],
              "edge_weight_left num heads should be equal to edge_weight_right num heads.");

  bool grad_required = edge_weight_left.requires_grad() || edge_weight_right.requires_grad();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  torch::TensorOptions options;
  options = options.dtype(edge_weight_left.dtype())
                .device(edge_weight_left.device())
                .requires_grad(grad_required);
  std::vector<int64_t> output_size{csr_col_ind.sizes()[0], num_head};
  torch::Tensor output = torch::empty(output_size, options);

  SpAddGATCSRForward(whole_graph::pytorch::C10ScalarToWMType(edge_weight_left.dtype().toScalarType()),
                     csr_row_ptr.data_ptr<int>(), target_vec_count,
                     csr_col_ind.data_ptr<int>(),
                     edge_weight_left.data_ptr(),
                     edge_weight_right.data_ptr(),
                     num_head,
                     output.data_ptr(), stream);

  return output;
}

torch::autograd::variable_list spadd_gat_csr_backward(const torch::Tensor &csr_row_ptr,
                                                      const torch::Tensor &csr_col_ind,
                                                      const torch::Tensor &sample_dup_count,
                                                      const torch::Tensor &grad_y) {
  TORCH_CHECK(csr_row_ptr.dtype() == torch::ScalarType::Int, "CSR row_ptr shold be Int tensor.");
  TORCH_CHECK(csr_col_ind.dtype() == torch::ScalarType::Int, "CSR col_ind should be Int tensor.");
  TORCH_CHECK(sample_dup_count.dtype() == torch::ScalarType::Int, "sample_dup_count should be Int tensor.");
  TORCH_CHECK(csr_row_ptr.dim() == 1, "CSR row_ptr should be 1-D tensor.");
  TORCH_CHECK(csr_col_ind.dim() == 1, "CSR col_ind should be 1-D tensor.");
  TORCH_CHECK(sample_dup_count.dim() == 1, "sample_dup_count should be 1-D tensor.");

  int64_t target_vec_count = csr_row_ptr.sizes()[0] - 1;
  int64_t neighbor_count = sample_dup_count.sizes()[0];
  int num_head = grad_y.sizes()[1];

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  torch::TensorOptions options;
  options = options.dtype(grad_y.dtype())
                .device(grad_y.device());
  std::vector<int64_t> output_weight_left_size{target_vec_count, num_head};
  std::vector<int64_t> output_weight_right_size{neighbor_count, num_head};
  torch::Tensor grad_weight_left = torch::empty(output_weight_left_size, options);
  torch::Tensor grad_weight_right = torch::empty(output_weight_right_size, options);

  if (grad_weight_left.dtype() == torch::ScalarType::Float) {

    SpAddGATCSRBackward(whole_graph::pytorch::C10ScalarToWMType(grad_y.dtype().toScalarType()),
                        csr_row_ptr.data_ptr<int>(),
                        csr_col_ind.data_ptr<int>(),
                        sample_dup_count.data_ptr<int>(),
                        grad_y.data_ptr(),
                        target_vec_count,
                        neighbor_count,
                        num_head,
                        grad_weight_left.data_ptr(),
                        grad_weight_right.data_ptr(),
                        stream);
  }

  return {grad_weight_left, grad_weight_right};
}

torch::Tensor edge_weight_softmax_forward(const torch::Tensor &csr_row_ptr,
                                          const torch::Tensor &edge_weight) {
  TORCH_CHECK(csr_row_ptr.dtype() == torch::ScalarType::Int, "CSR row_ptr shold be Int tensor.");
  //TORCH_CHECK(edge_weight.dtype() == torch::ScalarType::Float || edge_weight.dtype() == torch::ScalarType::Half, "edge_weight shold be half or float.");
  TORCH_CHECK(edge_weight.dtype() == torch::ScalarType::Float || edge_weight.dtype() == torch::ScalarType::Half,
              "edge_weight shold be half or float.");
  TORCH_CHECK(csr_row_ptr.dim() == 1, "CSR row_ptr should be 1-D tensor.");
  TORCH_CHECK(edge_weight.dim() == 2, "edge_weight should be 2-D tensor.");

  int64_t target_vec_count = csr_row_ptr.sizes()[0] - 1;
  int num_head = edge_weight.sizes()[1];
  bool grad_required = edge_weight.requires_grad();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  torch::TensorOptions options;
  options = options.dtype(edge_weight.dtype())
                .device(edge_weight.device())
                .requires_grad(grad_required);

  std::vector<int64_t> output_size{edge_weight.sizes()[0], edge_weight.sizes()[1]};
  torch::Tensor output = torch::empty(output_size, options);

  EdgeWeightSoftmaxForward(whole_graph::pytorch::C10ScalarToWMType(edge_weight.dtype().toScalarType()),
                           csr_row_ptr.data_ptr<int>(), target_vec_count,
                           edge_weight.data_ptr(),
                           num_head,
                           output.data_ptr(),
                           stream);

  return output;
}

torch::Tensor edge_weight_softmax_backward(const torch::Tensor &csr_row_ptr,
                                           const torch::Tensor &edge_weight_softmax,
                                           const torch::Tensor &grad_y) {
  TORCH_CHECK(csr_row_ptr.dtype() == torch::ScalarType::Int, "CSR row_ptr shold be Int tensor.");
  TORCH_CHECK(
      edge_weight_softmax.dtype() == torch::ScalarType::Float || edge_weight_softmax.dtype() == torch::ScalarType::Half,
      "edge_weight shold be half or float.");
  TORCH_CHECK(csr_row_ptr.dim() == 1, "CSR row_ptr should be 1-D tensor.");
  TORCH_CHECK(edge_weight_softmax.dim() == 2, "edge_weight should be 2-D tensor.");
  TORCH_CHECK(grad_y.dim() == 2, "grad_y should be 2-D tensor");
  TORCH_CHECK(grad_y.dtype() == edge_weight_softmax.dtype(), "grad_y and edge_weight_softmax should have same type.");
  TORCH_CHECK(edge_weight_softmax.sizes() == grad_y.sizes());

  int64_t target_vec_count = csr_row_ptr.sizes()[0] - 1;
  int num_head = edge_weight_softmax.sizes()[1];
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  torch::TensorOptions options;
  options = options.dtype(grad_y.dtype())
                .device(grad_y.device());

  std::vector<int64_t> output_size{edge_weight_softmax.sizes()[0], edge_weight_softmax.sizes()[1]};
  torch::Tensor output = torch::empty(output_size, options);

  EdgeWeightSoftmaxBackward(whole_graph::pytorch::C10ScalarToWMType(grad_y.dtype().toScalarType()),
                            csr_row_ptr.data_ptr<int>(),
                            edge_weight_softmax.data_ptr(),
                            grad_y.data_ptr(),
                            target_vec_count,
                            num_head,
                            output.data_ptr(),
                            stream);

  return output;
}

variable_list csr_add_self_loop(const torch::Tensor &csr_row_ptr,
                                const torch::Tensor &csr_col_ind,
                                const torch::Tensor &sample_dup_count) {
  TORCH_CHECK(csr_row_ptr.dtype() == torch::ScalarType::Int, "CSR row_ptr should be Int tensor.");
  TORCH_CHECK(csr_col_ind.dtype() == torch::ScalarType::Int, "CSR col_ind should be Int tensor.");
  TORCH_CHECK(sample_dup_count.dtype() == torch::ScalarType::Int, "CSR sample_dup_count should be Int tensor.");
  TORCH_CHECK(csr_row_ptr.dim() == 1, "CSR row_ptr should be 1-D tensor.");
  TORCH_CHECK(csr_col_ind.dim() == 1, "CSR col_ind should be 1-D tensor.");
  TORCH_CHECK(sample_dup_count.dim() == 1, "CSR sample_dup_count should be 1-D tensor.");
  TORCH_CHECK(csr_row_ptr.sizes()[0] >= 1, "CSR row_ptr size should be larger than 0 tensor.");
  int64_t total_target_count = csr_row_ptr.size(0) - 1;
  int64_t total_edge_count = csr_col_ind.size(0);
  torch::TensorOptions options;
  options = options.dtype(torch::ScalarType::Int).device(csr_row_ptr.device());
  torch::Tensor csr_row_ptr_looped = torch::empty(csr_row_ptr.sizes(), options);
  torch::Tensor csr_col_ind_looped = torch::empty({total_edge_count + total_target_count}, options);
  torch::Tensor sample_dup_count_looped = torch::empty(sample_dup_count.sizes(), options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  whole_graph::CSRAddSelfLoop(csr_row_ptr.data_ptr<int>(),
                              csr_col_ind.data_ptr<int>(),
                              sample_dup_count.data_ptr<int>(),
                              total_target_count,
                              sample_dup_count.size(0),
                              csr_row_ptr_looped.data_ptr<int>(),
                              csr_col_ind_looped.data_ptr<int>(),
                              sample_dup_count_looped.data_ptr<int>(),
                              stream);

  return {csr_row_ptr_looped, csr_col_ind_looped, sample_dup_count_looped};
}

torch::Tensor spmm_csr_relational_noweight_forward(const torch::Tensor &csr_row_ptr,
                                                   const torch::Tensor &csr_col_ind,
                                                   const torch::Tensor &edge_type,
                                                   const torch::Tensor &x,
                                                   int64_t num_relation,
                                                   int64_t aggregator) {
  TORCH_CHECK(num_relation > 0, "num_relation should be > 0");
  TORCH_CHECK(csr_row_ptr.dtype() == torch::ScalarType::Int, "CSR row_ptr should be Int tensor.");
  TORCH_CHECK(csr_col_ind.dtype() == torch::ScalarType::Int, "CSR col_ind should be Int tensor.");
  TORCH_CHECK(edge_type.dtype() == torch::ScalarType::Char, "edge_type should be Char tensor.");
  TORCH_CHECK(x.dtype() == torch::ScalarType::Float || x.dtype() == torch::ScalarType::Half,
              "x should be half or float.");
  TORCH_CHECK(csr_row_ptr.dim() == 1, "CSR row_ptr should be 1-D tensor.");
  TORCH_CHECK(csr_col_ind.dim() == 1, "CSR col_ind should be 1-D tensor.");
  TORCH_CHECK(x.dim() == 2, "x should be 2-D tensor.");
  TORCH_CHECK(csr_row_ptr.sizes()[0] >= 1, "CSR row_ptr size should be larger than 0 tensor.");
  TORCH_CHECK(csr_col_ind.sizes()[0] == edge_type.sizes()[0], "CSR col_ind should be same size as edge_type");
  TORCH_CHECK(aggregator >= 0 && aggregator < 2, "aggregator should be 0, 1");
  int64_t target_vec_count = csr_row_ptr.sizes()[0] - 1;
  int64_t embedding_dim = x.sizes()[1];
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  torch::TensorOptions options;

  options = options.dtype(x.dtype())
                .device(x.device())
                .requires_grad(x.requires_grad());
  std::vector<int64_t> size{target_vec_count, num_relation * embedding_dim};
  torch::Tensor output = torch::empty(size, options);

  whole_graph::SpmmCsrRelationalNoWeightForward(whole_graph::pytorch::C10ScalarToWMType(x.dtype().toScalarType()),
                                                csr_row_ptr.data_ptr<int>(),
                                                target_vec_count,
                                                csr_col_ind.data_ptr<int>(),
                                                csr_col_ind.sizes()[0],
                                                edge_type.data_ptr<int8_t>(),
                                                x.data_ptr(),
                                                embedding_dim,// embedding dim
                                                x.stride(0),
                                                x.sizes()[0],// total column count
                                                output.data_ptr(),
                                                embedding_dim * num_relation,
                                                num_relation,
                                                aggregator,
                                                stream);

  return output;
}

torch::Tensor spmm_csr_relational_noweight_backward(const torch::Tensor &csr_row_ptr,
                                                    const torch::Tensor &csr_col_ind,
                                                    const torch::Tensor &edge_type,
                                                    const torch::Tensor &sample_dup_count,
                                                    const torch::Tensor &grad_output,
                                                    int64_t num_relation,
                                                    int64_t aggregator) {
  TORCH_CHECK(num_relation > 0, "num_relation should be > 0");
  TORCH_CHECK(csr_row_ptr.dtype() == torch::ScalarType::Int, "CSR row_ptr should be Int tensor.");
  TORCH_CHECK(csr_col_ind.dtype() == torch::ScalarType::Int, "CSR col_ind should be Int tensor.");
  TORCH_CHECK(edge_type.dtype() == torch::ScalarType::Char, "edge_type should be Char tensor.");
  TORCH_CHECK(sample_dup_count.dtype() == torch::ScalarType::Int, "sample_dup_count should be Int tensor.");
  TORCH_CHECK(grad_output.dtype() == torch::ScalarType::Float || grad_output.dtype() == torch::ScalarType::Half,
              "grad_output should be half or float.");
  TORCH_CHECK(csr_row_ptr.dim() == 1, "CSR row_ptr should be 1-D tensor.");
  TORCH_CHECK(csr_col_ind.dim() == 1, "CSR col_ind should be 1-D tensor.");
  TORCH_CHECK(grad_output.dim() == 2, "x should be 2-D tensor.");
  TORCH_CHECK(sample_dup_count.dim() == 1, "sample_dup_count should be 1-D tensor.");
  TORCH_CHECK(csr_row_ptr.sizes()[0] >= 1, "CSR row_ptr size should be larger than 0.");
  TORCH_CHECK(csr_col_ind.sizes()[0] == edge_type.sizes()[0], "CSR col_ind should be same size as edge_type");
  TORCH_CHECK(aggregator >= 0 && aggregator < 2, "aggregator should be 0, 1");

  int64_t target_vec_count = csr_row_ptr.sizes()[0] - 1;
  int64_t neighboors_count = csr_col_ind.sizes()[0];
  int64_t input_count = sample_dup_count.sizes()[0];

  TORCH_CHECK(grad_output.size(1) % num_relation == 0, "grad_output last not divided by num_relation");
  int64_t embedding_dim = grad_output.size(1) / num_relation;
  int64_t embedding_stride = grad_output.stride(0);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  torch::TensorOptions x_options;

  x_options = x_options.dtype(grad_output.dtype())
                  .device(grad_output.device());
  std::vector<int64_t> size{input_count, embedding_dim};
  torch::Tensor grad_x = torch::empty(size, x_options);

  SpmmCsrRelationalNoWeightBackward(whole_graph::pytorch::C10ScalarToWMType(grad_output.dtype().toScalarType()),
                                    csr_row_ptr.data_ptr<int>(),
                                    csr_col_ind.data_ptr<int>(),
                                    edge_type.data_ptr<int8_t>(),
                                    sample_dup_count.data_ptr<int>(),
                                    grad_output.data_ptr(),
                                    embedding_stride,
                                    grad_x.data_ptr(),
                                    embedding_dim,
                                    target_vec_count,// total row count,
                                    neighboors_count,
                                    input_count,// total column count
                                    embedding_dim,
                                    num_relation,
                                    aggregator,
                                    stream);

  return grad_x;
}

static auto registry =
    torch::RegisterOperators()
        .op("wholegraph::spmm_csr_noweight_forward", &spmm_csr_noweight_forward)
        .op("wholegraph::spmm_csr_noweight_backward", &spmm_csr_noweight_backward)
        .op("wholegraph::gspmm_csr_weighted_forward", &gspmm_csr_weighted_forward)
        .op("wholegraph::gspmm_csr_weighted_backward", &gspmm_csr_weighted_backward)
        .op("wholegraph::spadd_gat_csr_forward", &spadd_gat_csr_forward)
        .op("wholegraph::spadd_gat_csr_backward", &spadd_gat_csr_backward)
        .op("wholegraph::edge_weight_softmax_forward", &edge_weight_softmax_forward)
        .op("wholegraph::edge_weight_softmax_backward", &edge_weight_softmax_backward)
        .op("wholegraph::csr_add_self_loop", &csr_add_self_loop)
        .op("wholegraph::spmm_csr_relational_noweight_forward", &spmm_csr_relational_noweight_forward)
        .op("wholegraph::spmm_csr_relational_noweight_backward", &spmm_csr_relational_noweight_backward);
