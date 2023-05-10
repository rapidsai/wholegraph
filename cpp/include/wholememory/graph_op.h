/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#pragma once
#include <cuda_runtime_api.h>

#include <wholememory/env_func_ptrs.h>
#include <wholememory/wholememory.h>
#include <wholememory/wholememory_tensor.h>
#define SUM_AGGREGATOR  0
#define MEAN_AGGREGATOR 1
#define GCN_AGGREGATOR  2

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Append Unique op
 * @param target_nodes_tensor : Wholememory Tensor of graph csr_row_ptr
 * @param neighbor_nodes_tensor : Wholememory Tensor of graph csr_col_ptr
 * @param output_unique_node_memory_context : memory context to output dest nodes
 * @param output_neighbor_raw_to_unique_mapping_tensor : pointer to output sample offset, optional
 * output
 * @param p_env_fns : pointers to environment functions.
 * @param stream : CUDA stream to use
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t graph_append_unique(
  wholememory_tensor_t target_nodes_tensor,
  wholememory_tensor_t neighbor_nodes_tensor,
  void* output_unique_node_memory_context,
  wholememory_tensor_t output_neighbor_raw_to_unique_mapping_tensor,
  wholememory_env_func_t* p_env_fns,
  void* stream);

/**
 * Spmm CSR no Weight Forward Op
 * @param csr_row_ptr_tensor : Wholememory Tensor of local graph csr_row_ptr
 * @param csr_col_ptr_tensor : Wholememory Tensor of local graph csr_col_ptr
 * @param feature_tensor : Wholememory Tensor of features
 * @param aggregator : aggreagtor type
 * @param output_feature_tensor : Wholememory Tensor of output features
 * @param stream : CUDA stream to use
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t spmm_csr_no_weight_forward(wholememory_tensor_t csr_row_ptr_tensor,
                                                    wholememory_tensor_t csr_col_ptr_tensor,
                                                    wholememory_tensor_t feature_tensor,
                                                    int64_t aggregator,
                                                    wholememory_tensor_t output_feature_tensor,
                                                    void* stream);

/**
 * Spmm CSR no Weight Backward Op
 * @param csr_row_ptr_tensor : Wholememory Tensor of local graph csr_row_ptr
 * @param csr_col_ptr_tensor : Wholememory Tensor of local graph csr_col_ptr
 * @param input_grad_tensor : Wholememory Tensor of input_grad_tensor
 * @param aggregator : aggreagtor type
 * @param output_grad_feature_tensor : Wholememory Tensor of output_grad_feature_tensor
 * @param stream : CUDA stream to use
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t spmm_csr_no_weight_backward(
  wholememory_tensor_t csr_row_ptr_tensor,
  wholememory_tensor_t csr_col_ptr_tensor,
  wholememory_tensor_t input_grad_tensor,
  int64_t aggregator,
  wholememory_tensor_t output_grad_feature_tensor,
  void* stream);

/**
 * SpADD CSR Forward Op
 * @param csr_row_ptr_tensor : Wholememory Tensor of local graph csr_row_ptr
 * @param csr_col_ptr_tensor : Wholememory Tensor of local graph csr_col_ptr
 * @param edge_weight_left_tensor : Wholememory Tensor of edge_weight_left_tensor
 * @param edge_weight_right_tensor : Wholememory Tensor of edge_weight_right_tensor
 * @param output_score_tensor : Wholememory Tensor of output_score_tensor
 * @param stream : CUDA stream to use
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t spadd_gat_csr_foward(wholememory_tensor_t csr_row_ptr_tensor,
                                              wholememory_tensor_t csr_col_ptr_tensor,
                                              wholememory_tensor_t edge_weight_left_tensor,
                                              wholememory_tensor_t edge_weight_right_tensor,
                                              wholememory_tensor_t output_score_tensor,
                                              void* stream);

/**
 * SpADD CSR Backward Op
 * @param csr_row_ptr_tensor : Wholememory Tensor of local graph csr_row_ptr
 * @param csr_col_ptr_tensor : Wholememory Tensor of local graph csr_col_ptr
 * @param grad_score_tensor : Wholememory Tensor of grad_score_tensor
 * @param output_grad_edge_weight_left_tensor : Wholememory Tensor of output_edge_weight_left_tensor
 * @param output_grad_edge_weight_right_tensor : Wholememory Tensor of
 * output_edge_weight_right_tensor
 * @param stream : CUDA stream to use
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t spadd_gat_csr_backward(
  wholememory_tensor_t csr_row_ptr_tensor,
  wholememory_tensor_t csr_col_ptr_tensor,
  wholememory_tensor_t grad_score_tensor,
  wholememory_tensor_t output_grad_edge_weight_left_tensor,
  wholememory_tensor_t output_grad_edge_weight_right_tensor,
  void* stream);

/**
 * EdgeWeightSoftmax Forwrd Op
 * @param csr_row_ptr_tensor : Wholememory Tensor of local graph csr_row_ptr
 * @param edge_weight_tensor : Wholememory Tensor of edge_weight_tensor
 * @param output_edge_weight_softmax_tensor : Wholememory Tensor of
 * output_edge_weight_softmax_tensor
 * @param stream : CUDA stream to use
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t edge_weight_softmax_csr_forward(
  wholememory_tensor_t csr_row_ptr_tensor,
  wholememory_tensor_t edge_weight_tensor,
  wholememory_tensor_t output_edge_weight_softmax_tensor,
  void* stream);

/**
 * EdgeWeightSoftmax Backward Op
 * @param csr_row_ptr_tensor : Wholememory Tensor of local graph csr_row_ptr
 * @param edge_weight_tensor : Wholememory Tensor of edge_weight_tensor
 * @param grad_edge_weight_softmax_tensor : Wholememory Tensor of grad_edge_weight_softmax_tensor
 * @param output_edge_weight_tensor : Wholememory Tensor of output_edge_weight_tensor
 * @param stream : CUDA stream to use
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t edge_weight_softmax_csr_backward(
  wholememory_tensor_t csr_row_ptr_tensor,
  wholememory_tensor_t edge_weight_tensor,
  wholememory_tensor_t grad_edge_weight_softmax_tensor,
  wholememory_tensor_t output_edge_weight_tensor,
  void* stream);

/**
 * Csr Add Self Loop Op
 * @param csr_row_ptr_tensor : Wholememory Tensor of local graph csr_row_ptr
 * @param csr_col_ptr_tensor : Wholememory Tensor of csr_col_ptr
 * @param output_csr_row_ptr_tensor : Wholememory Tensor of output_csr_row_ptr
 * @param output_csr_col_ptr_tensor : Wholememory Tensor of output_csr_col_ptr
 * @param stream : CUDA stream to use
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t csr_add_self_loop(wholememory_tensor_t csr_row_ptr_tensor,
                                           wholememory_tensor_t csr_col_ptr_tensor,
                                           wholememory_tensor_t output_csr_row_ptr_tensor,
                                           wholememory_tensor_t output_csr_col_ptr_tensor,
                                           void* stream);

/**
 * Csr Add Self Loop Op
 * @param csr_row_ptr_tensor : Wholememory Tensor of local graph csr_row_ptr
 * @param csr_col_ptr_tensor : Wholememory Tensor of csr_col_ptr
 * @param edge_weight_tensor : Wholememory Tensor of edge_weight
 * @param feature_tensor : Wholememory Tensor of feature
 * @param output_feature_tensor : Wholememory Tensor of output_feature
 * @param stream : CUDA stream to use
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t gspmm_csr_weighted_forward(wholememory_tensor_t csr_row_ptr_tensor,
                                                    wholememory_tensor_t csr_col_ptr_tensor,
                                                    wholememory_tensor_t edge_weight_tensor,
                                                    wholememory_tensor_t feature_tensor,
                                                    wholememory_tensor_t output_feature_tensor,
                                                    void* stream);

/**
 * Csr Add Self Loop Op
 * @param csr_row_ptr_tensor : Wholememory Tensor of local graph csr_row_ptr
 * @param csr_col_ptr_tensor : Wholememory Tensor of csr_col_ptr
 * @param edge_weight_tensor : Wholememory Tensor of edge_weight
 * @param feature_tensor : Wholememory Tensor of feature
 * @param input_grad_feature_tensor : Wholememory Tensor of input_grad_feature
 * @param output_grad_edge_weight_tensor : Wholememory Tensor of output_grad_edge_weight
 * @param output_grad_feature_tensor : Wholememory Tensor of output_grad_feature
 * @param stream : CUDA stream to use
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t gspmm_csr_weighted_backward(
  wholememory_tensor_t csr_row_ptr_tensor,
  wholememory_tensor_t csr_col_ptr_tensor,
  wholememory_tensor_t edge_weight_tensor,
  wholememory_tensor_t feature_tensor,
  wholememory_tensor_t input_grad_feature_tensor,
  wholememory_tensor_t output_grad_edge_weight_tensor,
  wholememory_tensor_t output_grad_feature_tensor,
  void* stream);

#ifdef __cplusplus
}
#endif
