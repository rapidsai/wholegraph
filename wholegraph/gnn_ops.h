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
#pragma once

#include "data_type.h"
#include "whole_chunked_memory.h"

namespace whole_graph {

void SpmmCsrNoWeightForward(WMType data_type,
                            const int *csr_row_ptr,
                            int m,// total row count
                            const int *csr_col_ind,
                            int total_nz_element,
                            const void *input,
                            int n,// embedding dim
                            int ldb,
                            int k,// total column count
                            void *output,
                            int ldc,
                            int aggregator,
                            cudaStream_t stream);

void SpmmCsrNoWeightBackword(WMType data_type,
                             const int *csr_row_ptr,
                             const int *csr_col_ind,
                             const int *sample_dup_count,
                             const void *grad_output,
                             int grad_out_stride,
                             void *grad_x,
                             int grad_x_stride,
                             int m,// total row count,
                             int total_nz_element,
                             int k,// total column count
                             int embedding_dim,
                             int aggregator,
                             cudaStream_t stream);

void gSpmmCsrWeightedForward(WMType data_type,
                             const int *csr_row_ptr,
                             int m,// total row count
                             const int *csr_col_ind,
                             int total_nz_element,
                             const void *weight,
                             int num_head,
                             const void *input,
                             int n,// embedding dim
                             int ldb,
                             int ldb_head,
                             int k,// total column count
                             void *output,
                             int ldc,
                             int ldc_head,
                             cudaStream_t stream);

void gSpmmCsrWeightedFusedSharedMemoryBackward(WMType data_type,
                                               bool backward_x, bool backward_w,
                                               const int *csr_row_ptr, int target_count,
                                               const int *csr_col_ind, int edge_count,
                                               const void *edge_weight,
                                               const int *sample_dup_count,
                                               const void *x, int embedding_dim, int num_head,
                                               int ldx, int ldx_head, int neighbor_count,
                                               const void *output_grad,
                                               int ld_grad_out, int ld_grad_out_head,
                                               void *x_grad,
                                               int ld_grad_x, int ld_grad_x_head,
                                               void *edge_weight_grad,
                                               cudaStream_t stream);

void SpAddGATCSRForward(WMType data_type,
                        const int *csr_row_ptr,
                        int target_vec_count,
                        const int *csr_col_ind,
                        const void *edge_weight_left,
                        const void *edge_weight_right,
                        int num_head,
                        void *output, cudaStream_t stream);

void SpAddGATCSRBackward(WMType data_type,
                         const int *csr_row_ptr,
                         const int *csr_col_ind,
                         const int *sample_dup_count,
                         const void *grad_y,
                         int target_vec_count,
                         int neighbor_count,
                         int num_head,
                         void *grad_weight_left,
                         void *grad_weight_right,
                         cudaStream_t stream);

void EdgeWeightSoftmaxForward(WMType data_type,
                              const int *csr_row_ptr,
                              const int64_t target_vec_count,
                              const void *edge_weight,
                              const int num_head,
                              void *edge_weight_softmax,
                              cudaStream_t stream);

void EdgeWeightSoftmaxBackward(WMType data_type,
                               const int *csr_row_ptr,
                               const void *edge_weight_softmax,
                               const void *grad_y,
                               const int64_t target_vec_count,
                               const int num_head,
                               void *grad_weight_softmax,
                               cudaStream_t stream);

void CSRAddSelfLoop(const int *csr_row_ptr,
                    const int *csr_col_ind,
                    const int *sample_dup_count,
                    int total_target_count,
                    int unique_neighbor_and_target_count,
                    int *csr_row_ptr_looped,
                    int *csr_col_ind_looped,
                    int *sample_dup_count_looped,
                    cudaStream_t stream);

void MixedGraphSGC(WMType param_type,
                   WMType id_type,
                   void *param,
                   const int64_t *csr_row_ptr,
                   const void *csr_col_idx,
                   const int64_t *to_typed,
                   int target_type,
                   int neighbor_type,
                   int64_t storage_offset,
                   int64_t embedding_dim,
                   int64_t embedding_stride,
                   int64_t node_count,
                   cudaStream_t stream);

void MixedGraphSGCChunked(WMType param_type,
                          WMType id_type,
                          WholeChunkedMemory_t param,
                          WholeChunkedMemory_t csr_row_ptr,
                          WholeChunkedMemory_t csr_col_idx,
                          WholeChunkedMemory_t to_typed,
                          int target_type,
                          int neighbor_type,
                          int64_t storage_offset,
                          int64_t embedding_dim,
                          int64_t embedding_stride,
                          int64_t node_count,
                          cudaStream_t stream);

void SpmmCsrRelationalNoWeightForward(WMType data_type,
                                      const int *csr_row_ptr,
                                      int m,// total row count
                                      const int *csr_col_ind,
                                      int total_nz_element,
                                      const int8_t *edge_type,
                                      const void *input,
                                      int n,// embedding dim
                                      int ldb,
                                      int k,// total column count
                                      void *output,
                                      int ldc,
                                      int num_relations,
                                      int aggregator,
                                      cudaStream_t stream);

void SpmmCsrRelationalNoWeightBackward(WMType data_type,
                                       const int *csr_row_ptr,
                                       const int *csr_col_ind,
                                       const int8_t *edge_type,
                                       const int *sample_dup_count,
                                       const void *grad_output,
                                       int grad_out_stride,
                                       void *grad_x,
                                       int grad_x_stride,
                                       int m,// total row count,
                                       int total_nz_element,
                                       int k,// total column count
                                       int embedding_dim,
                                       int num_relations,
                                       int aggregator,
                                       cudaStream_t stream);

}// namespace whole_graph