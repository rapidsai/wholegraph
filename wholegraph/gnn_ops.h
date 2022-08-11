#pragma once

#include "data_type.h"

namespace whole_graph {

void SpmmCsrNoWeightForward(WMType data_type,
                            const int *csr_row_ptr,
                            int m,  // total row count
                            const int *csr_col_ind,
                            int total_nz_element,
                            const void *input,
                            int n,  // embedding dim
                            int ldb,
                            int k,  // total column count
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
                             int m, // total row count,
                             int total_nz_element,
                             int k,  // total column count
                             int embedding_dim,
                             int aggregator,
                             cudaStream_t stream);

void gSpmmCsrWeightedForward(WMType data_type,
                             const int *csr_row_ptr,
                             int m,  // total row count
                             const int *csr_col_ind,
                             int total_nz_element,
                             const void *weight,
                             int num_head,
                             const void *input,
                             int n,  // embedding dim
                             int ldb,
                             int ldb_head,
                             int k,  // total column count
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
}