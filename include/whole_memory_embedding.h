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

#include <cstdint>
#include <string>

#include "cuda_env_fns.h"
#include "data_type.h"
#include "whole_chunked_memory.h"
#include "whole_graph_optimizers.h"
#include "whole_memory.h"
#include "whole_nccl_memory.h"

namespace whole_graph {

/*!
 * Load embedding from host memory to WholeMemory, can copy part by specifying src_start_id and src_table_size
 * @param wm_embedding : pointer to WholeMemory
 * @param storage_offset : storage_offset in terms of emb_type of WholeMemory
 * @param table_size : embedding table size
 * @param embedding_dim : embedding dim of each entry
 * @param embedding_stride : stride of each entry in number of elements
 * @param src : src embedding host memory pointer
 * @param src_stride : src embedding stride in number of elements
 * @param src_type : src embedding data type
 * @param emb_type : WholeMemory embedding data type
 * @param src_start_id : start_id of src embedding
 * @param src_table_size : entry count of src embedding
 * @param stream : CUDA stream to use
 */
void WmmpLoadEmbeddingFromMemory(void *wm_embedding,
                                 int64_t storage_offset,
                                 int64_t table_size,
                                 int64_t embedding_dim,
                                 int64_t embedding_stride,
                                 const void *src,
                                 int64_t src_stride,
                                 WMType src_type,
                                 WMType emb_type,
                                 int64_t src_start_id,
                                 int64_t src_table_size,
                                 cudaStream_t stream = nullptr);

/*!
 * Load embedding from host memory to chunked WholeMemory, can copy part by specifying src_start_id and src_table_size
 * @param wm_embedding : WholeChunkedMemory_t pointer to WholeChunkedMemory
 * @param storage_offset : storage_offset in terms of emb_type of WholeChunkedMemory
 * @param table_size : embedding table size
 * @param embedding_dim : embedding dim of each entry
 * @param embedding_stride : stride of each entry in number of elements
 * @param src : src embedding host memory pointer
 * @param src_stride : src embedding stride in number of elements
 * @param src_type : src embedding data type
 * @param emb_type : chunked WholeMemory embedding data type
 * @param src_start_id : start_id of src embedding
 * @param src_table_size : entry count of src embedding
 * @param stream : CUDA stream to use
 */
void WmmpLoadChunkedEmbeddingFromMemory(WholeChunkedMemory_t wm_embedding,
                                        int64_t storage_offset,
                                        int64_t table_size,
                                        int64_t embedding_dim,
                                        int64_t embedding_stride,
                                        const void *src,
                                        int64_t src_stride,
                                        WMType src_type,
                                        WMType emb_type,
                                        int64_t src_start_id,
                                        int64_t src_table_size,
                                        cudaStream_t stream = nullptr);

/*!
 * Store local embedding to part file, should call this function together
 * @param emb_type : embedding type
 * @param emb_ptr : pointer to the local_embedding
 * @param embedding_count : local embedding table entry count
 * @param embedding_dim : embedding dimension
 * @param embedding_stride : embedding stride
 * @param filename : file name to store
 */
void WmmpStoreLocalEmbeddingToFile(WMType emb_type,
                                   const void *emb_ptr,
                                   int64_t embedding_count,
                                   int64_t embedding_dim,
                                   int64_t embedding_stride,
                                   const std::string &filename);

/*!
 * Load local embedding from part file, should call this function together
 * @param emb_type : embedding type
 * @param emb_ptr : pointer to the local_embedding
 * @param embedding_count : local embedding table entry count
 * @param embedding_dim : embedding dimension
 * @param embedding_stride : embedding stride
 * @param file_prefix : file prefix to load from
 * @param part_count : file count
 */
void WmmpLoadLocalEmbeddingFromFile(WMType emb_type,
                                    void *emb_ptr,
                                    int64_t embedding_count,
                                    int64_t embedding_dim,
                                    int64_t embedding_stride,
                                    const std::string &file_prefix,
                                    int part_count,
                                    BootstrapCommunicator *bootstrap_communicator);

/*!
 * Gather from WholeMemory
 * @param output_t : output data type
 * @param param_t : WholeMemory parameter data type
 * @param index_t : indice data type
 * @param output : output pointer
 * @param parameter : WholeMemory pointer
 * @param indice : indice pointer
 * @param storage_offset : storage offset in terms of param_t of WholeMemory
 * @param indice_count : indice count to gather
 * @param embedding_dim : embedding dim of WholeMemory
 * @param embedding_stride : embedding_stride of WholeMemory
 * @param output_stride : output store stride
 * @param stream : CUDA stream to use
 */
void WholeMemoryGather(WMType output_t,
                       WMType param_t,
                       WMType index_t,
                       void *output,
                       const void *parameter,
                       const void *indice,
                       size_t storage_offset,
                       int64_t indice_count,
                       int64_t embedding_dim,
                       int64_t embedding_stride,
                       int64_t output_stride,
                       cudaStream_t stream = nullptr);

/*!
 * Gather from chunked WholeMemory
 * @param output_t : output data type
 * @param param_t : chunked WholeMemory parameter data type
 * @param index_t : indice data type
 * @param output : output pointer
 * @param parameter : chunked WholeMemory pointer
 * @param indice : indice pointer
 * @param storage_offset : storage offset in terms of param_t of chunked WholeMemory
 * @param indice_count : indice count to gather
 * @param embedding_dim : embedding dim of chunked WholeMemory
 * @param embedding_stride : embedding_stride of chunked WholeMemory
 * @param output_stride : output store stride
 * @param stream : CUDA stream to use
 */
void WholeMemoryChunkedGather(WMType output_t,
                              WMType param_t,
                              WMType index_t,
                              void *output,
                              const void *parameter,
                              const void *indice,
                              size_t storage_offset,
                              int64_t indice_count,
                              int64_t embedding_dim,
                              int64_t embedding_stride,
                              int64_t output_stride,
                              cudaStream_t stream = nullptr);

/*!
 * Gather from NCCL WholeMemory
 * @param param_t : NCCL WholeMemory parameter data type
 * @param index_t : indice data type
 * @param output : output pointer
 * @param parameter_wnmt : NCCL WholeMemory pointer
 * @param indice : indice pointer
 * @param storage_offset : storage offset in terms of param_t of NCCL WholeMemory
 * @param indice_count : indice count to gather
 * @param embedding_dim : embedding dim of NCCL WholeMemory
 * @param embedding_stride : embedding_stride of NCCL WholeMemory
 * @param output_stride : output store stride
 * @param cuda_env_fns : CUDA environment functions
 * @param stream : CUDA stream to use
 */
void WholeMemoryNCCLGather(WMType param_t,
                           WMType index_t,
                           void *output,
                           whole_graph::WholeNCCLMemory_t parameter_wnmt,
                           const void *indice,
                           size_t storage_offset,
                           int64_t indice_count,
                           int64_t embedding_dim,
                           int64_t embedding_stride,
                           int64_t output_stride,
                           CUDAEnvFns cuda_env_fns,
                           cudaStream_t stream = nullptr);

/*!
 * Scatter to NCCL WholeMemory
 * @param param_t : NCCL WholeMemory parameter data type
 * @param index_t : indice data type
 * @param input : input pointer
 * @param parameter_wnmt : NCCL WholeMemory pointer
 * @param indice : indice pointer
 * @param storage_offset : storage offset in terms of param_t of NCCL WholeMemory
 * @param indice_count : indice count to gather
 * @param embedding_dim : embedding dim of NCCL WholeMemory
 * @param embedding_stride : embedding_stride of NCCL WholeMemory
 * @param input_stride : input stride
 * @param cuda_env_fns : CUDA environment functions
 * @param stream : CUDA stream to use
 */
void WholeMemoryNCCLScatter(WMType param_t,
                            WMType index_t,
                            const void *input,
                            whole_graph::WholeNCCLMemory_t parameter_wnmt,
                            const void *indice,
                            size_t storage_offset,
                            int64_t indice_count,
                            int64_t embedding_dim,
                            int64_t embedding_stride,
                            int64_t input_stride,
                            const CUDAEnvFns &cuda_env_fns,
                            cudaStream_t stream);

/*!
 * Exchange embedding gradients
 * @param index_t : indice data type
 * @param grad_t : gradient data type
 * @param local_indice_allocator : allocator for local indice
 * @param local_grad_allocator : allocator for local gradients
 * @param sparse_indices : pointer to indices
 * @param sparse_grads : pointer to gradients
 * @param indice_count : indice count of sparse_indices
 * @param embedding_dim : embedding dim for sparse_grads
 * @param embedding_stride : embedding stride for sparse_grads
 * @param total_entry_count : total entry of the embedding table, embedding table not passed in
 * @param bootstrap_communicator : bootstrap_communicator of the embedding table
 * @param cuda_env_fns : CUDA environment functions
 * @param stream : CUDA stream to use
 */
void WholeMemoryExchangeEmbeddingGrads(WMType index_t,
                                       WMType grad_t,
                                       std::function<void *(size_t)> local_indice_allocator,
                                       std::function<void *(size_t)> local_grad_allocator,
                                       const void *sparse_indices,
                                       const void *sparse_grads,
                                       int64_t indice_count,
                                       int64_t embedding_dim,
                                       int64_t embedding_stride,
                                       int64_t total_entry_count,
                                       BootstrapCommunicator *bootstrap_communicator,
                                       const CUDAEnvFns &cuda_env_fns,
                                       cudaStream_t stream);

void WholeMemoryScatter(WMType param_t,
                        WMType index_t,
                        const void *input,
                        void *parameter,
                        const void *indice,
                        size_t storage_offset,
                        int64_t indice_count,
                        int64_t embedding_dim,
                        int64_t embedding_stride,
                        int64_t input_stride,
                        cudaStream_t stream);

void WholeMemoryChunkedScatter(WMType param_t,
                               WMType index_t,
                               const void *input,
                               void *parameter,
                               const void *indice,
                               size_t storage_offset,
                               int64_t indice_count,
                               int64_t embedding_dim,
                               int64_t embedding_stride,
                               int64_t input_stride,
                               cudaStream_t stream);

void WholeMemoryEmbeddingLocalApplyGradients(WMType grad_t,
                                             WMType state_t,
                                             OptimizerInfo opt_info,
                                             const int *update_list,
                                             void *embedding,
                                             void *grad,
                                             void *per_element_state_0,
                                             void *per_element_state_1,
                                             void *per_embedding_state,
                                             int64_t update_count,
                                             int64_t embedding_table_size,
                                             int64_t embedding_dim,
                                             int64_t embedding_stride,
                                             int64_t grad_per_element_state_stride,
                                             cudaStream_t stream);

}// namespace whole_graph