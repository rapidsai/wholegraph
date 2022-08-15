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

#include <functional>
#include <string>

#include "cuda_env_fns.h"
#include "data_type.h"
#include "whole_chunked_memory.h"

namespace whole_graph {

#if 0
/*!
 * Load edge data into local buffers from part file list. The file should be binary format of many records.
 *     Each record contains src_node_id, dst_node_id and optional edge features.
 *     src_node_id and dst_node_id are type id_type, edge features has length of edge_feature_size.
 * @param plocal_edge_buffer : [output] pointer to local buffer pointer of edges(src and dst node id)
 * @param plocal_feature_buffer : [output] pointer to local buffer pointer of edge features
 * @param edge_count : [output] total edge count
 * @param src_node_count : [inout] src node count, if larger than 0, will check all node_ids less than this,
 *     and the value will not change, if is -1, the max_src_node_id will be write to src_node_count
 * @param dst_node_count : [inout] dst node count, if larger than 0, will check all node_ids less than this,
 *     and the value will not change, if is -1, the max_dst_node_id will be write to dst_node_count
 * @param file_prefix : [input] prefix of part file
 * @param id_type : [input] type of ids
 * @param edge_feature_size : [input] edge feature size for each edge
 * @param ranks : ranks participate in this load, nullptr means all ranks
 * @param rank_count : rank count participate in this load, 0 means all ranks
 * @return local edge count
 */
int64_t WmmpTraverseLoadEdgeDataFromFileList(void **plocal_edge_buffer,
                                             void **plocal_feature_buffer,
                                             int64_t *edge_count,
                                             int64_t *src_node_count,
                                             int64_t *dst_node_count,
                                             const std::string &file_prefix,
                                             WMType id_type,
                                             size_t edge_feature_size,
                                             const int *ranks = nullptr,
                                             int rank_count = 0);

/*!
 * Free edge data loaded by WmmpTraverseLoadEdgeDataFromFileList.
 * @param local_edge_buffer : pointer to local buffer pointer of edges
 * @param local_feature_buffer : pointer to local buffer pointer of edge features
 */
void WmmpFreeEdgeData(void *local_edge_buffer, void *local_feature_buffer);

/*!
 * Estimate edge count for WholeMemory allocation.
 * @param total_src_node_count : total src node count, if reverse edge, this should be dst node count
 * @param total_edge_count : total edge data count
 * @param directed : is directed graph
 * @param add_self_loop : should add self loop to graph
 * @return estimated edge count
 */
inline int64_t EstimateEdgeCount(int64_t total_src_node_count,
                                 int64_t total_edge_count,
                                 bool directed,
                                 bool add_self_loop) {
  int64_t edge_count = total_edge_count;
  if (!directed) edge_count *= 2;
  if (add_self_loop) edge_count += total_src_node_count;
  return edge_count;
}

/*!
 * Load Edge to previous allocated WholeMemory
 * @param wm_csr_row_ptr : allocated csr_row_ptr, int64_t element count should be total_src_node_count + 1
 * @param wm_csr_col_idx : allocated csr_col_idx, id_type element count can be computed by EstimateEdgeCount
 * @param local_edge_buffer : buffer holding local part of edge data
 * @param local_edge_count : local part edge count
 * @param total_src_node_count : total src node count
 * @param total_edge_count : total raw edge data count
 * @param final_total_edge_count : [output] final total edge count, may be directed, self loop added, deduped
 * @param id_type : id type
 * @param directed : is directed graph
 * @param reverse_edge : should reverse edge
 * @param add_self_loop : should add self loop to graph
 * @param stream : cudaStream to use
 * @param ranks : ranks participate in this load, nullptr means all ranks
 * @param rank_count : rank count participate in this load, 0 means all ranks
 */
void WmmpLoadToCSRGraphFromEdgeBuffer(void *wm_csr_row_ptr,
                                      void *wm_csr_col_idx,
                                      void *local_edge_buffer,
                                      int64_t local_edge_count,
                                      int64_t total_src_node_count,
                                      int64_t total_edge_count,
                                      int64_t *final_total_edge_count,
                                      WMType id_type,
                                      bool directed,
                                      bool reverse_edge,
                                      bool add_self_loop,
                                      cudaStream_t stream = nullptr,
                                      const int *ranks = nullptr,
                                      int rank_count = 0);

/*!
 * Load Edge to previous allocated WholeChunkedMemory
 * @param wm_csr_row_ptr : allocated csr_row_ptr, int64_t element count should be total_src_node_count + 1
 * @param wm_csr_col_idx : allocated csr_col_idx, id_type element count can be computed by EstimateEdgeCount
 * @param local_edge_buffer : buffer holding local part of edge data
 * @param local_edge_count : local part edge count
 * @param total_src_node_count : total src node count
 * @param total_edge_count : total raw edge data count
 * @param final_total_edge_count : [output] final total edge count, may be directed, self loop added, deduped
 * @param id_type : id type
 * @param directed : is directed graph
 * @param reverse_edge : should reverse edge
 * @param add_self_loop : should add self loop to graph
 * @param stream : cudaStream to use
 * @param ranks : ranks participate in this load, nullptr means all ranks
 * @param rank_count : rank count participate in this load, 0 means all ranks
 */
void WmmpLoadToChunkedCSRGraphFromEdgeBuffer(void *wm_csr_row_ptr,
                                             void *wm_csr_col_idx,
                                             void *local_edge_buffer,
                                             int64_t local_edge_count,
                                             int64_t total_src_node_count,
                                             int64_t total_edge_count,
                                             int64_t *final_total_edge_count,
                                             WMType id_type,
                                             bool directed,
                                             bool reverse_edge,
                                             bool add_self_loop,
                                             cudaStream_t stream = nullptr,
                                             const int *ranks = nullptr,
                                             int rank_count = 0);
#endif

/*!
 * Stat file element count
 * @param file_prefix : prefix of file
 * @param id_type : element type
 * @return element count
 */
int64_t StatFilelistEltCount(const std::string &file_prefix,
                             WMType id_type);

/*!
 * return Jump COO size
 * @param total_edge_count : total edge count
 * @return Jump COO size
 */
int64_t WmmpGetJumpCOORowSize(int64_t total_edge_count);

/*!
 * Generate Jump COO row Tensor
 * @param wm_jump_coo_row : allocated wm_jump_coo_row, id_type type
 * @param wm_csr_row_ptr : csr_row_ptr, int64_t element count should be total_src_node_count + 1
 * @param total_node_count : total_src_node_count
 * @param total_edge_count : total edge count
 * @param id_type : id type
 * @param bootstrap_communicator : bootstrap communicator
 * @param stream : cudaStream to use
 */
void WmmpGenerateJumpCOORow(void *wm_jump_coo_row,
                            void *wm_csr_row_ptr,
                            int64_t total_node_count,
                            int64_t total_edge_count,
                            WMType id_type,
                            cudaStream_t stream = nullptr);

/*!
 * Generate Chunked Jump COO row Tensor
 * @param wm_jump_coo_row : allocated wm_jump_coo_row, id_type type
 * @param wm_csr_row_ptr : csr_row_ptr, int64_t element count should be total_src_node_count + 1
 * @param total_node_count : total_src_node_count
 * @param total_edge_count : total edge count
 * @param id_type : id type
 * @param bootstrap_communicator : bootstrap communicator
 * @param stream : cudaStream to use
 */
void WmmpGenerateChunkedJumpCOORow(WholeChunkedMemory_t wm_jump_coo_row,
                                   WholeChunkedMemory_t wm_csr_row_ptr,
                                   int64_t total_node_count,
                                   int64_t total_edge_count,
                                   WMType id_type,
                                   cudaStream_t stream = nullptr);

/*!
 * Get src and dst node from Edge id
 * @param wm_csr_row_ptr : csr_row_ptr, int64_t element count should be total_src_node_count + 1
 * @param wm_csr_col_idx : csr_col_idx, id_type element count
 * @param wm_jump_coo_row : Jump COO row
 * @param edge_idx_list : edge id list, int64_t type
 * @param id_type : id type
 * @param total_src_node_count : total src node count
 * @param total_edge_count : total edge count
 * @param src_ptr : output src node id pointer
 * @param dst_ptr : output dst node id pointer
 * @param edge_list_count : edge list count
 * @param stream : CUDA stream to use
 */
void WmmpGetEdgeNodesFromEid(void *wm_csr_row_ptr,
                             void *wm_csr_col_idx,
                             void *wm_jump_coo_row,
                             const int64_t *edge_idx_list,
                             WMType id_type,
                             int64_t total_src_node_count,
                             int64_t total_edge_count,
                             void *src_ptr,
                             void *dst_ptr,
                             int64_t edge_list_count,
                             cudaStream_t stream);

/*!
 * Get src and dst node from Edge id
 * @param wm_csr_row_ptr : Chunked csr_row_ptr, int64_t element count should be total_src_node_count + 1
 * @param wm_csr_col_idx : Chunked csr_col_idx, id_type element count
 * @param wm_jump_coo_row : Chunked Jump COO row
 * @param edge_idx_list : edge id list, int64_t type
 * @param id_type : id type
 * @param total_src_node_count : total src node count
 * @param total_edge_count : total edge count
 * @param src_ptr : output src node id pointer
 * @param dst_ptr : output dst node id pointer
 * @param edge_list_count : edge list count
 * @param stream : CUDA stream to use
 */
void WmmpGetEdgeNodesFromEidChunked(WholeChunkedMemory_t wm_csr_row_ptr,
                                    WholeChunkedMemory_t wm_csr_col_idx,
                                    WholeChunkedMemory_t wm_jump_coo_row,
                                    const int64_t *edge_idx_list,
                                    WMType id_type,
                                    int64_t total_src_node_count,
                                    int64_t total_edge_count,
                                    void *src_ptr,
                                    void *dst_ptr,
                                    int64_t edge_list_count,
                                    cudaStream_t stream);

/*!
 * Unweighted sample without replacement kernel on WholeMemory
 * @param sample_output_allocator : allocator for sample_output, function argument is element_count
 * @param center_localid_allocator : allocator for center_localid, function argument is element_count
 * @param sample_offset : memory pointer for output sample offset of each center node
 * @param wm_csr_row_ptr : allocated csr_row_ptr, int64_t
 * @param wm_csr_col_ptr : allocated csr_row_ptr, id_type
 * @param id_type : type for nodeID
 * @param center_nodes : center node list to sample
 * @param center_node_count : center node count
 * @param max_sample_count : maximum sample count
 * @param cuda_env_fns : CUDA environment function struct
 * @param stream : CUDA stream to use
 */
void WmmpUnweightedSampleWithoutReplacement(const std::function<void *(size_t)> &sample_output_allocator,
                                            const std::function<void *(size_t)> &center_localid_allocator,
                                            int *sample_offset,
                                            void *wm_csr_row_ptr,
                                            void *wm_csr_col_ptr,
                                            WMType id_type,
                                            const void *center_nodes,
                                            int center_node_count,
                                            int max_sample_count,
                                            const CUDAEnvFns &cuda_env_fns,
                                            cudaStream_t stream);

/*!
 * Unweighted sample without replacement kernel on WholeChunkedMemory
 * @param sample_output_allocator : allocator for sample_output, function argument is element_count
 * @param center_localid_allocator : allocator for center_localid, function argument is element_count
 * @param sample_offset : memory pointer for output sample offset of each center node
 * @param wm_csr_row_ptr : allocated csr_row_ptr, int64_t
 * @param wm_csr_col_ptr : allocated csr_row_ptr, id_type
 * @param id_type : type for nodeID
 * @param center_nodes : center node list to sample
 * @param center_node_count : center node count
 * @param max_sample_count : maximum sample count
 * @param cuda_env_fns : CUDA environment function struct
 * @param stream : CUDA stream to use
 */
void WmmpChunkedUnweightedSampleWithoutReplacement(const std::function<void *(size_t)> &sample_output_allocator,
                                                   const std::function<void *(size_t)> &center_localid_allocator,
                                                   int *sample_offset,
                                                   void *wm_csr_row_ptr,
                                                   void *wm_csr_col_ptr,
                                                   WMType id_type,
                                                   const void *center_nodes,
                                                   int center_node_count,
                                                   int max_sample_count,
                                                   const CUDAEnvFns &cuda_env_fns,
                                                   cudaStream_t stream = nullptr);

/*!
 * AppendUnique function, append neighbor to target and then do unique, keeping targets first.
 * @param target : target ids
 * @param target_count : target count
 * @param neighbor : neighbor ids
 * @param id_type : target and neighbor id type
 * @param neighbor_count : neighbor count
 * @param unique_total_output_allocator : allocator for unique_total_output
 * @param neighbor_raw_to_unique_mapping_allocator : allocator for neighbor_raw_to_unique_mapping
 * @param unique_output_neighbor_count_allocator : allocator for unique_output_neighbor_count
 * @param cuda_env_fns : CUDA environment function struct
 * @param stream : CUDA stream to use
 */
void AppendUnique(const void *target,
                  int target_count,
                  const void *neighbor,
                  int neighbor_count,
                  WMType id_type,
                  const std::function<void *(size_t)> &unique_total_output_allocator,
                  const std::function<int32_t *(size_t)> &neighbor_raw_to_unique_mapping_allocator,
                  const std::function<int32_t *(size_t)> &unique_output_neighbor_count_allocator,
                  const CUDAEnvFns &cuda_env_fns,
                  cudaStream_t stream = nullptr);

/*!
 * Get the element count for hashset, used for memory allocation.
 * @param edge_count : edge count to insert into hashset.
 * @return element count for hashset
 */
int64_t GetEdgeHashSetEltCount(int edge_count);

/*!
 * Create and insert edges into GPU hashset.
 * @param id_type : id type of edges, should be int32 or int64
 * @param src_ids : src_ids for edges to insert into hashset
 * @param dst_ids : dst_ids for edges to insert into hashset
 * @param edge_count : edge count
 * @param hash_set_mem : memory allocated for the hashset, should have at least GetEdgeHashSetEltCount elements.
 * @param hash_set_elt_count : element count should get from GetEdgeHashSetEltCount
 * @param stream : CUDA stream to use
 */
void CreateEdgeHashSet(WMType id_type,
                       const void *src_ids,
                       const void *dst_ids,
                       int edge_count,
                       void *hash_set_mem,
                       int hash_set_elt_count,
                       cudaStream_t stream = nullptr);

/*!
 * Retrieve if Edges are in the hashset
 * @param id_type : id type of edges, should be int32 or int64
 * @param src_ids : src_ids for edges to retrieve in hashset
 * @param dst_ids : dst_ids for edges to retrieve in hashset
 * @param edge_count : edge count to retrieve
 * @param hash_set_mem : memory storing hash values.
 * @param hash_set_elt_count : element count in hashset
 * @param output : output flags, int type, 1 for exist, 0 for not exist
 * @param stream : CUDA stream to use.
 */
void RetrieveCOOEdges(WMType id_type,
                      const void *src_ids,
                      const void *dst_ids,
                      int edge_count,
                      void *hash_set_mem,
                      int hash_set_elt_count,
                      int *output,
                      cudaStream_t stream = nullptr);

void FilterCSREdges(WMType id_type,
                    const void *src_ids,
                    int node_count,
                    const int *gids_offset,
                    void *dst_ids_vdata,
                    int edge_count,
                    void *hash_set_mem_tensor,
                    int hash_memory_elt_count,
                    int *new_gids_offset,
                    const std::function<void *(size_t)> &new_dst_ids_allocator,
                    const std::function<int *(size_t)> &new_src_lids_allocator,
                    const CUDAEnvFns &env_fns,
                    cudaStream_t stream = nullptr);

/*!
 * WmmpPerNodeUniformNegativeSample on WholeMemory
 * @param sample_output_allocator : allocator for sample_output, function argument is element_count
 * @param wm_csr_row_ptr : allocated csr_row_ptr, int64_t
 * @param wm_csr_col_ptr : allocated csr_row_ptr, id_type
 * @param id_type : type for nodeID
 * @param target_nodes : target node list to sample
 * @param target_node_count : target node count
 * @param graph_dst_node_count: graph dst node count
 * @param negative_sample_count : negative sample count per target node
 * @param cuda_env_fns : CUDA environment function struct
 * @param stream : CUDA stream to use
 */
void WmmpPerNodeUniformNegativeSample(const std::function<void *(size_t)> &sample_output_allocator,
                                      void *wm_csr_row_ptr,
                                      void *wm_csr_col_ptr,
                                      WMType id_type,
                                      const void *target_nodes,
                                      int target_node_count,
                                      int graph_dst_node_count,
                                      int negative_sample_count,
                                      const CUDAEnvFns &cuda_env_fns,
                                      cudaStream_t stream = nullptr);

/*!
 * WmmpChunkedPerNodeUniformNegativeSample on WholeChunkedMemory
 * @param sample_output_allocator : allocator for sample_output, function argument is element_count
 * @param wm_csr_row_ptr : allocated csr_row_ptr, int64_t
 * @param wm_csr_col_ptr : allocated csr_row_ptr, id_type
 * @param id_type : type for nodeID
 * @param target_nodes : target node list to sample
 * @param target_node_count : target node count
 * @param graph_dst_node_count: graph dst node count
 * @param negative_sample_count : negative sample count per target node
 * @param cuda_env_fns : CUDA environment function struct
 * @param stream : CUDA stream to use
 */
void WmmpChunkedPerNodeUniformNegativeSample(const std::function<void *(size_t)> &sample_output_allocator,
                                             void *wm_csr_row_ptr,
                                             void *wm_csr_col_ptr,
                                             WMType id_type,
                                             const void *target_nodes,
                                             int target_node_count,
                                             int graph_dst_node_count,
                                             int negative_sample_count,
                                             const CUDAEnvFns &cuda_env_fns,
                                             cudaStream_t stream = nullptr);

void WmmpGetCSRMixedSubGraphEdgeTypes(WMType mixid_type,
                                      int8_t *output,
                                      const void *src_mixid,
                                      const int *sub_graph_csr_row_ptr,
                                      const void *sub_graph_csr_col_mixid,
                                      const int8_t *edge_type_dict,
                                      const int64_t *to_typed_id,
                                      int64_t src_node_count,
                                      int64_t dst_node_count,
                                      int64_t node_type_count,
                                      int64_t edge_type_count,
                                      cudaStream_t stream);

void WmmpGetCSRMixedSubGraphEdgeTypesChunked(WMType mixid_type,
                                             int8_t *output,
                                             const void *src_mixid,
                                             const int *sub_graph_csr_row_ptr,
                                             const void *sub_graph_csr_col_mixid,
                                             const int8_t *edge_type_dict,
                                             WholeChunkedMemory_t to_typed_id_wcmt,
                                             int64_t src_node_count,
                                             int64_t dst_node_count,
                                             int64_t node_type_count,
                                             int64_t edge_type_count,
                                             cudaStream_t stream);

void WmmpGetBucketedCSRFromSortedTypedIDs(int *output_csr,
                                          const int64_t *typed_ids,
                                          int64_t id_count,
                                          int64_t node_type_count,
                                          cudaStream_t stream);

void WmmpPackToTypedIDs(int64_t *output,
                        const int8_t *type_ids,
                        const int *ids,
                        int64_t id_count,
                        cudaStream_t stream);

void WmmpUnpackTypedIDs(const int64_t *typed_ids,
                        int8_t *type_ids,
                        int *ids,
                        int64_t id_count,
                        cudaStream_t stream);

void WmmpWeightedSampleWithoutReplacement(const std::function<void *(size_t)> &sample_output_allocator,
                                          const std::function<void *(size_t)> &center_localid_allocator,
                                          int *sample_offset,
                                          void *wm_csr_row_ptr,
                                          void *wm_csr_col_ptr,
                                          void *wm_csr_weight_ptr,
                                          void *wm_csr_local_sorted_map_indices_ptr,
                                          WMType id_type,
                                          WMType weight_type,
                                          const void *center_nodes,
                                          int center_node_count,
                                          int max_sample_count,
                                          const CUDAEnvFns &cuda_env_fns,
                                          cudaStream_t stream);

void WmmpChunkedWeightedSampleWithoutReplacement(const std::function<void *(size_t)> &sample_output_allocator,
                                                 const std::function<void *(size_t)> &center_localid_allocator,
                                                 int *sample_offset,
                                                 void *wm_csr_row_ptr,
                                                 void *wm_csr_col_ptr,
                                                 void *wm_csr_weight_ptr,
                                                 void *wm_csr_local_sorted_map_indices_ptr,
                                                 WMType id_type,
                                                 WMType weight_type,
                                                 const void *center_nodes,
                                                 int center_node_count,
                                                 int max_sample_count,
                                                 const CUDAEnvFns &cuda_env_fns,
                                                 cudaStream_t stream);

void WmmpExtractSubGraphWithFilter(WMType id_type,
                                   WMType edge_value_type,
                                   int extract_type,
                                   bool need_value_output,
                                   int64_t target_count,
                                   int64_t total_node_count,
                                   int64_t *subgraph_row_ptr,
                                   const std::function<void *(size_t)> &subgraph_col_allocator,
                                   const std::function<void *(size_t)> &subgraph_edge_value_allocator,
                                   const void *target_gid_ptr,
                                   const void *filter_target_value_ptr,
                                   const int64_t *edges_csr_row,
                                   const void *edges_csr_col,
                                   const void *edges_value,
                                   const CUDAEnvFns &cuda_env_fns,
                                   cudaStream_t stream);

void WmmpExtractSubGraphWithFilterChunked(WMType id_type,
                                          WMType edge_value_type,
                                          int extract_type,
                                          bool need_value_output,
                                          int64_t target_count,
                                          int64_t total_node_count,
                                          int64_t *subgraph_row_ptr,
                                          const std::function<void *(size_t)> &subgraph_col_allocator,
                                          const std::function<void *(size_t)> &subgraph_edge_value_allocator,
                                          const void *target_gid_ptr,
                                          const void *filter_target_value_ptr,
                                          WholeChunkedMemory_t edges_csr_row,
                                          WholeChunkedMemory_t edges_csr_col,
                                          WholeChunkedMemory_t edges_value,
                                          const CUDAEnvFns &cuda_env_fns,
                                          cudaStream_t stream);

}// namespace whole_graph