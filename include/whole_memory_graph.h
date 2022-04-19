#pragma once

#include <functional>
#include <string>

#include "data_type.h"
#include "cuda_env_fns.h"

namespace whole_memory {

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
void WmmpFreeEdgeData(void* local_edge_buffer, void* local_feature_buffer);

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
                                      void* local_edge_buffer,
                                      int64_t local_edge_count,
                                      int64_t total_src_node_count,
                                      int64_t total_edge_count,
                                      int64_t* final_total_edge_count,
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
void WmmpUnweightedSampleWithoutReplacement(const std::function<void *(size_t)>& sample_output_allocator,
                                            const std::function<void *(size_t)>& center_localid_allocator,
                                            int* sample_offset,
                                            void *wm_csr_row_ptr,
                                            void *wm_csr_col_ptr,
                                            WMType id_type,
                                            const void *center_nodes,
                                            int center_node_count,
                                            int max_sample_count,
                                            const CUDAEnvFns& cuda_env_fns,
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
void WmmpChunkedUnweightedSampleWithoutReplacement(const std::function<void *(size_t)>& sample_output_allocator,
                                                   const std::function<void *(size_t)>& center_localid_allocator,
                                                   int *sample_offset,
                                                   void *wm_csr_row_ptr,
                                                   void *wm_csr_col_ptr,
                                                   WMType id_type,
                                                   const void *center_nodes,
                                                   int center_node_count,
                                                   int max_sample_count,
                                                   const CUDAEnvFns& cuda_env_fns,
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
void AppendUnique(const void* target,
                  int target_count,
                  const void* neighbor,
                  int neighbor_count,
                  WMType id_type,
                  const std::function<void*(size_t)>& unique_total_output_allocator,
                  const std::function<int32_t*(size_t)>& neighbor_raw_to_unique_mapping_allocator,
                  const std::function<int32_t*(size_t)>& unique_output_neighbor_count_allocator,
                  const CUDAEnvFns& cuda_env_fns,
                  cudaStream_t stream = nullptr);

}