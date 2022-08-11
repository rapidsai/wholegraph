#pragma once

#include <cstdint>
#include <string>

#include "data_type.h"
#include "whole_chunked_memory.h"

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
 * @param stream : cuda stream to use
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
 * @param stream : cuda stream to use
 */
void WmmpLoadChunkedEmbeddingFromMemory(void *wm_embedding,
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
 * Load embedding from part file list to WholeMemory, can copy part by specifying src_start_id and src_table_size
 * @param wm_embedding : pointer to WholeMemory
 * @param storage_offset : storage_offset in terms of emb_type of WholeMemory
 * @param table_size : embedding table size
 * @param embedding_dim : embedding dim of each entry
 * @param embedding_stride : stride of each entry in number of elements
 * @param file_prefix : prefix of part file
 * @param src_type : src embedding data type
 * @param emb_type : WholeMemory embedding data type
 * @param stream : cuda stream to use
 * @param ranks : ranks participate in this aggregation, nullptr means all ranks
 * @param rank_count : rank count participate in this aggregation, 0 means all ranks
 */
void WmmpLoadEmbeddingFromFilelists(void *wm_embedding,
                                    int64_t storage_offset,
                                    int64_t table_size,
                                    int64_t embedding_dim,
                                    int64_t embedding_stride,
                                    const std::string &file_prefix,
                                    WMType src_type,
                                    WMType emb_type,
                                    cudaStream_t stream = nullptr,
                                    const int* ranks = nullptr,
                                    int rank_count = 0);

/*!
 * Load embedding from part file list to chunked WholeMemory, can copy part by specifying src_start_id and src_table_size
 * @param wm_embedding : WholeChunkedMemory_t pointer to chunked WholeMemory
 * @param storage_offset : storage_offset in terms of emb_type of chunked WholeMemory
 * @param table_size : embedding table size
 * @param embedding_dim : embedding dim of each entry
 * @param embedding_stride : stride of each entry in number of elements
 * @param file_prefix : prefix of part file
 * @param src_type : src embedding data type
 * @param emb_type : chunked WholeMemory embedding data type
 * @param stream : cuda stream to use
 * @param ranks : ranks participate in this aggregation, nullptr means all ranks
 * @param rank_count : rank count participate in this aggregation, 0 means all ranks
 */
void WmmpLoadChunkedEmbeddingFromFilelists(void *wm_embedding,
                                           int64_t storage_offset,
                                           int64_t table_size,
                                           int64_t embedding_dim,
                                           int64_t embedding_stride,
                                           const std::string &file_prefix,
                                           WMType src_type,
                                           WMType emb_type,
                                           cudaStream_t stream = nullptr,
                                           const int *ranks = nullptr,
                                           int rank_count = 0);

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
 * @param output_stride : outpupt store stride
 * @param stream : cuda stream to use
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
 * @param output_stride : outpupt store stride
 * @param stream : cuda stream to use
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

}