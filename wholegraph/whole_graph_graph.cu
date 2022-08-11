#include "whole_graph_graph.h"

#include <string>
#include <unordered_set>

#include "whole_graph.h"
#include "whole_chunked_memory.h"
#include "data_type.h"
#include "file_utils.h"
#include "whole_graph_communicator.h"
#include "parallel_utils.h"
#include "cuda_env_fns.h"
#include "macros.h"

namespace whole_graph {

template <typename IdType>
void FindMaxNodeID(int64_t* src_max_id, int64_t* dst_max_id, const void* local_edge_buffer, int64_t local_edge_count, const int* ranks, int rank_count) {
  int64_t src_max_node_id = 0;
  int64_t dst_max_node_id = 0;
  const auto* local_edges = (const IdType*)local_edge_buffer;
  for (int64_t i = 0; i < local_edge_count; i++) {
    src_max_node_id = std::max<int64_t>(src_max_node_id, local_edges[i * 2]);
    dst_max_node_id = std::max<int64_t>(dst_max_node_id, local_edges[i * 2 + 1]);
  }
  std::vector<int64_t> all_src_max, all_dst_max;
  CollAllGather(src_max_node_id, &all_src_max, ranks, rank_count);
  CollAllGather(dst_max_node_id, &all_dst_max, ranks, rank_count);
  for (size_t i = 0; i < all_src_max.size(); i++) {
    src_max_node_id = std::max<int64_t>(src_max_node_id, all_src_max[i]);
    dst_max_node_id = std::max<int64_t>(dst_max_node_id, all_dst_max[i]);
  }
  *src_max_id = src_max_node_id;
  *dst_max_id = dst_max_node_id;
}

REGISTER_DISPATCH_ONE_TYPE(FindMaxNodeID, FindMaxNodeID, SINT)

int64_t WmmpTraverseLoadEdgeDataFromFileList(void **plocal_edge_buffer,
                                             void **plocal_feature_buffer,
                                             int64_t *edge_count,
                                             int64_t *src_node_count,
                                             int64_t *dst_node_count,
                                             const std::string &file_prefix,
                                             WMType id_type,
                                             size_t edge_feature_size,
                                             const int *ranks,
                                             int rank_count) {
  assert(id_type == WMT_Int32 || id_type == WMT_Int64);
  size_t single_edge_size_bytes = 2 * GetWMTSize(id_type) + edge_feature_size;
  std::vector<std::string> filelist;
  if (!GetPartFileListFromPrefix(file_prefix, &filelist)) {
    std::cerr << "[WmmpLoadToCSRGraphTraverseEdgeFileList] GetPartFileListFromPrefix from prefix " << file_prefix << " failed.\n";
    abort();
  }
  const int64_t magic_num = 0x10aded9ef70e1157LL;
  CollCheckAllSame<int64_t>(magic_num, ranks, rank_count);
  CollCheckAllSame(filelist.size(), ranks, rank_count);
  std::vector<int64_t> file_start_edge_ids, edge_counts;
  int64_t start_edge_id = 0;
  for (const auto& filename : filelist) {
    size_t file_size = StatFileSize(filename);
    if (file_size % single_edge_size_bytes != 0) {
      std::cerr << "File " << filename << " size is " << file_size << ", but type " << GetWMTName(id_type)
                << ", edge_feature_size=" << edge_feature_size << ", single_edge_size_bytes=" << single_edge_size_bytes << std::endl;
      abort();
    }
    int64_t file_edge_count = file_size / single_edge_size_bytes;
    CollCheckAllSame(file_edge_count, ranks, rank_count);
    file_start_edge_ids.push_back(start_edge_id);
    start_edge_id += file_edge_count;
    edge_counts.push_back(file_edge_count);
  }
  *edge_count = start_edge_id;
  int64_t local_edge_count, local_edge_start;
  local_edge_start = WmmpCollOffsetAndSize(*edge_count, reinterpret_cast<size_t *>(&local_edge_count), ranks, rank_count);
  const int64_t kMemoryBlockSize = 64 * 1024 * 1024;
  int64_t max_edge_count = kMemoryBlockSize / single_edge_size_bytes;
  auto *edge_load_buffer = (char*)malloc(kMemoryBlockSize);
  auto *local_edge_buffer = (int64_t*)malloc(2 * GetWMTSize(id_type) * local_edge_count);
  char *local_feature_buffer = nullptr;
  if (edge_feature_size > 0) local_feature_buffer = (char*)malloc(edge_feature_size * local_edge_count);
  for (int fidx = 0; fidx < (int)filelist.size(); fidx++) {
    int64_t file_start_edge_id = file_start_edge_ids[fidx];
    int64_t file_edge_count = edge_counts[fidx];
    if (file_start_edge_id + file_edge_count < local_edge_start) continue;
    if (file_start_edge_id >= local_edge_start + local_edge_count) break;
    FILE* fp = fopen(filelist[fidx].c_str(), "rb");
    if (fp == nullptr) {
      std::cerr << "Open file " << filelist[fidx] << " failed.\n";
      abort();
    }
    printf("Rank=%d, loading from file %s\n", CCRank(), filelist[fidx].c_str());
    int64_t edge_offset_in_file = 0;
    if (file_start_edge_id < local_edge_start) {
      edge_offset_in_file += local_edge_start - file_start_edge_id;
      assert(fseeko64(fp, edge_offset_in_file * single_edge_size_bytes, SEEK_SET) == 0);
    }
    while (edge_offset_in_file < file_edge_count && edge_offset_in_file + file_start_edge_id < local_edge_start + local_edge_count) {
      int64_t read_edge_count = std::min(file_edge_count - edge_offset_in_file, local_edge_start + local_edge_count - file_start_edge_id - edge_offset_in_file);
      if (read_edge_count > max_edge_count) read_edge_count = max_edge_count;
      assert(fread(edge_load_buffer, single_edge_size_bytes, read_edge_count, fp) == read_edge_count);
      int64_t start_id = file_start_edge_ids[fidx] + edge_offset_in_file;
      int64_t local_start_id = start_id - local_edge_start;
      if (id_type == WMT_Int32) {
        for (int read_id = 0; read_id < read_edge_count; read_id++) {
          local_edge_buffer[local_start_id + read_id] = *(int64_t*)(edge_load_buffer + single_edge_size_bytes * read_id);
          char* cpsrc = edge_load_buffer + single_edge_size_bytes * read_id + sizeof(int64_t);
          char* cpdst = local_feature_buffer + edge_feature_size * (read_id + local_start_id);
          if (edge_feature_size > 0) memcpy(cpdst, cpsrc, edge_feature_size);
        }
      } else if (id_type == WMT_Int64) {
        for (int read_id = 0; read_id < read_edge_count; read_id++) {
          local_edge_buffer[(local_start_id + read_id) * 2] = *(int64_t*)(edge_load_buffer + single_edge_size_bytes * read_id);
          local_edge_buffer[(local_start_id + read_id) * 2 + 1] = *(int64_t*)(edge_load_buffer + single_edge_size_bytes * read_id + sizeof(int64_t));
          char* cpsrc = edge_load_buffer + single_edge_size_bytes * read_id + sizeof(int64_t) * 2;
          char* cpdst = local_feature_buffer + edge_feature_size * (read_id + local_start_id);
          if (edge_feature_size > 0) memcpy(cpdst, cpsrc, edge_feature_size);
        }
      }
      edge_offset_in_file += read_edge_count;
    }
    fclose(fp);
  }
  free(edge_load_buffer);

  printf("Rank=%d, load completed\n", CCRank());
  int64_t max_src_id, max_dst_id;
  DISPATCH_ONE_TYPE(id_type, FindMaxNodeID, &max_src_id, &max_dst_id, local_edge_buffer, local_edge_count, ranks, rank_count);
  //printf("Rank=%d, max_src_id=%ld, max_dst_id=%ld\n", CCRank(), max_src_id, max_dst_id);
  if (*src_node_count == -1) {
    *src_node_count = max_src_id + 1;
  } else if (*src_node_count <= max_src_id) {
    std::cerr << "src_node_count is " << *src_node_count << " but get node id " << max_src_id << "\n";
  }
  if (*dst_node_count == -1) {
    *dst_node_count = max_dst_id + 1;
  } else if (*dst_node_count <= max_dst_id) {
    std::cerr << "dst_node_count is " << *dst_node_count << " but get node id " << max_dst_id << "\n";
  }

  *plocal_edge_buffer = local_edge_buffer;
  *plocal_feature_buffer = local_feature_buffer;
  return local_edge_count;
}

void WmmpFreeEdgeData(void* local_edge_buffer, void* local_feature_buffer) {
  if (local_edge_buffer) free(local_edge_buffer);
  if (local_feature_buffer) free(local_feature_buffer);
}
template <typename IdType>
struct EdgePair {
  inline IdType& operator[](int idx) {
    return data[idx];
  }
  IdType data[2];
};
template <typename IdType>
void ExchangeEdges(std::vector<std::vector<IdType>>& local_edges_2d, int64_t local_node_start, int64_t local_node_count, const int *ranks, int rank_count) {
  int64_t total_src_node_count = local_edges_2d.size();
  int group_rank = GetRankOffsetInRanks(CCRank(), ranks, rank_count);
  int group_size = GetSizeInRanks(ranks, rank_count);
  const int64_t kMaxEdgeCountPerBatch = 4 * 1024 * 1024;
  size_t host_mem_size = kMaxEdgeCountPerBatch * sizeof(EdgePair<IdType>) * group_size * group_size;
  EdgePair<IdType>* shared_edge_buffer;
  WmmpMallocHost((void**)&shared_edge_buffer, host_mem_size, ranks, rank_count);
  std::vector<EdgePair<IdType>*> send_edge_buffer(group_size), recv_edge_buffer(group_size);
  for (int i = 0; i < group_size; i++) {
    send_edge_buffer[i] = shared_edge_buffer + (group_rank * group_size + i) * kMaxEdgeCountPerBatch;
    recv_edge_buffer[i] = shared_edge_buffer + (group_rank + i * group_size) * kMaxEdgeCountPerBatch;
  }
  std::vector<int64_t> all_rank_node_start;
  CollAllGather(local_node_start, &all_rank_node_start, ranks, rank_count);
  all_rank_node_start.push_back(total_src_node_count);
  std::vector<int64_t> local_seg_edge_count(group_size, 0);
  int64_t max_send_count = 0;
  for (int i = 0; i < group_size; i++) {
    int64_t start = all_rank_node_start[i];
    int64_t end = all_rank_node_start[i + 1];
    int64_t seg_edge_count = 0;
    for (auto nid = (IdType)start; nid < (IdType)end; nid++) {
      seg_edge_count += local_edges_2d[nid].size();
    }
    local_seg_edge_count[i] = seg_edge_count;
    int64_t send_count = (seg_edge_count - kMaxEdgeCountPerBatch + 1) / kMaxEdgeCountPerBatch;
    max_send_count = std::max(max_send_count, send_count);
  }
  std::vector<int64_t> all_max_send_count;
  CollAllGather(max_send_count, &all_max_send_count, ranks, rank_count);
  for (auto msc : all_max_send_count) {
    max_send_count = std::max(max_send_count, msc);
  }
  std::vector<IdType> nids(group_size);
  std::vector<int64_t> node_edge_indice(group_size);
  for (int i = 0; i < group_size; i++) {
    nids[i] = all_rank_node_start[i];
    node_edge_indice[i] = 0;
  }
  std::vector<int> send_edge_count(group_size), recv_edge_count(group_size);
  for (int64_t iter = 0; iter < max_send_count; iter++) {
    CCBarrier(ranks, rank_count);
    for (int r = 0; r < group_size; r++) {
      int64_t edge_count = 0;
      while (edge_count < kMaxEdgeCountPerBatch && nids[r] < all_rank_node_start[r + 1]) {
        if (node_edge_indice[r] >= local_edges_2d[nids[r]].size()) {
          nids[r]++;
          node_edge_indice[r] = 0;
          continue;
        }
        send_edge_buffer[r][edge_count][0] = nids[r];
        send_edge_buffer[r][edge_count][1] = local_edges_2d[nids[r]][node_edge_indice[r]];
        node_edge_indice[r]++;
        edge_count++;
      }
      send_edge_count[r] = edge_count;
    }
    CCAllToAll(send_edge_count.data(), sizeof(int), recv_edge_count.data(), sizeof(int), ranks, rank_count);
    for (int r = 0; r < group_size; r++) {
      if (r == group_rank) continue;
      for (int i = 0; i < recv_edge_count[r]; i++) {
        IdType src_id = recv_edge_buffer[r][i][0];
        IdType dst_id = recv_edge_buffer[r][i][1];
        assert(src_id >= local_node_start && src_id < local_node_start + local_node_count);
        local_edges_2d[src_id].push_back(dst_id);
      }
    }
  }
  CCBarrier(ranks, rank_count);
  WmmpFree(shared_edge_buffer);
}

template <typename IdType, bool IsChunked>
void LoadToCSRGraphFromEdgeBufferCommon(void *wm_csr_row_ptr,
                                        void *wm_csr_col_idx,
                                        void *local_edge_buffer,
                                        int64_t local_edge_count,
                                        int64_t total_src_node_count,
                                        int64_t total_edge_count,
                                        int64_t *final_total_edge_count,
                                        bool directed,
                                        bool reverse_edge,
                                        bool add_self_loop,
                                        cudaStream_t stream,
                                        const int *ranks,
                                        int rank_count) {
  int group_rank = GetRankOffsetInRanks(CCRank(), ranks, rank_count);
  const auto* local_edges = (const IdType*)local_edge_buffer;
  std::vector<std::vector<IdType>> local_edges_2d(total_src_node_count);
  for (int64_t i = 0; i < local_edge_count; i++) {
    IdType src_id, dst_id;
    src_id = local_edges[i * 2];
    dst_id = local_edges[i * 2 + 1];
    if (reverse_edge) {
      std::swap(src_id, dst_id);
    }
    local_edges_2d[src_id].push_back(dst_id);
    if (!directed) local_edges_2d[dst_id].push_back(src_id);
  }

  int64_t local_node_count;
  int64_t local_node_start = WmmpCollOffsetAndSize(total_src_node_count, (size_t*)&local_node_count, ranks, rank_count);

  ExchangeEdges<IdType>(local_edges_2d, local_node_start, local_node_count, ranks, rank_count);

  if (add_self_loop) {
    for (int64_t id = local_node_start; id < local_node_start + local_node_count; id++) {
      local_edges_2d[id].push_back(id);
    }
  }
  int64_t final_local_edge_count = 0;
  for (int64_t id = local_node_start; id < local_node_start + local_node_count; id++) {
    std::unordered_set<IdType> id_set;
    for (auto dst_id : local_edges_2d[id]) {
      id_set.insert(dst_id);
    }
    local_edges_2d[id].clear();
    for (auto dst_id : id_set) {
      local_edges_2d[id].push_back(dst_id);
    }
    final_local_edge_count += id_set.size();
  }
  std::vector<int64_t> all_final_edge_count;
  std::vector<int64_t> all_final_edge_start;
  CollAllGather(final_local_edge_count, &all_final_edge_count, ranks, rank_count);
  int64_t final_edge_start = 0;
  all_final_edge_start.push_back(final_edge_start);
  for (auto final_edge_count : all_final_edge_count) {
    final_edge_start += final_edge_count;
    all_final_edge_start.push_back(final_edge_start);
  }
  *final_total_edge_count = all_final_edge_start.back();
  int64_t local_edge_start_offset = all_final_edge_start[group_rank];
  const int kCopyBufferEltCount = 1024 * 1024 * 4;
  int64_t* csr_row_ptr_buffer_h = nullptr;
  IdType* csr_col_idx_buffer_h = nullptr;
  WM_CUDA_CHECK(cudaMallocHost(&csr_row_ptr_buffer_h, kCopyBufferEltCount * sizeof(int64_t)));
  WM_CUDA_CHECK(cudaMallocHost(&csr_col_idx_buffer_h, kCopyBufferEltCount * sizeof(IdType)));

  // copy csr_row_ptr
  int64_t node_edge_start_offset = local_edge_start_offset;
  IdType nid = local_node_start;
  WholeChunkedMemoryHandle* wcmh_csr_row_ptr = nullptr;
  WholeChunkedMemoryHandle* wcmh_csr_col_ind = nullptr;
  if (IsChunked) {
    int dev_id = 0;
    WM_CUDA_CHECK(cudaGetDevice(&dev_id));
    wcmh_csr_row_ptr = GetDeviceChunkedHandle((WholeChunkedMemory_t)wm_csr_row_ptr, dev_id);
    wcmh_csr_col_ind = GetDeviceChunkedHandle((WholeChunkedMemory_t)wm_csr_col_idx, dev_id);
  }
  while (true) {
    int node_count = 0;
    IdType batch_start = nid;
    while (node_count < kCopyBufferEltCount && nid < local_node_start + local_node_count) {
      csr_row_ptr_buffer_h[node_count++] = node_edge_start_offset;
      node_edge_start_offset += local_edges_2d[nid++].size();
    }
    if (IsChunked) {
      WcmmpMemcpyToWholeChunkedMemory(wcmh_csr_row_ptr, batch_start * sizeof(int64_t), csr_row_ptr_buffer_h, sizeof(int64_t) * node_count, stream);
      WM_CUDA_CHECK(cudaGetLastError());
      WM_CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
      WM_CUDA_CHECK(cudaMemcpy((int64_t *) wm_csr_row_ptr + batch_start,
                        csr_row_ptr_buffer_h,
                        sizeof(int64_t) * node_count,
                        cudaMemcpyHostToDevice));
    }
    if (nid == local_node_start + local_node_count) {
      break;
    }
  }
  csr_row_ptr_buffer_h[0] = node_edge_start_offset;
  if (IsChunked) {
    WcmmpMemcpyToWholeChunkedMemory(wcmh_csr_row_ptr, (local_node_start + local_node_count) * sizeof(int64_t), csr_row_ptr_buffer_h, sizeof(int64_t), stream);
    WM_CUDA_CHECK(cudaGetLastError());
    WM_CUDA_CHECK(cudaStreamSynchronize(stream));
  } else {
    WM_CUDA_CHECK(cudaMemcpy((int64_t *) wm_csr_row_ptr + local_node_start + local_node_count,
                      csr_row_ptr_buffer_h,
                      sizeof(int64_t),
                      cudaMemcpyHostToDevice));
  }

  // copy csr_col_idx
  int64_t edge_offset = local_edge_start_offset;
  nid = local_node_start;
  int64_t idx_for_nid = 0;

  while(true) {
    int edge_count = 0;
    while (edge_count < kCopyBufferEltCount) {
      int nid_left_neighbor_count = local_edges_2d[nid].size() - idx_for_nid;
      int copy_count = nid_left_neighbor_count;
      if (nid_left_neighbor_count > kCopyBufferEltCount - edge_count) {
        copy_count = kCopyBufferEltCount - edge_count;
      }
      if (copy_count > 0) {
        memcpy(csr_col_idx_buffer_h + edge_count,
               local_edges_2d[nid].data() + idx_for_nid,
               copy_count * sizeof(IdType));
      }
      edge_count += copy_count;
      if (nid_left_neighbor_count <= kCopyBufferEltCount - edge_count) {
        nid++;
        idx_for_nid = 0;
      } else {
        idx_for_nid += copy_count;
      }
      if (nid >= local_node_start + local_node_count) {
        break;
      }
    }
    if (edge_count > 0) {
      if (IsChunked) {
        WcmmpMemcpyToWholeChunkedMemory(wcmh_csr_col_ind, edge_offset * sizeof(IdType), csr_col_idx_buffer_h, sizeof(IdType) * edge_count, stream);
        WM_CUDA_CHECK(cudaGetLastError());
        WM_CUDA_CHECK(cudaStreamSynchronize(stream));
      } else {
        WM_CUDA_CHECK(cudaMemcpy((IdType *) wm_csr_col_idx + edge_offset,
                          csr_col_idx_buffer_h,
                          sizeof(IdType) * edge_count,
                          cudaMemcpyHostToDevice));
      }
      edge_offset += edge_count;
    }
    if (nid == local_node_start + local_node_count) {
      break;
    }
  }

  WM_CUDA_CHECK(cudaFreeHost(csr_row_ptr_buffer_h));
  WM_CUDA_CHECK(cudaFreeHost(csr_col_idx_buffer_h));
}

template <typename IdType>
void LoadToCSRGraphFromEdgeBuffer(void *wm_csr_row_ptr,
                                  void *wm_csr_col_idx,
                                  void* local_edge_buffer,
                                  int64_t local_edge_count,
                                  int64_t total_src_node_count,
                                  int64_t total_edge_count,
                                  int64_t* final_total_edge_count,
                                  bool directed,
                                  bool reverse_edge,
                                  bool add_self_loop,
                                  cudaStream_t stream,
                                  const int *ranks,
                                  int rank_count) {
  LoadToCSRGraphFromEdgeBufferCommon<IdType, false>(wm_csr_row_ptr,
                                    wm_csr_col_idx,
                                    local_edge_buffer,
                                    local_edge_count,
                                    total_src_node_count,
                                    total_edge_count,
                                    final_total_edge_count,
                                    directed,
                                    reverse_edge,
                                    add_self_loop,
                                    stream,
                                    ranks,
                                    rank_count);
}

REGISTER_DISPATCH_ONE_TYPE(LoadToCSRGraphFromEdgeBuffer, LoadToCSRGraphFromEdgeBuffer, SINT)

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
                                      cudaStream_t stream,
                                      const int *ranks,
                                      int rank_count) {
  assert(id_type == WMT_Int32 || id_type == WMT_Int64);
  if (!directed) reverse_edge = false;
  DISPATCH_ONE_TYPE(id_type, LoadToCSRGraphFromEdgeBuffer,
                    wm_csr_row_ptr, wm_csr_col_idx, local_edge_buffer, local_edge_count,
                    total_src_node_count, total_edge_count, final_total_edge_count,
                    directed, reverse_edge, add_self_loop,
                    stream, ranks, rank_count);
}

template <typename IdType>
void LoadToChunkedCSRGraphFromEdgeBuffer(void *wm_csr_row_ptr,
                                         void *wm_csr_col_idx,
                                         void *local_edge_buffer,
                                         int64_t local_edge_count,
                                         int64_t total_src_node_count,
                                         int64_t total_edge_count,
                                         int64_t *final_total_edge_count,
                                         bool directed,
                                         bool reverse_edge,
                                         bool add_self_loop,
                                         cudaStream_t stream,
                                         const int *ranks,
                                         int rank_count) {
  LoadToCSRGraphFromEdgeBufferCommon<IdType, true>(wm_csr_row_ptr,
                                                    wm_csr_col_idx,
                                                    local_edge_buffer,
                                                    local_edge_count,
                                                    total_src_node_count,
                                                    total_edge_count,
                                                    final_total_edge_count,
                                                    directed,
                                                    reverse_edge,
                                                    add_self_loop,
                                                    stream,
                                                    ranks,
                                                    rank_count);
}

REGISTER_DISPATCH_ONE_TYPE(LoadToChunkedCSRGraphFromEdgeBuffer, LoadToChunkedCSRGraphFromEdgeBuffer, SINT)

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
                                             cudaStream_t stream,
                                             const int *ranks,
                                             int rank_count) {
  assert(id_type == WMT_Int32 || id_type == WMT_Int64);
  if (!directed) reverse_edge = false;
  DISPATCH_ONE_TYPE(id_type, LoadToChunkedCSRGraphFromEdgeBuffer,
                    wm_csr_row_ptr, wm_csr_col_idx, local_edge_buffer, local_edge_count,
                    total_src_node_count, total_edge_count, final_total_edge_count,
                    directed, reverse_edge, add_self_loop,
                    stream, ranks, rank_count);
}
}