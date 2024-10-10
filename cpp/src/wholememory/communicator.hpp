/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <cuda_runtime_api.h>
#include <nccl.h>

#include <cstring>
#include <map>
#include <mutex>
#include <vector>

#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

#include "cuda_macros.hpp"

namespace wholememory {

class nccl_comms;

}

struct wholememory_comm_ {
  wholememory_comm_(ncclComm_t nccl_comm, int num_ranks, int rank, cudaStream_t stream);
  ~wholememory_comm_();

  void barrier() const;

  void abort() const;

  void allreduce(const void* sendbuff,
                 void* recvbuff,
                 size_t count,
                 wholememory_dtype_t datatype,
                 ncclRedOp_t op,
                 cudaStream_t stream) const;

  void host_allreduce(const void* sendbuff,
                      void* recvbuff,
                      size_t count,
                      wholememory_dtype_t datatype,
                      ncclRedOp_t op) const;

  void bcast(
    void* buff, size_t count, wholememory_dtype_t datatype, int root, cudaStream_t stream) const;

  void bcast(const void* sendbuff,
             void* recvbuff,
             size_t count,
             wholememory_dtype_t datatype,
             int root,
             cudaStream_t stream) const;

  void host_bcast(const void* sendbuff,
                  void* recvbuff,
                  size_t count,
                  wholememory_dtype_t datatype,
                  int root) const;

  void host_bcast(void* buff, size_t count, wholememory_dtype_t datatype, int root) const;

  void reduce(const void* sendbuff,
              void* recvbuff,
              size_t count,
              wholememory_dtype_t datatype,
              ncclRedOp_t op,
              int root,
              cudaStream_t stream) const;

  void host_reduce(const void* sendbuff,
                   void* recvbuff,
                   size_t count,
                   wholememory_dtype_t datatype,
                   ncclRedOp_t op,
                   int root) const;

  void allgather(const void* sendbuff,
                 void* recvbuff,
                 size_t sendcount,
                 wholememory_dtype_t datatype,
                 cudaStream_t stream) const;

  void host_allgather(const void* sendbuff,
                      void* recvbuff,
                      size_t sendcount,
                      wholememory_dtype_t datatype) const;

  void allgatherv(const void* sendbuf,
                  void* recvbuf,
                  const size_t* recvcounts,
                  const size_t* displs,
                  wholememory_dtype_t datatype,
                  cudaStream_t stream) const;

  void host_allgatherv(const void* sendbuf,
                       void* recvbuf,
                       const size_t* recvcounts,
                       const size_t* displs,
                       wholememory_dtype_t datatype) const;

  void gather(const void* sendbuff,
              void* recvbuff,
              size_t sendcount,
              wholememory_dtype_t datatype,
              int root,
              cudaStream_t stream) const;

  void host_gather(const void* sendbuff,
                   void* recvbuff,
                   size_t sendcount,
                   wholememory_dtype_t datatype,
                   int root) const;

  void gatherv(const void* sendbuff,
               void* recvbuff,
               size_t sendcount,
               const size_t* recvcounts,
               const size_t* displs,
               wholememory_dtype_t datatype,
               int root,
               cudaStream_t stream) const;

  void reducescatter(const void* sendbuff,
                     void* recvbuff,
                     size_t recvcount,
                     wholememory_dtype_t datatype,
                     ncclRedOp_t op,
                     cudaStream_t stream) const;

  void alltoall(const void* sendbuff,
                void* recvbuff,
                size_t sendcount,
                wholememory_dtype_t datatype,
                cudaStream_t stream) const;

  void host_alltoall(const void* sendbuff,
                     void* recvbuff,
                     size_t sendcount,
                     wholememory_dtype_t datatype) const;

  void alltoallv(const void* sendbuff,
                 void* recvbuff,
                 const size_t* sendcounts,
                 const size_t* senddispls,
                 const size_t* recvcounts,
                 const size_t* recvdispls,
                 wholememory_dtype_t datatype,
                 cudaStream_t stream) const;

  wholememory_error_code_t sync_stream(cudaStream_t stream) const;

  wholememory_error_code_t sync_stream() const;

  // if a thread is sending & receiving at the same time, use device_sendrecv to avoid deadlock
  void device_send(const void* send_buf, size_t send_size, int dest, cudaStream_t stream) const;

  // if a thread is sending & receiving at the same time, use device_sendrecv to avoid deadlock
  void device_recv(void* recv_buf, size_t recv_size, int source, cudaStream_t stream) const;

  void device_sendrecv(const void* sendbuf,
                       size_t sendsize,
                       int dest,
                       void* recvbuf,
                       size_t recvsize,
                       int source,
                       cudaStream_t stream) const;

  void device_multicast_sendrecv(const void* sendbuf,
                                 std::vector<size_t> const& sendsizes,
                                 std::vector<size_t> const& sendoffsets,
                                 std::vector<int> const& dests,
                                 void* recvbuf,
                                 std::vector<size_t> const& recvsizes,
                                 std::vector<size_t> const& recvoffsets,
                                 std::vector<int> const& sources,
                                 cudaStream_t stream) const;

  bool is_intranode() const;

  bool is_intra_mnnvl() const;
  bool support_type_location(wholememory_memory_type_t memory_type,
                             wholememory_memory_location_t memory_location) const;

  void group_start() const;

  void group_end() const;

  wholememory::nccl_comms* raft_nccl_comm;
  cudaStream_t comm_stream = nullptr;
  cudaEvent_t cuda_event   = nullptr;
  ncclComm_t raw_nccl_comm = nullptr;

  int world_rank = 0;
  int world_size = 1;

  int intra_node_first_rank     = -1;
  int intra_node_rank           = -1;
  int intra_node_rank_num       = 0;
  int intra_node_first_rank_pid = -1;

  clique_info_t clique_info;

  int comm_id = -1;

  int dev_id            = -1;
  int local_gpu_ids[16] = {0};
  bool support_mnnvl    = false;

  size_t alloc_granularity = 2 * 1024 * 1024UL;

  std::mutex mu;
  std::map<int, wholememory_handle_t> wholememory_map;
  wholememory_distributed_backend_t distributed_backend = WHOLEMEMORY_DB_NCCL;
#ifdef WITH_NVSHMEM_SUPPORT
  bool bind_to_nvshmem = false;
#endif
} __attribute__((aligned(64)));

template <typename TypeT>
inline bool wm_comm_check_all_same(wholememory_comm_t comm, const TypeT& t)
{
  std::unique_ptr<TypeT[]> t_array(new TypeT[comm->world_size]());
  comm->host_allgather(&t, t_array.get(), sizeof(TypeT), WHOLEMEMORY_DT_INT8);
  for (int r = 0; r < comm->world_size; r++) {
    if (t_array.get()[r] != t) return false;
  }
  return true;
}

template <>
inline bool wm_comm_check_all_same(wholememory_comm_t comm, const std::string& str)
{
  size_t str_len = str.size();
  if (!wm_comm_check_all_same(comm, str_len)) return false;
  std::string cat_str;
  cat_str.resize(str_len * comm->world_size, '\0');
  comm->host_allgather(
    str.data(), const_cast<char*>(cat_str.c_str()), str_len, WHOLEMEMORY_DT_INT8);
  for (int r = 0; r < comm->world_size; r++) {
    if (std::strncmp(str.data(), cat_str.data() + r * str_len, str_len) != 0) return false;
  }
  return true;
}

#define WM_COMM_CHECK_ALL_SAME(comm, data)                                                         \
  do {                                                                                             \
    if (!wm_comm_check_all_same(comm, data)) { WHOLEMEMORY_FATAL("COMM_CHECK_ALL_SAME failed."); } \
  } while (0)

namespace wholememory {

wholememory_error_code_t create_unique_id(wholememory_unique_id_t* unique_id) noexcept;

wholememory_error_code_t create_communicator(wholememory_comm_t* comm,
                                             wholememory_unique_id_t unique_id,
                                             int rank,
                                             int size) noexcept;

wholememory_error_code_t split_communicator(wholememory_comm_t* new_comm,
                                            wholememory_comm_t parent_comm,
                                            int color,
                                            int key) noexcept;

wholememory_error_code_t destroy_communicator_locked(wholememory_comm_t comm) noexcept;

wholememory_error_code_t destroy_communicator(wholememory_comm_t comm) noexcept;

wholememory_error_code_t communicator_support_type_location(
  wholememory_comm_t comm,
  wholememory_memory_type_t memory_type,
  wholememory_memory_location_t memory_location) noexcept;

wholememory_error_code_t destroy_all_communicators() noexcept;

wholememory_error_code_t communicator_get_rank(int* rank, wholememory_comm_t comm) noexcept;

wholememory_error_code_t communicator_get_size(int* size, wholememory_comm_t comm) noexcept;

wholememory_error_code_t communicator_get_local_size(int* local_size,
                                                     wholememory_comm_t comm) noexcept;

wholememory_error_code_t communicator_get_clique_info(clique_info_t* clique_info,
                                                      wholememory_comm_t comm) noexcept;

void communicator_barrier(wholememory_comm_t comm);

bool is_intranode_communicator(wholememory_comm_t comm) noexcept;

bool is_intra_mnnvl_communicator(wholememory_comm_t comm) noexcept;

std::string get_temporary_directory_path(wholememory_comm_t comm);

std::string get_shm_prefix(wholememory_comm_t comm);

wholememory_error_code_t communicator_set_distributed_backend(
  wholememory_comm_t comm, wholememory_distributed_backend_t distributed_backend) noexcept;

wholememory_distributed_backend_t communicator_get_distributed_backend(
  wholememory_comm_t comm) noexcept;

#ifdef WITH_NVSHMEM_SUPPORT

bool communicator_is_bind_to_nvshmem(wholememory_comm_t comm) noexcept;

wholememory_error_code_t init_nvshmem_with_comm(wholememory_comm_t comm) noexcept;
wholememory_error_code_t finalize_nvshmem_locked(wholememory_comm_t comm) noexcept;

#endif

}  // namespace wholememory
