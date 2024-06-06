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

#include <vector>

#include <wholememory/wholememory.h>

namespace wholememory {

class nccl_comms {
 public:
  nccl_comms() = delete;

  /**
   * @brief Constructor for collective + point-to-point operation.
   * @param nccl_comm initialized nccl comm
   * @param num_ranks number of ranks in the cluster
   * @param rank rank of the current worker
   * @param stream cuda stream for synchronizing and ordering collective operations
   */
  nccl_comms(ncclComm_t nccl_comm, int num_ranks, int rank, cudaStream_t stream);

  void initialize();

  ~nccl_comms();

  int get_size() const { return num_ranks_; }

  int get_rank() const { return rank_; }

  void barrier() const;

  void abort() const;

  void allreduce(const void* sendbuff,
                 void* recvbuff,
                 size_t count,
                 ncclDataType_t datatype,
                 ncclRedOp_t op,
                 cudaStream_t stream) const;

  void host_allreduce(const void* sendbuff,
                      void* recvbuff,
                      size_t count,
                      ncclDataType_t datatype,
                      ncclRedOp_t op) const;

  void bcast(
    void* buff, size_t count, ncclDataType_t datatype, int root, cudaStream_t stream) const;

  void bcast(const void* sendbuff,
             void* recvbuff,
             size_t count,
             ncclDataType_t datatype,
             int root,
             cudaStream_t stream) const;

  void host_bcast(
    const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root) const;

  void host_bcast(void* buff, size_t count, ncclDataType_t datatype, int root) const;

  void reduce(const void* sendbuff,
              void* recvbuff,
              size_t count,
              ncclDataType_t datatype,
              ncclRedOp_t op,
              int root,
              cudaStream_t stream) const;

  void host_reduce(const void* sendbuff,
                   void* recvbuff,
                   size_t count,
                   ncclDataType_t datatype,
                   ncclRedOp_t op,
                   int root) const;

  void allgather(const void* sendbuff,
                 void* recvbuff,
                 size_t sendcount,
                 ncclDataType_t datatype,
                 cudaStream_t stream) const;

  void host_allgather(const void* sendbuff,
                      void* recvbuff,
                      size_t sendcount,
                      ncclDataType_t datatype) const;

  void allgatherv(const void* sendbuf,
                  void* recvbuf,
                  const size_t* recvcounts,
                  const size_t* displs,
                  ncclDataType_t datatype,
                  cudaStream_t stream) const;

  void host_allgatherv(const void* sendbuf,
                       void* recvbuf,
                       const size_t* recvcounts,
                       const size_t* displs,
                       ncclDataType_t datatype) const;

  void gather(const void* sendbuff,
              void* recvbuff,
              size_t sendcount,
              ncclDataType_t datatype,
              int root,
              cudaStream_t stream) const;

  void host_gather(const void* sendbuff,
                   void* recvbuff,
                   size_t sendcount,
                   ncclDataType_t datatype,
                   int root) const;

  void gatherv(const void* sendbuff,
               void* recvbuff,
               size_t sendcount,
               const size_t* recvcounts,
               const size_t* displs,
               ncclDataType_t datatype,
               int root,
               cudaStream_t stream) const;

  void reducescatter(const void* sendbuff,
                     void* recvbuff,
                     size_t recvcount,
                     ncclDataType_t datatype,
                     ncclRedOp_t op,
                     cudaStream_t stream) const;

  void alltoall(const void* sendbuff,
                void* recvbuff,
                size_t sendcount,
                ncclDataType_t datatype,
                cudaStream_t stream) const;

  void host_alltoall(const void* sendbuff,
                     void* recvbuff,
                     size_t sendcount,
                     ncclDataType_t datatype) const;

  void alltoallv(const void* sendbuff,
                 void* recvbuff,
                 const size_t* sendcounts,
                 const size_t* senddispls,
                 const size_t* recvcounts,
                 const size_t* recvdispls,
                 ncclDataType_t datatype,
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

  void group_start() const;

  void group_end() const;
  ncclComm_t raw_nccl_comm() const;

 private:
  ncclComm_t nccl_comm_;
  cudaStream_t rmm_stream_;

  int num_ranks_;
  int rank_;

  char* host_send_buffer_;
  char* host_recv_buffer_;
  static constexpr size_t HOST_BUFFER_SIZE_PER_RANK = 1LL * 1024 * 1024;
  int32_t* buf_;
};

}  // namespace wholememory
