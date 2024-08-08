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
#include "wholememory/nccl_comms.hpp"

#include <cuda_runtime.h>
#include <nccl.h>

#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <thread>

#include <raft/core/comms.hpp>

#include "cuda_macros.hpp"
#include <cstring>
#include <raft/comms/detail/util.hpp>
#include <raft/core/error.hpp>

namespace wholememory {

nccl_comms::nccl_comms(ncclComm_t nccl_comm, int num_ranks, int rank, cudaStream_t rmm_stream)
  : nccl_comm_(nccl_comm), rmm_stream_(rmm_stream), num_ranks_(num_ranks), rank_(rank)
{
  initialize();
};

void nccl_comms::initialize()
{
  WM_CUDA_CHECK(cudaMallocHost(&host_send_buffer_, HOST_BUFFER_SIZE_PER_RANK * num_ranks_));
  WM_CUDA_CHECK(cudaMallocHost(&host_recv_buffer_, HOST_BUFFER_SIZE_PER_RANK * num_ranks_));
  WM_CUDA_CHECK(cudaMalloc(&buf_, sizeof(int)));
}

nccl_comms::~nccl_comms()
{
  WM_CUDA_CHECK_NO_THROW(cudaFreeHost(host_send_buffer_));
  WM_CUDA_CHECK_NO_THROW(cudaFreeHost(host_recv_buffer_));
  WM_CUDA_CHECK_NO_THROW(cudaFree(buf_));
}

static size_t get_nccl_datatype_size(ncclDataType_t datatype)
{
  switch (datatype) {
    case ncclInt8: return 1;
    case ncclUint8: return 1;
    case ncclInt32: return 4;
    case ncclUint32: return 4;
    case ncclInt64: return 8;
    case ncclUint64: return 8;
    case ncclFloat16: return 2;
    case ncclFloat32: return 4;
    case ncclFloat64: return 8;
#if defined(__CUDA_BF16_TYPES_EXIST__)
    case ncclBfloat16: return 2;
#endif
    default: WHOLEMEMORY_FAIL("get_nccl_datatype_size");
  }
  return WHOLEMEMORY_SUCCESS;
}

void nccl_comms::barrier() const
{
  allreduce(buf_, buf_, 1, ncclInt32, ncclSum, rmm_stream_);
  WM_CUDA_CHECK(cudaStreamSynchronize(rmm_stream_));
}

void nccl_comms::abort() const { RAFT_NCCL_TRY(ncclCommAbort(nccl_comm_)); }

void nccl_comms::allreduce(const void* sendbuff,
                           void* recvbuff,
                           size_t count,
                           ncclDataType_t datatype,
                           ncclRedOp_t op,
                           cudaStream_t stream) const
{
  RAFT_NCCL_TRY(ncclAllReduce(sendbuff, recvbuff, count, datatype, op, nccl_comm_, stream));
}

void nccl_comms::host_allreduce(
  const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op) const
{
  const size_t datatype_size = get_nccl_datatype_size(datatype);
  const size_t max_elt_count = HOST_BUFFER_SIZE_PER_RANK * num_ranks_ / datatype_size;
  for (size_t offset = 0; offset < count; offset += max_elt_count) {
    size_t elt_count = (count - offset > max_elt_count) ? max_elt_count : count - offset;
    std::memcpy(host_send_buffer_,
                static_cast<const char*>(sendbuff) + datatype_size * offset,
                elt_count * datatype_size);
    RAFT_NCCL_TRY(ncclAllReduce(
      host_send_buffer_, host_recv_buffer_, elt_count, datatype, op, nccl_comm_, rmm_stream_));
    WM_CUDA_CHECK(cudaStreamSynchronize(rmm_stream_));
    std::memcpy(static_cast<char*>(recvbuff) + datatype_size * offset,
                host_recv_buffer_,
                elt_count * datatype_size);
  }
}

void nccl_comms::bcast(
  void* buff, size_t count, ncclDataType_t datatype, int root, cudaStream_t stream) const
{
  RAFT_NCCL_TRY(ncclBroadcast(buff, buff, count, datatype, root, nccl_comm_, stream));
}

void nccl_comms::bcast(const void* sendbuff,
                       void* recvbuff,
                       size_t count,
                       ncclDataType_t datatype,
                       int root,
                       cudaStream_t stream) const
{
  RAFT_NCCL_TRY(ncclBroadcast(sendbuff, recvbuff, count, datatype, root, nccl_comm_, stream));
}

void nccl_comms::host_bcast(
  const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root) const
{
  const size_t datatype_size = get_nccl_datatype_size(datatype);
  const size_t max_elt_count = HOST_BUFFER_SIZE_PER_RANK * num_ranks_ / datatype_size;
  for (size_t offset = 0; offset < count; offset += max_elt_count) {
    size_t elt_count = (count - offset > max_elt_count) ? max_elt_count : count - offset;
    std::memcpy(host_send_buffer_,
                static_cast<const char*>(sendbuff) + datatype_size * offset,
                elt_count * datatype_size);
    RAFT_NCCL_TRY(ncclBroadcast(
      host_send_buffer_, host_recv_buffer_, elt_count, datatype, root, nccl_comm_, rmm_stream_));
    WM_CUDA_CHECK(cudaStreamSynchronize(rmm_stream_));
    std::memcpy(static_cast<char*>(recvbuff) + datatype_size * offset,
                host_recv_buffer_,
                elt_count * datatype_size);
  }
}

void nccl_comms::host_bcast(void* buff, size_t count, ncclDataType_t datatype, int root) const
{
  host_bcast(buff, buff, count, datatype, root);
}

void nccl_comms::reduce(const void* sendbuff,
                        void* recvbuff,
                        size_t count,
                        ncclDataType_t datatype,
                        ncclRedOp_t op,
                        int root,
                        cudaStream_t stream) const
{
  RAFT_NCCL_TRY(ncclReduce(sendbuff, recvbuff, count, datatype, op, root, nccl_comm_, stream));
}

void nccl_comms::host_reduce(const void* sendbuff,
                             void* recvbuff,
                             size_t count,
                             ncclDataType_t datatype,
                             ncclRedOp_t op,
                             int root) const
{
  const size_t datatype_size = get_nccl_datatype_size(datatype);
  const size_t max_elt_count = HOST_BUFFER_SIZE_PER_RANK * num_ranks_ / datatype_size;
  for (size_t offset = 0; offset < count; offset += max_elt_count) {
    size_t elt_count = (count - offset > max_elt_count) ? max_elt_count : count - offset;
    std::memcpy(host_send_buffer_,
                static_cast<const char*>(sendbuff) + datatype_size * offset,
                elt_count * datatype_size);
    RAFT_NCCL_TRY(ncclReduce(host_send_buffer_,
                             host_recv_buffer_,
                             elt_count,
                             datatype,
                             op,
                             root,
                             nccl_comm_,
                             rmm_stream_));
    WM_CUDA_CHECK(cudaStreamSynchronize(rmm_stream_));
    if (get_rank() == root) {
      std::memcpy(static_cast<char*>(recvbuff) + datatype_size * offset,
                  host_recv_buffer_,
                  elt_count * datatype_size);
    }
  }
}

void nccl_comms::allgather(const void* sendbuff,
                           void* recvbuff,
                           size_t sendcount,
                           ncclDataType_t datatype,
                           cudaStream_t stream) const
{
  RAFT_NCCL_TRY(ncclAllGather(sendbuff, recvbuff, sendcount, datatype, nccl_comm_, stream));
}

void nccl_comms::host_allgather(const void* sendbuff,
                                void* recvbuff,
                                size_t sendcount,
                                ncclDataType_t datatype) const
{
  const size_t datatype_size = get_nccl_datatype_size(datatype);
  const size_t max_elt_count = HOST_BUFFER_SIZE_PER_RANK / datatype_size;
  for (size_t offset = 0; offset < sendcount; offset += max_elt_count) {
    size_t elt_count = (sendcount - offset > max_elt_count) ? max_elt_count : sendcount - offset;
    std::memcpy(host_send_buffer_,
                static_cast<const char*>(sendbuff) + datatype_size * offset,
                elt_count * datatype_size);
    RAFT_NCCL_TRY(ncclAllGather(
      host_send_buffer_, host_recv_buffer_, sendcount, datatype, nccl_comm_, rmm_stream_));
    WM_CUDA_CHECK(cudaStreamSynchronize(rmm_stream_));
    for (int i = 0; i < get_size(); i++) {
      std::memcpy(
        static_cast<char*>(recvbuff) + datatype_size * offset + i * sendcount * datatype_size,
        host_recv_buffer_ + i * elt_count * datatype_size,
        elt_count * datatype_size);
    }
  }
}

void nccl_comms::allgatherv(const void* sendbuf,
                            void* recvbuf,
                            const size_t* recvcounts,
                            const size_t* displs,
                            ncclDataType_t datatype,
                            cudaStream_t stream) const
{
  // From: "An Empirical Evaluation of Allgatherv on Multi-GPU Systems" -
  // https://arxiv.org/pdf/1812.05964.pdf Listing 1 on page 4.
  WHOLEMEMORY_EXPECTS(
    num_ranks_ <= 2048,
    "# NCCL operations between ncclGroupStart & ncclGroupEnd shouldn't exceed 2048.");
  RAFT_NCCL_TRY(ncclGroupStart());
  for (int root = 0; root < num_ranks_; ++root) {
    size_t dtype_size = get_nccl_datatype_size(datatype);
    RAFT_NCCL_TRY(ncclBroadcast(sendbuf,
                                static_cast<char*>(recvbuf) + displs[root] * dtype_size,
                                recvcounts[root],
                                datatype,
                                root,
                                nccl_comm_,
                                stream));
  }
  RAFT_NCCL_TRY(ncclGroupEnd());
}

void nccl_comms::host_allgatherv(const void* sendbuf,
                                 void* recvbuf,
                                 const size_t* recvcounts,
                                 const size_t* displs,
                                 ncclDataType_t datatype) const
{
  size_t dtype_size = get_nccl_datatype_size(datatype);
  for (int root = 0; root < num_ranks_; ++root) {
    host_bcast(sendbuf,
               static_cast<char*>(recvbuf) + displs[root] * dtype_size,
               recvcounts[root],
               datatype,
               root);
  }
}

void nccl_comms::gather(const void* sendbuff,
                        void* recvbuff,
                        size_t sendcount,
                        ncclDataType_t datatype,
                        int root,
                        cudaStream_t stream) const
{
  size_t dtype_size = get_nccl_datatype_size(datatype);
  RAFT_NCCL_TRY(ncclGroupStart());
  if (get_rank() == root) {
    for (int r = 0; r < get_size(); ++r) {
      RAFT_NCCL_TRY(ncclRecv(static_cast<char*>(recvbuff) + sendcount * r * dtype_size,
                             sendcount,
                             datatype,
                             r,
                             nccl_comm_,
                             stream));
    }
  }
  RAFT_NCCL_TRY(ncclSend(sendbuff, sendcount, datatype, root, nccl_comm_, stream));
  RAFT_NCCL_TRY(ncclGroupEnd());
}

void nccl_comms::host_gather(
  const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, int root) const
{
  const size_t datatype_size = get_nccl_datatype_size(datatype);
  const size_t max_elt_count = HOST_BUFFER_SIZE_PER_RANK / datatype_size;
  for (size_t offset = 0; offset < sendcount; offset += max_elt_count) {
    size_t elt_count = (sendcount - offset > max_elt_count) ? max_elt_count : sendcount - offset;
    std::memcpy(host_send_buffer_,
                static_cast<const char*>(sendbuff) + datatype_size * offset,
                elt_count * datatype_size);
    gather(host_send_buffer_, host_recv_buffer_, sendcount, datatype, root, rmm_stream_);
    WM_CUDA_CHECK(cudaStreamSynchronize(rmm_stream_));
    if (rank_ == root) {
      for (int i = 0; i < num_ranks_; i++) {
        std::memcpy(
          static_cast<char*>(recvbuff) + datatype_size * offset + i * sendcount * datatype_size,
          host_recv_buffer_ + i * elt_count * datatype_size,
          elt_count * datatype_size);
      }
    }
  }
}

void nccl_comms::gatherv(const void* sendbuff,
                         void* recvbuff,
                         size_t sendcount,
                         const size_t* recvcounts,
                         const size_t* displs,
                         ncclDataType_t datatype,
                         int root,
                         cudaStream_t stream) const
{
  size_t dtype_size = get_nccl_datatype_size(datatype);
  RAFT_NCCL_TRY(ncclGroupStart());
  if (get_rank() == root) {
    for (int r = 0; r < get_size(); ++r) {
      RAFT_NCCL_TRY(ncclRecv(static_cast<char*>(recvbuff) + displs[r] * dtype_size,
                             recvcounts[r],
                             datatype,
                             r,
                             nccl_comm_,
                             stream));
    }
  }
  RAFT_NCCL_TRY(ncclSend(sendbuff, sendcount, datatype, root, nccl_comm_, stream));
  RAFT_NCCL_TRY(ncclGroupEnd());
}

void nccl_comms::reducescatter(const void* sendbuff,
                               void* recvbuff,
                               size_t recvcount,
                               ncclDataType_t datatype,
                               ncclRedOp_t op,
                               cudaStream_t stream) const
{
  RAFT_NCCL_TRY(ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, nccl_comm_, stream));
}
void nccl_comms::alltoall(const void* sendbuff,
                          void* recvbuff,
                          size_t sendcount,
                          ncclDataType_t datatype,
                          cudaStream_t stream) const
{
  size_t dtype_size = get_nccl_datatype_size(datatype);
  RAFT_NCCL_TRY(ncclGroupStart());
  for (int r = 0; r < get_size(); ++r) {
    RAFT_NCCL_TRY(ncclRecv(static_cast<char*>(recvbuff) + sendcount * r * dtype_size,
                           sendcount,
                           datatype,
                           r,
                           nccl_comm_,
                           stream));
  }
  for (int r = 0; r < get_size(); ++r) {
    RAFT_NCCL_TRY(ncclSend(static_cast<const char*>(sendbuff) + sendcount * r * dtype_size,
                           sendcount,
                           datatype,
                           r,
                           nccl_comm_,
                           stream));
  }
  RAFT_NCCL_TRY(ncclGroupEnd());
}

void nccl_comms::host_alltoall(const void* sendbuff,
                               void* recvbuff,
                               size_t sendcount,
                               ncclDataType_t datatype) const
{
  const size_t datatype_size = get_nccl_datatype_size(datatype);
  const size_t max_elt_count = HOST_BUFFER_SIZE_PER_RANK / datatype_size;
  for (size_t offset = 0; offset < sendcount; offset += max_elt_count) {
    size_t elt_count = (sendcount - offset > max_elt_count) ? max_elt_count : sendcount - offset;
    for (int i = 0; i < num_ranks_; i++) {
      std::memcpy(
        host_send_buffer_ + i * elt_count * datatype_size,
        static_cast<const char*>(sendbuff) + datatype_size * offset + i * sendcount * datatype_size,
        elt_count * datatype_size);
    }
    alltoall(host_send_buffer_, host_recv_buffer_, sendcount, datatype, rmm_stream_);
    WM_CUDA_CHECK(cudaStreamSynchronize(rmm_stream_));
    for (int i = 0; i < num_ranks_; i++) {
      std::memcpy(
        static_cast<char*>(recvbuff) + datatype_size * offset + i * sendcount * datatype_size,
        host_recv_buffer_ + i * elt_count * datatype_size,
        elt_count * datatype_size);
    }
  }
}

void nccl_comms::alltoallv(const void* sendbuff,
                           void* recvbuff,
                           const size_t* sendcounts,
                           const size_t* senddispls,
                           const size_t* recvcounts,
                           const size_t* recvdispls,
                           ncclDataType_t datatype,
                           cudaStream_t stream) const
{
  size_t dtype_size = get_nccl_datatype_size(datatype);
  RAFT_NCCL_TRY(ncclGroupStart());
  for (int r = 0; r < get_size(); ++r) {
    RAFT_NCCL_TRY(ncclRecv(static_cast<char*>(recvbuff) + recvdispls[r] * dtype_size,
                           recvcounts[r],
                           datatype,
                           r,
                           nccl_comm_,
                           stream));
  }
  for (int r = 0; r < get_size(); ++r) {
    RAFT_NCCL_TRY(ncclSend(static_cast<const char*>(sendbuff) + senddispls[r] * dtype_size,
                           sendcounts[r],
                           datatype,
                           r,
                           nccl_comm_,
                           stream));
  }
  RAFT_NCCL_TRY(ncclGroupEnd());
}

wholememory_error_code_t nccl_comms::sync_stream(cudaStream_t stream) const
{
  if (raft::comms::detail::nccl_sync_stream(nccl_comm_, stream) != raft::comms::status_t::SUCCESS) {
    return WHOLEMEMORY_COMMUNICATION_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t nccl_comms::sync_stream() const { return sync_stream(rmm_stream_); }

// if a thread is sending & receiving at the same time, use device_sendrecv to avoid deadlock
void nccl_comms::device_send(const void* send_buf,
                             size_t send_size,
                             int dest,
                             cudaStream_t stream) const
{
  RAFT_NCCL_TRY(ncclSend(send_buf, send_size, ncclUint8, dest, nccl_comm_, stream));
}

// if a thread is sending & receiving at the same time, use device_sendrecv to avoid deadlock
void nccl_comms::device_recv(void* recv_buf,
                             size_t recv_size,
                             int source,
                             cudaStream_t stream) const
{
  RAFT_NCCL_TRY(ncclRecv(recv_buf, recv_size, ncclUint8, source, nccl_comm_, stream));
}

void nccl_comms::device_sendrecv(const void* sendbuf,
                                 size_t sendsize,
                                 int dest,
                                 void* recvbuf,
                                 size_t recvsize,
                                 int source,
                                 cudaStream_t stream) const
{
  // ncclSend/ncclRecv pair needs to be inside ncclGroupStart/ncclGroupEnd to avoid deadlock
  RAFT_NCCL_TRY(ncclGroupStart());
  RAFT_NCCL_TRY(ncclSend(sendbuf, sendsize, ncclUint8, dest, nccl_comm_, stream));
  RAFT_NCCL_TRY(ncclRecv(recvbuf, recvsize, ncclUint8, source, nccl_comm_, stream));
  RAFT_NCCL_TRY(ncclGroupEnd());
}

void nccl_comms::device_multicast_sendrecv(const void* sendbuf,
                                           std::vector<size_t> const& sendsizes,
                                           std::vector<size_t> const& sendoffsets,
                                           std::vector<int> const& dests,
                                           void* recvbuf,
                                           std::vector<size_t> const& recvsizes,
                                           std::vector<size_t> const& recvoffsets,
                                           std::vector<int> const& sources,
                                           cudaStream_t stream) const
{
  // ncclSend/ncclRecv pair needs to be inside ncclGroupStart/ncclGroupEnd to avoid deadlock
  RAFT_NCCL_TRY(ncclGroupStart());
  for (size_t i = 0; i < sendsizes.size(); ++i) {
    RAFT_NCCL_TRY(ncclSend(static_cast<const char*>(sendbuf) + sendoffsets[i],
                           sendsizes[i],
                           ncclUint8,
                           dests[i],
                           nccl_comm_,
                           stream));
  }
  for (size_t i = 0; i < recvsizes.size(); ++i) {
    RAFT_NCCL_TRY(ncclRecv(static_cast<char*>(recvbuf) + recvoffsets[i],
                           recvsizes[i],
                           ncclUint8,
                           sources[i],
                           nccl_comm_,
                           stream));
  }
  RAFT_NCCL_TRY(ncclGroupEnd());
}

void nccl_comms::group_start() const { RAFT_NCCL_TRY(ncclGroupStart()); }

void nccl_comms::group_end() const { RAFT_NCCL_TRY(ncclGroupEnd()); }

ncclComm_t nccl_comms::raw_nccl_comm() const { return nccl_comm_; }

}  // namespace wholememory
