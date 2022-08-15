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
#include <cuda_runtime_api.h>
#include <nccl.h>
#include <pthread.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <vector>

#include "macros.h"
#include "whole_memory.h"

namespace whole_graph {

#define WM_NCCL_CHECK(X)                                   \
  do {                                                     \
    auto result = X;                                       \
    if (result != ncclSuccess) {                           \
      const char *p_err_str = ncclGetErrorString(result);  \
      fprintf(stderr, "File %s Line %d %s returned %s.\n", \
              __FILE__, __LINE__, #X, p_err_str);          \
      abort();                                             \
    }                                                      \
  } while (0)

struct WholeMemoryPeerInfo {
  int rank;
  int cudaDev;
  pid_t pid;
  dev_t shmDev;
  int64_t busId;
  char hostId[1024];
};

void WmmpGetUniqueId(WmmpUniqueId *uniqueId) {
  WM_NCCL_CHECK(ncclGetUniqueId(&uniqueId->nccl_unique_id_));
}

class BootstrapCommunicatorImpl {
 public:
  BootstrapCommunicatorImpl() {
    rank_ = 0;
    size_ = 1;
    busId = 0;
    cudaDev = -1;
    intraNodeRank0pid = 0;
    peerInfo = nullptr;
    nccl_comm_ = nullptr;
  }
  ~BootstrapCommunicatorImpl() {
    if (peerInfo) {
      free(peerInfo);
      peerInfo = nullptr;
    }
    if (nccl_comm_) {
      ncclCommDestroy(nccl_comm_);
      nccl_comm_ = nullptr;
    }
    if (device_buffer_) {
      WM_CUDA_CHECK(cudaFree(device_buffer_));
      device_buffer_ = nullptr;
    }
    if (host_send_buffer_ || host_recv_buffer_) {
      WM_CHECK(host_recv_buffer_ != nullptr && host_send_buffer_ != nullptr);
      WM_CUDA_CHECK(cudaFreeHost(host_send_buffer_));
      WM_CUDA_CHECK(cudaFreeHost(host_recv_buffer_));
      host_send_buffer_ = host_recv_buffer_ = nullptr;
    }
    if (nccl_stream_) {
      WM_CUDA_CHECK(cudaStreamDestroy(nccl_stream_));
      nccl_stream_ = nullptr;
    }
    if (event_) {
      WM_CUDA_CHECK(cudaEventDestroy(event_));
      event_ = nullptr;
    }
  }
  void InitRank(int size, WmmpUniqueId unique_id, int rank);
  int Rank() const {
    return rank_;
  }
  int Size() const {
    return size_;
  }
  int64_t BusID() const {
    return busId;
  }
  int CUDADev() const {
    return cudaDev;
  }
  bool IsIntraNode() const {
    return intraNodeRanks == size_;
  }
  void AllToAll(const void *send_buf,
                int send_size,
                void *recv_buf,
                int recv_size);
  void AllToAllV(const void **send_bufs,
                 const int *send_sizes,
                 void **recv_bufs,
                 const int *recv_sizes);
  void Barrier();
  ncclComm_t GetNCCLCommunicator() const {
    return nccl_comm_;
  }
  cudaStream_t GetNCCLStream() const {
    return nccl_stream_;
  }
  void RecordEvent() {
    WM_CUDA_CHECK(cudaEventRecord(event_, nccl_stream_));
  }
  void Synchronize();

 private:
  void CommInitRankSync();

  ncclComm_t nccl_comm_;

  struct WholeMemoryPeerInfo *peerInfo;
  int rank_;
  int size_;
  int cudaDev;
  int64_t busId;
  pid_t intraNodeRank0pid;
  int intraProcRank0 = -1, intraProcRank = -1, intraProcRanks = 0;
  int intraNodeRank0 = -1, intraNodeRank = -1, intraNodeRanks = 0;

  char *host_send_buffer_ = nullptr;
  char *host_recv_buffer_ = nullptr;
  char *device_buffer_ = nullptr;
  cudaStream_t nccl_stream_ = nullptr;
  cudaEvent_t event_ = nullptr;

  static constexpr int kDeviceBufferSize = 256;
  static constexpr int kHostBufferSizePerRank = 1 * 1024 * 1024;
};

BootstrapCommunicator::BootstrapCommunicator() {
  impl_ = new BootstrapCommunicatorImpl();
}
BootstrapCommunicator::~BootstrapCommunicator() {
  if (impl_) {
    delete impl_;
    impl_ = nullptr;
  }
}

void BootstrapCommunicator::InitRank(int size, WmmpUniqueId unique_id, int rank) const {
  impl_->InitRank(size, unique_id, rank);
}

int BootstrapCommunicator::Rank() {
  return impl_->Rank();
}
int BootstrapCommunicator::Size() {
  return impl_->Size();
}
void BootstrapCommunicator::AllToAll(const void *send_buf,
                                     int send_size,
                                     void *recv_buf,
                                     int recv_size) const {
  return impl_->AllToAll(send_buf, send_size, recv_buf, recv_size);
}
void BootstrapCommunicator::AllToAllV(const void **send_bufs,
                                      const int *send_sizes,
                                      void **recv_bufs,
                                      const int *recv_sizes) const {
  return impl_->AllToAllV(send_bufs, send_sizes, recv_bufs, recv_sizes);
}
void BootstrapCommunicator::Barrier() {
  return impl_->Barrier();
}
ncclComm_t BootstrapCommunicator::GetNCCLCommunicator() const {
  return impl_->GetNCCLCommunicator();
}
cudaStream_t BootstrapCommunicator::GetNCCLStream() const {
  return impl_->GetNCCLStream();
}
void BootstrapCommunicator::RecordEvent() {
  impl_->RecordEvent();
}
void BootstrapCommunicator::Synchronize() {
  impl_->Synchronize();
}
int BootstrapCommunicator::CUDADevID() const {
  return impl_->CUDADev();
}
bool BootstrapCommunicator::IsIntraNode() const {
  return impl_->IsIntraNode();
}
bool IsCUDAAccessibleMemory(const void *ptr) {
  cudaPointerAttributes attr{};
  cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
  return !(err != cudaSuccess || attr.type == cudaMemoryTypeUnregistered);
}
void BootstrapCommunicatorImpl::AllToAll(const void *send_buf, int send_size, void *recv_buf, int recv_size) {
  bool send_cuda = IsCUDAAccessibleMemory(send_buf);
  bool recv_cuda = IsCUDAAccessibleMemory(recv_buf);
  WM_CHECK(send_size == recv_size);
  if (send_cuda && recv_cuda) {
    WM_CUDA_CHECK(cudaStreamSynchronize(nccl_stream_));
    WM_NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < size_; i++) {
      WM_NCCL_CHECK(ncclSend((char *) send_buf + i * send_size, send_size, ncclChar, i, nccl_comm_, nccl_stream_));
      WM_NCCL_CHECK(ncclRecv((char *) recv_buf + i * recv_size, recv_size, ncclChar, i, nccl_comm_, nccl_stream_));
    }
    WM_NCCL_CHECK(ncclGroupEnd());
    WM_CUDA_CHECK(cudaStreamSynchronize(nccl_stream_));
  } else {
    WM_CUDA_CHECK(cudaStreamSynchronize(nccl_stream_));
    for (int offset = 0; offset < send_size; offset += kHostBufferSizePerRank) {
      int chunk_size = send_size - offset;
      if (chunk_size > kHostBufferSizePerRank) chunk_size = kHostBufferSizePerRank;
      WM_CUDA_CHECK(cudaStreamSynchronize(nccl_stream_));
      for (int i = 0; i < size_; i++) {
        WM_CUDA_CHECK(cudaMemcpyAsync(host_send_buffer_ + i * kHostBufferSizePerRank,
                                      (const char *) send_buf + i * send_size + offset,
                                      chunk_size,
                                      cudaMemcpyDefault,
                                      nccl_stream_));
      }
      WM_NCCL_CHECK(ncclGroupStart());
      for (int i = 0; i < size_; i++) {
        WM_NCCL_CHECK(ncclSend(host_send_buffer_ + i * kHostBufferSizePerRank,
                               chunk_size,
                               ncclChar,
                               i,
                               nccl_comm_,
                               nccl_stream_));
        WM_NCCL_CHECK(ncclRecv(host_recv_buffer_ + i * kHostBufferSizePerRank,
                               chunk_size,
                               ncclChar,
                               i,
                               nccl_comm_,
                               nccl_stream_));
      }
      WM_NCCL_CHECK(ncclGroupEnd());
      for (int i = 0; i < size_; i++) {
        WM_CUDA_CHECK(cudaMemcpyAsync((char *) recv_buf + i * recv_size + offset,
                                      host_recv_buffer_ + i * kHostBufferSizePerRank,
                                      chunk_size,
                                      cudaMemcpyDefault,
                                      nccl_stream_));
      }
      WM_CUDA_CHECK(cudaStreamSynchronize(nccl_stream_));
    }
  }
}
void BootstrapCommunicatorImpl::AllToAllV(const void **send_bufs,
                                          const int *send_sizes,
                                          void **recv_bufs,
                                          const int *recv_sizes) {
  int max_size = 0;
  bool send_cuda = true;
  bool recv_cuda = true;
  for (int i = 0; i < size_; i++) {
    max_size = std::max<int>(max_size, send_sizes[i]);
    max_size = std::max<int>(max_size, recv_sizes[i]);
    if (send_sizes[i] > 0 && !IsCUDAAccessibleMemory(send_bufs[i])) send_cuda = false;
    if (recv_sizes[i] > 0 && !IsCUDAAccessibleMemory(recv_bufs[i])) recv_cuda = false;
  }
  if (send_cuda && recv_cuda) {
    WM_CUDA_CHECK(cudaStreamSynchronize(nccl_stream_));
    WM_NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < size_; i++) {
      if (send_sizes[i] > 0) {
        WM_NCCL_CHECK(ncclSend(send_bufs[i], send_sizes[i], ncclChar, i, nccl_comm_, nccl_stream_));
      }
      if (recv_sizes[i] > 0) {
        WM_NCCL_CHECK(ncclRecv(recv_bufs[i], recv_sizes[i], ncclChar, i, nccl_comm_, nccl_stream_));
      }
    }
    WM_NCCL_CHECK(ncclGroupEnd());
    WM_CUDA_CHECK(cudaStreamSynchronize(nccl_stream_));
  } else {
    WM_CUDA_CHECK(cudaStreamSynchronize(nccl_stream_));
    for (int offset = 0; offset < max_size; offset += kHostBufferSizePerRank) {
      WM_CUDA_CHECK(cudaStreamSynchronize(nccl_stream_));
      for (int i = 0; i < size_; i++) {
        int send_chunk_size = std::max<int>(send_sizes[i] - offset, 0);
        send_chunk_size = std::min<int>(send_chunk_size, kHostBufferSizePerRank);
        if (send_chunk_size > 0) {
          WM_CUDA_CHECK(cudaMemcpyAsync(host_send_buffer_ + i * kHostBufferSizePerRank,
                                        (const char *) send_bufs[i] + offset,
                                        send_chunk_size,
                                        cudaMemcpyDefault, nccl_stream_));
        }
      }
      WM_NCCL_CHECK(ncclGroupStart());
      for (int i = 0; i < size_; i++) {
        int send_chunk_size = std::max<int>(send_sizes[i] - offset, 0);
        send_chunk_size = std::min<int>(send_chunk_size, kHostBufferSizePerRank);
        if (send_chunk_size > 0) {
          WM_NCCL_CHECK(ncclSend(host_send_buffer_ + i * kHostBufferSizePerRank,
                                 send_chunk_size,
                                 ncclChar,
                                 i,
                                 nccl_comm_,
                                 nccl_stream_));
        }
        int recv_chunk_size = std::max<int>(recv_sizes[i] - offset, 0);
        recv_chunk_size = std::min<int>(recv_chunk_size, kHostBufferSizePerRank);
        if (recv_chunk_size > 0) {
          WM_NCCL_CHECK(ncclRecv(host_recv_buffer_ + i * kHostBufferSizePerRank,
                                 recv_chunk_size,
                                 ncclChar,
                                 i,
                                 nccl_comm_,
                                 nccl_stream_));
        }
      }
      WM_NCCL_CHECK(ncclGroupEnd());
      for (int i = 0; i < size_; i++) {
        int recv_chunk_size = std::max<int>(recv_sizes[i] - offset, 0);
        recv_chunk_size = std::min<int>(recv_chunk_size, kHostBufferSizePerRank);
        if (recv_chunk_size > 0) {
          WM_CUDA_CHECK(cudaMemcpyAsync((char *) recv_bufs[i] + offset,
                                        host_recv_buffer_ + i * kHostBufferSizePerRank,
                                        recv_chunk_size,
                                        cudaMemcpyDefault, nccl_stream_));
        }
      }
      WM_CUDA_CHECK(cudaStreamSynchronize(nccl_stream_));
    }
    if (max_size == 0) WM_CUDA_CHECK(cudaStreamSynchronize(nccl_stream_));
  }
}
void BootstrapCommunicatorImpl::Barrier() {
  WM_NCCL_CHECK(ncclAllReduce(device_buffer_, device_buffer_, 1, ncclFloat, ncclSum, nccl_comm_, nccl_stream_));
  WM_CUDA_CHECK(cudaStreamSynchronize(nccl_stream_));
}

void BootstrapCommunicatorImpl::InitRank(int size, WmmpUniqueId unique_id, int rank) {
  rank_ = rank;
  size_ = size;
  WM_CUDA_CHECK(cudaGetDevice(&cudaDev));
  WM_CUDA_CHECK(cudaMalloc(&device_buffer_, kDeviceBufferSize));
  WM_CUDA_CHECK(cudaMallocHost(&host_send_buffer_, kHostBufferSizePerRank * size_));
  WM_CUDA_CHECK(cudaMallocHost(&host_recv_buffer_, kHostBufferSizePerRank * size_));
  WM_CUDA_CHECK(cudaStreamCreate(&nccl_stream_));
  WM_CUDA_CHECK(cudaEventCreate(&event_));

  // Make sure the CUDA runtime is initialized.
  WM_CUDA_CHECK(cudaFree(nullptr));

  if (size < 1 || rank < 0 || rank >= size) {
    fprintf(stderr, "Invalid rank requested : %d/%d", rank, size);
    abort();
  }

  WM_NCCL_CHECK(ncclCommInitRank(&nccl_comm_, size_, unique_id.nccl_unique_id_, rank_));

  CommInitRankSync();
}

void BootstrapCommunicatorImpl::Synchronize() {
  while (true) {
    cudaError_t cuda_error = cudaEventQuery(event_);
    if (cuda_error != cudaSuccess && cuda_error != cudaErrorNotReady) {
      fprintf(stderr, "[BootstrapCommunicatorImpl::Synchronize] failed.\n");
      WM_CUDA_CHECK(cuda_error);
    }
    if (cuda_error == cudaSuccess) break;
    ncclResult_t nccl_error;
    WM_NCCL_CHECK(ncclCommGetAsyncError(nccl_comm_, &nccl_error));
    WM_CHECK(nccl_error == ncclSuccess);
  }
}

#if 0
static int GetCudaCompCap() {
  int cudaDev;
  if (cudaGetDevice(&cudaDev) != cudaSuccess) return 0;
  int ccMajor, ccMinor;
  if (cudaDeviceGetAttribute(&ccMajor, cudaDevAttrComputeCapabilityMajor, cudaDev) != cudaSuccess) return 0;
  if (cudaDeviceGetAttribute(&ccMinor, cudaDevAttrComputeCapabilityMinor, cudaDev) != cudaSuccess) return 0;
  return ccMajor * 10 + ccMinor;
}

static void int64ToBusId(int64_t id, char *busId) {
  sprintf(busId, "%04lx:%02lx:%02lx.%01lx", (id) >> 20, (id & 0xff000) >> 12, (id & 0xff0) >> 4, (id & 0xf));
}
#endif

static void busIdToInt64(const char *busId, int64_t *id) {
  const int size = strlen(busId);
  char *hexStr = (char *) malloc(size + 1);
  int hexOffset = 0;
  for (int i = 0; i < size; i++) {
    char c = busId[i];
    if (c == '.' || c == ':') continue;
    if ((c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f')) {
      hexStr[hexOffset++] = busId[i];
    } else
      break;
  }
  hexStr[hexOffset] = '\0';
  *id = strtol(hexStr, NULL, 16);
  free(hexStr);
}

// Convert a logical cudaDev index to the NVML device minor number
static void getBusId(int cudaDev, int64_t *busId) {
  // On most systems, the PCI bus ID comes back as in the 0000:00:00.0
  // format. Still need to allocate proper space in case PCI domain goes
  // higher.
  char busIdStr[] = "00000000:00:00.0";
  WM_CUDA_CHECK(cudaDeviceGetPCIBusId(busIdStr, sizeof(busIdStr), cudaDev));
  busIdToInt64(busIdStr, busId);
}

static void getHostName(char *hostname, int maxlen, const char delim) {
  if (gethostname(hostname, maxlen) != 0) {
    strncpy(hostname, "unknown", maxlen);
    fprintf(stderr, "Get hostname failed.\n");
    abort();
  }
  int i = 0;
  while ((hostname[i] != delim) && (hostname[i] != '\0') && (i < maxlen - 1)) i++;
  hostname[i] = '\0';
}

/* Get the hostname and boot id
 * Equivalent of:
 *
 * $(hostname)$(cat /proc/sys/kernel/random/boot_id)
 *
 * This string can be overridden by using the WHOLEGRAPH_HOSTID env var.
 */
void getHostId(char *host_id, size_t len) {
  char *env_host_id;

  // Fall back is the full hostname if something fails
  (void) getHostName(host_id, len, '\0');
  int offset = strlen(host_id);

#define HOSTID_FILE "/proc/sys/kernel/random/boot_id"

  if ((env_host_id = getenv("WHOLEGRAPH_HOSTID")) != nullptr) {
    fprintf(stderr, "WHOLEGRAPH_HOSTID set by environment to %s\n", env_host_id);
    strncpy(host_id, env_host_id, len - 1);
    offset = strlen(env_host_id);
  } else {
    FILE *file = fopen(HOSTID_FILE, "r");
    if (file != nullptr) {
      char *p;
      if (fscanf(file, "%ms", &p) == 1) {
        strncpy(host_id + offset, p, len - offset - 1);
        offset += strlen(p);
        free(p);
      }
    }
    fclose(file);
  }

#undef HOSTID_FILE

  host_id[offset] = '\0';
}

static void fillInfo(BootstrapCommunicatorImpl *comm, struct WholeMemoryPeerInfo *info) {
  info->rank = comm->Rank();
  WM_CUDA_CHECK(cudaGetDevice(&info->cudaDev));
  getHostId(info->hostId, sizeof(info->hostId));
  info->pid = getpid();

  // Get the device MAJOR:MINOR of /dev/shm so we can use that
  // information to decide whether we can use SHM for inter-process
  // communication in a container environment
  struct stat statbuf {};
  WM_CHECK(stat("/dev/shm", &statbuf) == 0);
  info->shmDev = statbuf.st_dev;

  info->busId = comm->BusID();
}

void BootstrapCommunicatorImpl::CommInitRankSync() {
  WM_CUDA_CHECK(cudaGetDevice(&cudaDev));
  getBusId(cudaDev, &busId);

  struct WholeMemoryPeerInfo my_info {};
  bzero(&my_info, sizeof(my_info));
  fillInfo(this, &my_info);

  peerInfo = (struct WholeMemoryPeerInfo *) calloc(size_, sizeof(struct WholeMemoryPeerInfo));
  std::vector<const void *> send_ptrs(size_, &my_info);
  std::vector<void *> recv_ptrs(size_);
  std::vector<int> sr_size(size_, sizeof(WholeMemoryPeerInfo));
  for (int r = 0; r < size_; r++) {
    recv_ptrs[r] = &peerInfo[r];
  }
  AllToAllV(send_ptrs.data(), sr_size.data(), recv_ptrs.data(), sr_size.data());
  Synchronize();

  for (int i = 0; i < size_; i++) {
    if (strcmp(peerInfo[i].hostId, peerInfo[rank_].hostId) == 0) {
      // Rank is on same node
      if (intraNodeRanks == 0) intraNodeRank0 = i;
      if (i == rank_) intraNodeRank = intraNodeRanks;
      //intraNodeGlobalRanks[intraNodeRanks] = i;
      intraNodeRanks++;
      if (peerInfo[i].pid == peerInfo[rank_].pid) {
        // Rank is in same process
        if (intraProcRanks == 0) intraProcRank0 = i;
        if (i == rank_) intraProcRank = intraProcRanks;
        intraProcRanks++;
      }
    }
  }
  if (intraProcRank == -1 || intraProcRank0 == -1) {
    fprintf(stderr,
            "WholeMemory failed to determine intra proc ranks rank %d host %s pid %ld intraProcRank %d intraProcRanks %d intraProcRank0 %d\n",
            rank_,
            peerInfo[rank_].hostId,
            (int64_t) peerInfo[rank_].pid,
            intraProcRank,
            intraProcRanks,
            intraProcRank0);
    abort();
  }
  if (intraNodeRank == -1 || intraNodeRank0 == -1 || intraNodeRanks == 0) {
    fprintf(stderr,
            "WholeMemory failed to determine intra node ranks rank %d host %s pid %ld intraNodeRank %d intraNodeRanks %d intraNodeRank0 %d\n",
            rank_,
            peerInfo[rank_].hostId,
            (int64_t) peerInfo[rank_].pid,
            intraNodeRank,
            intraNodeRanks,
            intraNodeRank0);
    abort();
  }
  intraNodeRank0pid = peerInfo[intraNodeRank0].pid;
}

}// namespace whole_graph