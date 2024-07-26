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
#include "communicator.hpp"

#include <cstdlib>
#include <set>
#include <sys/stat.h>
#include <unistd.h>

#include <cuda.h>

#include <memory>
#include <raft/core/error.hpp>
#include <string>
#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

#include "cuda_macros.hpp"
#include "logger.hpp"
#include "memory_handle.hpp"
#include "system_info.hpp"
#include "wholememory/nccl_comms.hpp"

#ifdef WITH_NVSHMEM_SUPPORT

#include "nvshmem.h"
#include "nvshmemx.h"

extern bool nvshmemi_is_nvshmem_bootstrapped;

#endif

wholememory_comm_::wholememory_comm_(ncclComm_t nccl_comm,
                                     int num_ranks,
                                     int rank,
                                     cudaStream_t stream)
{
  world_rank = rank;
  world_size = num_ranks;
  WM_CUDA_CHECK(cudaGetDevice(&dev_id));
  comm_stream   = stream;
  raw_nccl_comm = nccl_comm;
  WM_CUDA_CHECK(cudaEventCreate(&cuda_event));
  raft_nccl_comm = new wholememory::nccl_comms(nccl_comm, num_ranks, rank, stream);
}

wholememory_comm_::~wholememory_comm_()
{
  delete raft_nccl_comm;
  if (cuda_event != nullptr) {
    cudaEventDestroy(cuda_event);
    cuda_event = nullptr;
  }
}

static ncclDataType_t get_nccl_dtype(const wholememory_dtype_t dtype)
{
  switch (dtype) {
    case WHOLEMEMORY_DT_FLOAT: return ncclFloat;
    case WHOLEMEMORY_DT_DOUBLE: return ncclDouble;
    case WHOLEMEMORY_DT_HALF: return ncclHalf;
    case WHOLEMEMORY_DT_INT8: return ncclChar;
    case WHOLEMEMORY_DT_INT: return ncclInt32;
    case WHOLEMEMORY_DT_INT64: return ncclInt64;
#if defined(__CUDA_BF16_TYPES_EXIST__)
    case WHOLEMEMORY_DT_BF16: return ncclBfloat16;
#endif
    default: WHOLEMEMORY_FAIL("Not supported dtype.");
  }
}

static ncclDataType_t get_nccl_dtype_same_size(const wholememory_dtype_t dtype)
{
  switch (dtype) {
    case WHOLEMEMORY_DT_INT8: return ncclChar;
    case WHOLEMEMORY_DT_INT16: return ncclHalf;
    case WHOLEMEMORY_DT_BF16: return ncclHalf;
    case WHOLEMEMORY_DT_HALF: return ncclHalf;
    case WHOLEMEMORY_DT_INT: return ncclInt32;
    case WHOLEMEMORY_DT_FLOAT: return ncclFloat;
    case WHOLEMEMORY_DT_INT64: return ncclInt64;
    case WHOLEMEMORY_DT_DOUBLE: return ncclDouble;
    default: WHOLEMEMORY_FAIL("Not supported dtype.");
  }
}

void wholememory_comm_::barrier() const { raft_nccl_comm->barrier(); }

void wholememory_comm_::abort() const { raft_nccl_comm->abort(); }

void wholememory_comm_::allreduce(const void* sendbuff,
                                  void* recvbuff,
                                  size_t count,
                                  wholememory_dtype_t datatype,
                                  ncclRedOp_t op,
                                  cudaStream_t stream) const
{
  raft_nccl_comm->allreduce(sendbuff, recvbuff, count, get_nccl_dtype(datatype), op, stream);
}

void wholememory_comm_::host_allreduce(const void* sendbuff,
                                       void* recvbuff,
                                       size_t count,
                                       wholememory_dtype_t datatype,
                                       ncclRedOp_t op) const
{
  raft_nccl_comm->host_allreduce(sendbuff, recvbuff, count, get_nccl_dtype(datatype), op);
}

void wholememory_comm_::bcast(
  void* buff, size_t count, wholememory_dtype_t datatype, int root, cudaStream_t stream) const
{
  raft_nccl_comm->bcast(buff, count, get_nccl_dtype_same_size(datatype), root, stream);
}

void wholememory_comm_::bcast(const void* sendbuff,
                              void* recvbuff,
                              size_t count,
                              wholememory_dtype_t datatype,
                              int root,
                              cudaStream_t stream) const
{
  raft_nccl_comm->bcast(
    sendbuff, recvbuff, count, get_nccl_dtype_same_size(datatype), root, stream);
}

void wholememory_comm_::host_bcast(
  const void* sendbuff, void* recvbuff, size_t count, wholememory_dtype_t datatype, int root) const
{
  raft_nccl_comm->host_bcast(sendbuff, recvbuff, count, get_nccl_dtype_same_size(datatype), root);
}

void wholememory_comm_::host_bcast(void* buff,
                                   size_t count,
                                   wholememory_dtype_t datatype,
                                   int root) const
{
  raft_nccl_comm->host_bcast(buff, count, get_nccl_dtype_same_size(datatype), root);
}

void wholememory_comm_::reduce(const void* sendbuff,
                               void* recvbuff,
                               size_t count,
                               wholememory_dtype_t datatype,
                               ncclRedOp_t op,
                               int root,
                               cudaStream_t stream) const
{
  raft_nccl_comm->reduce(sendbuff, recvbuff, count, get_nccl_dtype(datatype), op, root, stream);
}

void wholememory_comm_::host_reduce(const void* sendbuff,
                                    void* recvbuff,
                                    size_t count,
                                    wholememory_dtype_t datatype,
                                    ncclRedOp_t op,
                                    int root) const
{
  raft_nccl_comm->host_reduce(sendbuff, recvbuff, count, get_nccl_dtype(datatype), op, root);
}

void wholememory_comm_::allgather(const void* sendbuff,
                                  void* recvbuff,
                                  size_t sendcount,
                                  wholememory_dtype_t datatype,
                                  cudaStream_t stream) const
{
  raft_nccl_comm->allgather(
    sendbuff, recvbuff, sendcount, get_nccl_dtype_same_size(datatype), stream);
}

void wholememory_comm_::host_allgather(const void* sendbuff,
                                       void* recvbuff,
                                       size_t sendcount,
                                       wholememory_dtype_t datatype) const
{
  raft_nccl_comm->host_allgather(sendbuff, recvbuff, sendcount, get_nccl_dtype_same_size(datatype));
}

void wholememory_comm_::allgatherv(const void* sendbuf,
                                   void* recvbuf,
                                   const size_t* recvcounts,
                                   const size_t* displs,
                                   wholememory_dtype_t datatype,
                                   cudaStream_t stream) const
{
  raft_nccl_comm->allgatherv(
    sendbuf, recvbuf, recvcounts, displs, get_nccl_dtype_same_size(datatype), stream);
}

void wholememory_comm_::host_allgatherv(const void* sendbuf,
                                        void* recvbuf,
                                        const size_t* recvcounts,
                                        const size_t* displs,
                                        wholememory_dtype_t datatype) const
{
  raft_nccl_comm->host_allgatherv(
    sendbuf, recvbuf, recvcounts, displs, get_nccl_dtype_same_size(datatype));
}

void wholememory_comm_::gather(const void* sendbuff,
                               void* recvbuff,
                               size_t sendcount,
                               wholememory_dtype_t datatype,
                               int root,
                               cudaStream_t stream) const
{
  raft_nccl_comm->gather(
    sendbuff, recvbuff, sendcount, get_nccl_dtype_same_size(datatype), root, stream);
}

void wholememory_comm_::host_gather(const void* sendbuff,
                                    void* recvbuff,
                                    size_t sendcount,
                                    wholememory_dtype_t datatype,
                                    int root) const
{
  raft_nccl_comm->host_gather(
    sendbuff, recvbuff, sendcount, get_nccl_dtype_same_size(datatype), root);
}

void wholememory_comm_::gatherv(const void* sendbuff,
                                void* recvbuff,
                                size_t sendcount,
                                const size_t* recvcounts,
                                const size_t* displs,
                                wholememory_dtype_t datatype,
                                int root,
                                cudaStream_t stream) const
{
  raft_nccl_comm->gatherv(sendbuff,
                          recvbuff,
                          sendcount,
                          recvcounts,
                          displs,
                          get_nccl_dtype_same_size(datatype),
                          root,
                          stream);
}

void wholememory_comm_::reducescatter(const void* sendbuff,
                                      void* recvbuff,
                                      size_t recvcount,
                                      wholememory_dtype_t datatype,
                                      ncclRedOp_t op,
                                      cudaStream_t stream) const
{
  raft_nccl_comm->reducescatter(
    sendbuff, recvbuff, recvcount, get_nccl_dtype(datatype), op, stream);
}

void wholememory_comm_::alltoall(const void* sendbuff,
                                 void* recvbuff,
                                 size_t sendcount,
                                 wholememory_dtype_t datatype,
                                 cudaStream_t stream) const
{
  raft_nccl_comm->alltoall(
    sendbuff, recvbuff, sendcount, get_nccl_dtype_same_size(datatype), stream);
}

void wholememory_comm_::host_alltoall(const void* sendbuff,
                                      void* recvbuff,
                                      size_t sendcount,
                                      wholememory_dtype_t datatype) const
{
  raft_nccl_comm->host_alltoall(sendbuff, recvbuff, sendcount, get_nccl_dtype_same_size(datatype));
}

void wholememory_comm_::alltoallv(const void* sendbuff,
                                  void* recvbuff,
                                  const size_t* sendcounts,
                                  const size_t* senddispls,
                                  const size_t* recvcounts,
                                  const size_t* recvdispls,
                                  wholememory_dtype_t datatype,
                                  cudaStream_t stream) const
{
  raft_nccl_comm->alltoallv(sendbuff,
                            recvbuff,
                            sendcounts,
                            senddispls,
                            recvcounts,
                            recvdispls,
                            get_nccl_dtype_same_size(datatype),
                            stream);
}

wholememory_error_code_t wholememory_comm_::sync_stream(cudaStream_t stream) const
{
  return raft_nccl_comm->sync_stream(stream);
}

wholememory_error_code_t wholememory_comm_::sync_stream() const
{
  return raft_nccl_comm->sync_stream();
}

// if a thread is sending & receiving at the same time, use device_sendrecv to avoid deadlock
void wholememory_comm_::device_send(const void* send_buf,
                                    size_t send_size,
                                    int dest,
                                    cudaStream_t stream) const
{
  raft_nccl_comm->device_send(send_buf, send_size, dest, stream);
}

// if a thread is sending & receiving at the same time, use device_sendrecv to avoid deadlock
void wholememory_comm_::device_recv(void* recv_buf,
                                    size_t recv_size,
                                    int source,
                                    cudaStream_t stream) const
{
  raft_nccl_comm->device_recv(recv_buf, recv_size, source, stream);
}

void wholememory_comm_::device_sendrecv(const void* sendbuf,
                                        size_t sendsize,
                                        int dest,
                                        void* recvbuf,
                                        size_t recvsize,
                                        int source,
                                        cudaStream_t stream) const
{
  raft_nccl_comm->device_sendrecv(sendbuf, sendsize, dest, recvbuf, recvsize, source, stream);
}

void wholememory_comm_::device_multicast_sendrecv(const void* sendbuf,
                                                  std::vector<size_t> const& sendsizes,
                                                  std::vector<size_t> const& sendoffsets,
                                                  std::vector<int> const& dests,
                                                  void* recvbuf,
                                                  std::vector<size_t> const& recvsizes,
                                                  std::vector<size_t> const& recvoffsets,
                                                  std::vector<int> const& sources,
                                                  cudaStream_t stream) const
{
  raft_nccl_comm->device_multicast_sendrecv(
    sendbuf, sendsizes, sendoffsets, dests, recvbuf, recvsizes, recvoffsets, sources, stream);
}

bool wholememory_comm_::is_intranode() const { return intra_node_rank_num == world_size; }

bool wholememory_comm_::is_intra_mnnvl() const { return support_mnnvl; }
bool wholememory_comm_::support_type_location(wholememory_memory_type_t memory_type,
                                              wholememory_memory_location_t memory_location) const
{
  if (memory_location == WHOLEMEMORY_ML_HOST) {
    if (is_intranode() || memory_type == WHOLEMEMORY_MT_DISTRIBUTED) return true;
    return is_intra_mnnvl() && SupportEGM();
  } else if (memory_location == WHOLEMEMORY_ML_DEVICE) {
    if (memory_type == WHOLEMEMORY_MT_DISTRIBUTED) return true;
    if (is_intranode()) {
      return DevicesCanAccessP2P(&local_gpu_ids[0], intra_node_rank_num);
    } else {
      return DevicesCanAccessP2P(&local_gpu_ids[0], intra_node_rank_num) && is_intra_mnnvl();
    }
  } else {
    return false;
  }
}

void wholememory_comm_::group_start() const { raft_nccl_comm->group_start(); }

void wholememory_comm_::group_end() const { raft_nccl_comm->group_end(); }

namespace wholememory {

static std::mutex comm_mu;
static std::map<int, wholememory_comm_t> communicator_map;

#ifdef WITH_NVSHMEM_SUPPORT
static bool nvshmem_has_initalized = false;
#endif

enum wm_comm_op : int32_t {
  WM_COMM_OP_STARTING = 0xEEC0EE,
  WM_COMM_OP_EXCHANGE_ID,
  WM_COMM_OP_WAIT_CREATE_TEMPDIR,
  WM_COMM_OP_DESTROY_ALL_HANDLES,
  WM_COMM_OP_DESTROY_COMM,
};

wholememory_error_code_t create_unique_id(wholememory_unique_id_t* unique_id) noexcept
{
  ncclUniqueId id;
  WHOLEMEMORY_CHECK_NOTHROW(sizeof(ncclUniqueId) <= sizeof(wholememory_unique_id_t));
  WHOLEMEMORY_CHECK_NOTHROW(ncclGetUniqueId(&id) == ncclSuccess);
  memcpy(unique_id->internal, id.internal, sizeof(ncclUniqueId));
  return WHOLEMEMORY_SUCCESS;
}

static constexpr int HOST_NAME_MAX_LENGTH = 1024;
static constexpr int BOOT_ID_MAX_LENGTH   = 1024;

struct host_info {
  char host_name[HOST_NAME_MAX_LENGTH];
  char boot_id[BOOT_ID_MAX_LENGTH];
  dev_t shm_dev;
  bool operator==(const host_info& rhs) const
  {
    if (std::strncmp(host_name, rhs.host_name, HOST_NAME_MAX_LENGTH) != 0) return false;
    if (std::strncmp(boot_id, rhs.boot_id, BOOT_ID_MAX_LENGTH) != 0) return false;
    if (shm_dev != rhs.shm_dev) return false;
    return true;
  }
} __attribute__((aligned(128)));

struct rank_info {
  host_info rank_host_info;
  pid_t pid;
  int rank;
  int size;
  int gpu_id;
// MNNVL support
#if CUDA_VERSION >= 12030
  nvmlGpuFabricInfo_t fabric_info;
#endif
};

static void get_host_name(char* hostname, int maxlen, const char delim)
{
  if (gethostname(hostname, maxlen) != 0) {
    strncpy(hostname, "unknown", maxlen);
    WHOLEMEMORY_FATAL("gethostname failed.");
  }
  int i = 0;
  while ((hostname[i] != delim) && (hostname[i] != '\0') && (i < maxlen - 1))
    i++;
  hostname[i] = '\0';
}

/* Get the hostname and boot id
 * Equivalent of:
 *
 * $(hostname)$(cat /proc/sys/kernel/random/boot_id)
 *
 * This string can be overridden by using the WHOLEGRAPH_HOSTID env var.
 */
void get_boot_id(char* host_id, size_t len)
{
  char* env_host_id;
  int offset = 0;

#define BOOTID_FILE "/proc/sys/kernel/random/boot_id"

  if ((env_host_id = getenv("WHOLEMEMORY_BOOTID")) != nullptr) {
    WHOLEMEMORY_LOG(LEVEL_INFO, "WHOLEMEMORY_BOOTID set by environment to %s", env_host_id);
    strncpy(host_id, env_host_id, len - 1);
    offset = strlen(env_host_id);
  } else {
    FILE* file = fopen(BOOTID_FILE, "r");
    if (file != nullptr) {
      char* p;
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

void get_shm_devid(dev_t* shm_dev)
{
  struct stat statbuf {};
  WHOLEMEMORY_CHECK(stat("/dev/shm", &statbuf) == 0);
  *shm_dev = statbuf.st_dev;
}

void get_host_info(host_info* phi)
{
  bzero(phi, sizeof(host_info));
  get_host_name(&phi->host_name[0], HOST_NAME_MAX_LENGTH, '\0');
  get_boot_id(&phi->boot_id[0], BOOT_ID_MAX_LENGTH);
  get_shm_devid(&phi->shm_dev);
}

bool comm_support_mnnvl(wholememory_comm_t wm_comm, const std::unique_ptr<rank_info[]>& p_rank_info)
{
#if CUDA_VERSION >= 12030
  int flag = 0;
  CUdevice currentDev;
  WM_CU_CHECK_NO_THROW(cuDeviceGet(&currentDev, wm_comm->dev_id));
  // Ignore error if CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED is not supported
  WM_CU_CHECK_NO_THROW(
    cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, currentDev));
  if (!flag) return false;

  nvmlGpuFabricInfo_t gpuFabricInfo;
  WHOLEMEMORY_CHECK_NOTHROW(wholememory::GetGpuFabricInfo(wm_comm->dev_id, &gpuFabricInfo) ==
                            WHOLEMEMORY_SUCCESS);

  if (gpuFabricInfo.state != NVML_GPU_FABRIC_STATE_COMPLETED) { return false; }

  // Check that all ranks have initialized the fabric fully
  for (int i = 0; i < wm_comm->world_rank; i++) {
    if (p_rank_info.get()[i].fabric_info.state != NVML_GPU_FABRIC_STATE_COMPLETED) return 0;
  }

  return GetCudaCompCap() >= 90;
#else

  return 0;
#endif
};

void exchange_rank_info(wholememory_comm_t wm_comm)
{
  rank_info ri;
  get_host_info(&ri.rank_host_info);
  ri.rank                           = wm_comm->world_rank;
  ri.size                           = wm_comm->world_size;
  ri.pid                            = getpid();
  ri.gpu_id                         = wm_comm->dev_id;
  wm_comm->clique_info.is_in_clique = 0;

#if CUDA_VERSION >= 12030
  memset(&ri.fabric_info, 0, sizeof(ri.fabric_info));
  WHOLEMEMORY_CHECK_NOTHROW(GetGpuFabricInfo(wm_comm->dev_id, &ri.fabric_info) ==
                            WHOLEMEMORY_SUCCESS);

  //    // A zero UUID means we don't have MNNVL fabric info
  if (((((long*)ri.fabric_info.clusterUuid)[0] | ((long*)ri.fabric_info.clusterUuid)[1]) == 0)) {
    wm_comm->clique_info.is_in_clique = 0;

  } else {
    wm_comm->clique_info.is_in_clique = 1;
  }

#endif

  std::unique_ptr<rank_info[]> p_rank_info(new rank_info[ri.size]);
  wm_comm->host_allgather(&ri, p_rank_info.get(), sizeof(rank_info), WHOLEMEMORY_DT_INT8);
  wm_comm->intra_node_first_rank = -1;
  wm_comm->intra_node_rank_num   = 0;
  wm_comm->intra_node_rank       = -1;

  wm_comm->clique_info.clique_first_rank = -1;
  wm_comm->clique_info.clique_rank       = -1;
  wm_comm->clique_info.clique_rank_num   = 0;

  std::set<std::string> clique_uuids{};

  for (int r = 0; r < wm_comm->world_size; r++) {
    WHOLEMEMORY_CHECK(r == p_rank_info.get()[r].rank);
    if (ri.rank_host_info == p_rank_info.get()[r].rank_host_info) {
      if (r == wm_comm->world_rank) { wm_comm->intra_node_rank = wm_comm->intra_node_rank_num; }
      if (wm_comm->intra_node_rank_num == 0) {
        wm_comm->intra_node_first_rank_pid = p_rank_info.get()[r].pid;
        wm_comm->intra_node_first_rank     = r;
      }
      wm_comm->local_gpu_ids[wm_comm->intra_node_rank_num] = p_rank_info.get()[r].gpu_id;
      wm_comm->intra_node_rank_num++;
    }

#if CUDA_VERSION >= 12030

    if ((memcmp(ri.fabric_info.clusterUuid,
                p_rank_info.get()[r].fabric_info.clusterUuid,
                NVML_GPU_FABRIC_UUID_LEN) == 0) &&
        (ri.fabric_info.cliqueId == p_rank_info.get()[r].fabric_info.cliqueId)) {
      if (r == wm_comm->world_rank) {
        wm_comm->clique_info.clique_rank = wm_comm->clique_info.clique_rank_num;
      }
      if (wm_comm->clique_info.clique_rank_num == 0) { wm_comm->clique_info.clique_first_rank = r; }
      wm_comm->clique_info.clique_rank_num++;
    }
    clique_uuids.insert(
      std::string(reinterpret_cast<const char*>(p_rank_info.get()[r].fabric_info.clusterUuid),
                  NVML_GPU_FABRIC_UUID_LEN));

#endif
  }

#if CUDA_VERSION >= 12030
  wm_comm->clique_info.clique_num = clique_uuids.size();

  std::string uuid = std::string(reinterpret_cast<const char*>(ri.fabric_info.clusterUuid),
                                 NVML_GPU_FABRIC_UUID_LEN);
  int id           = 0;
  for (auto clique_uuid : clique_uuids) {
    if (clique_uuid == uuid) { wm_comm->clique_info.clique_id = id; }
    id++;
  }

  wm_comm->support_mnnvl = (comm_support_mnnvl(wm_comm, p_rank_info)) &&
                           (wm_comm->clique_info.clique_rank_num == wm_comm->world_size);

#endif
}

void negotiate_communicator_id_locked(wholememory_comm_t wm_comm)
{
  WM_COMM_CHECK_ALL_SAME(wm_comm, WM_COMM_OP_EXCHANGE_ID);
  int id        = 0;
  bool all_same = false;
  std::vector<int> rank_ids(wm_comm->world_size);
  while (!all_same) {
    while (communicator_map.find(id) != communicator_map.end())
      id++;
    wm_comm->host_allgather(&id, rank_ids.data(), 1, WHOLEMEMORY_DT_INT);
    int max_id = -1;
    all_same   = true;
    for (int i = 0; i < wm_comm->world_size; i++) {
      if (rank_ids[i] > max_id) max_id = rank_ids[i];
      if (rank_ids[i] != id) all_same = false;
    }
    id = max_id;
  }
  wm_comm->comm_id = id;
  communicator_map.insert(std::pair<int, wholememory_comm_t>(id, wm_comm));
}

std::string get_temporary_directory_path(wholememory_comm_t comm)
{
  const char* sock_prefix            = getenv("WHOLEMEMORY_TMPPREFIX");
  std::string wholememory_prefix_str = "/tmp/wmtmp";
  if (sock_prefix != nullptr) { wholememory_prefix_str = sock_prefix; }
  char temp_path_buffer[128];
  (void)std::snprintf(
    temp_path_buffer, 128, "_comm_id_%d_pid_%d", comm->comm_id, comm->intra_node_first_rank_pid);
  wholememory_prefix_str.append(temp_path_buffer);
  return wholememory_prefix_str;
}

std::string get_shm_prefix(wholememory_comm_t comm)
{
  char temp_path_buffer[128];
  (void)std::snprintf(temp_path_buffer,
                      128,
                      "wmshm_comm_id_%d_pid_%d",
                      comm->comm_id,
                      comm->intra_node_first_rank_pid);
  std::string wholememory_prefix_str = temp_path_buffer;
  return wholememory_prefix_str;
}

void maybe_create_temp_dir(wholememory_comm_t wm_comm)
{
  if (!is_intranode_communicator(wm_comm)) return;
  if (wm_comm->intra_node_rank == 0) {
    auto temp_path = get_temporary_directory_path(wm_comm);
    WHOLEMEMORY_CHECK(mkdir(temp_path.c_str(), 0700) == 0);
  }
  WM_COMM_CHECK_ALL_SAME(wm_comm, WM_COMM_OP_WAIT_CREATE_TEMPDIR);
}

void maybe_remove_temp_dir(wholememory_comm_t wm_comm)
{
  if (!is_intranode_communicator(wm_comm)) return;
  if (wm_comm->intra_node_rank == 0) {
    auto temp_path = get_temporary_directory_path(wm_comm);
    WHOLEMEMORY_CHECK(remove(get_temporary_directory_path(wm_comm).c_str()) == 0);
  }
}

static size_t get_alloc_granularity(int dev_id)
{
  size_t granularity = 0;
  CUmemAllocationProp prop;
  memset(&prop, 0, sizeof(prop));
  prop.type                              = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.requestedHandleTypes              = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  prop.location.type                     = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.allocFlags.compressionType        = CU_MEM_ALLOCATION_COMP_NONE;
  CUmemAllocationGranularity_flags flags = CU_MEM_ALLOC_GRANULARITY_RECOMMENDED;
  prop.location.id                       = dev_id;
  WM_CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, flags));
  return granularity;
}

void determine_alloc_granularity(wholememory_comm_t comm)
{
  size_t granularity = get_alloc_granularity(comm->dev_id);
  std::vector<size_t> all_granularitys(comm->world_size);
  comm->host_allgather(&granularity, all_granularitys.data(), 1, WHOLEMEMORY_DT_INT64);
  size_t max_granularity = granularity;
  for (auto g : all_granularitys) {
    if (g > max_granularity) { max_granularity = g; }
  }
  comm->alloc_granularity = max_granularity;
}

wholememory_error_code_t create_communicator(wholememory_comm_t* comm,
                                             wholememory_unique_id_t unique_id,
                                             int world_rank,
                                             int world_size) noexcept
{
  try {
    std::unique_lock<std::mutex> mlock(comm_mu);
    ncclComm_t nccl_comm;
    WHOLEMEMORY_CHECK(
      ncclCommInitRank(&nccl_comm, world_size, (ncclUniqueId&)unique_id, world_rank) ==
      ncclSuccess);
    cudaStream_t cuda_stream;
    WM_CUDA_CHECK(cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking));
    auto* wm_comm = new wholememory_comm_(nccl_comm, world_size, world_rank, cuda_stream);
    *comm         = wm_comm;
    WM_COMM_CHECK_ALL_SAME(wm_comm, WM_COMM_OP_STARTING);

    exchange_rank_info(wm_comm);

    negotiate_communicator_id_locked(wm_comm);

    maybe_create_temp_dir(wm_comm);

    determine_alloc_granularity(wm_comm);

    return WHOLEMEMORY_SUCCESS;
  } catch (const wholememory::cu_error& wce) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", wce.what());
  } catch (const wholememory::cuda_error& wce) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", wce.what());
  } catch (const wholememory::logic_error& wle) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", wle.what());
  } catch (const raft::exception& re) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", re.what());
  } catch (const std::bad_alloc& sba) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", sba.what());
  } catch (...) {
    WHOLEMEMORY_FAIL_NOTHROW("Unknown exception.");
  }
}

/**
 *
 * Ranks which pass the same color value will be part of the same group; color must be a
 * non-negative value. If it is passed as WHOLEMEMORY_SPLIT_NOCOLOR, it means that the rank will not
 * be part of any group, therefore returning NULL as newcomm. The value of key will determine the
 * rank order, and the smaller key means the smaller rank in new communicator. If keys are equal
 * between ranks, then the rank in the original communicator will be used to order ranks.
 *
 */

wholememory_error_code_t split_communicator(wholememory_comm_t* new_comm,
                                            wholememory_comm_t parent_comm,
                                            int color,
                                            int key) noexcept
{
  try {
    std::unique_lock<std::mutex> mlock(comm_mu);

    WHOLEMEMORY_EXPECTS(wholememory_communicator_is_bind_to_nvshmem(parent_comm) == false,
                        "Cannot split a communicator that is already bind to NVSHMEM");

    ncclComm_t nccl_comm = parent_comm->raft_nccl_comm->raw_nccl_comm();
    WHOLEMEMORY_CHECK(nccl_comm != nullptr);
    ncclComm_t new_nccl_comm;
    WHOLEMEMORY_CHECK(ncclCommSplit(nccl_comm, color, key, &new_nccl_comm, NULL) == ncclSuccess);
    cudaStream_t cuda_stream;
    WM_CUDA_CHECK(cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking));
    if (new_nccl_comm == NULL) {
      *new_comm = nullptr;
      return WHOLEMEMORY_SUCCESS;
    }
    int new_rank;
    int new_size;
    WHOLEMEMORY_CHECK(ncclCommUserRank(new_nccl_comm, &new_rank) == ncclSuccess);
    WHOLEMEMORY_CHECK(ncclCommCount(new_nccl_comm, &new_size) == ncclSuccess);

    auto* wm_comm = new wholememory_comm_(new_nccl_comm, new_size, new_rank, cuda_stream);
    *new_comm     = wm_comm;
    WM_COMM_CHECK_ALL_SAME(wm_comm, WM_COMM_OP_STARTING);

    exchange_rank_info(wm_comm);

    negotiate_communicator_id_locked(wm_comm);

    maybe_create_temp_dir(wm_comm);

    determine_alloc_granularity(wm_comm);

    return WHOLEMEMORY_SUCCESS;
  } catch (const wholememory::cu_error& wce) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", wce.what());
  } catch (const wholememory::cuda_error& wce) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", wce.what());
  } catch (const wholememory::logic_error& wle) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", wle.what());
  } catch (const raft::exception& re) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", re.what());
  } catch (const std::bad_alloc& sba) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", sba.what());
  } catch (...) {
    WHOLEMEMORY_FAIL_NOTHROW("Unknown exception.");
  }
}

void destroy_all_wholememory(wholememory_comm_t comm) noexcept
{
  try {
    std::unique_lock<std::mutex> mlock(comm->mu);
    WM_COMM_CHECK_ALL_SAME(comm, WM_COMM_OP_DESTROY_ALL_HANDLES);
    WM_COMM_CHECK_ALL_SAME(comm, comm->wholememory_map.size());
    while (!comm->wholememory_map.empty()) {
      auto id_wm = comm->wholememory_map.begin();
      destroy_wholememory_with_comm_locked(id_wm->second);
    }
  } catch (const wholememory::logic_error& wle) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", wle.what());
  } catch (const wholememory::cuda_error& wce) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", wce.what());
  } catch (const raft::exception& re) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", re.what());
  } catch (...) {
    WHOLEMEMORY_FAIL_NOTHROW("Unknown exception.");
  }
}

wholememory_error_code_t destroy_communicator_locked(wholememory_comm_t comm) noexcept
{
  try {
    if (communicator_map.find(comm->comm_id) == communicator_map.end()) {
      return WHOLEMEMORY_INVALID_INPUT;
    }

    destroy_all_wholememory(comm);
#ifdef WITH_NVSHMEM_SUPPORT
    if (comm->bind_to_nvshmem) { finalize_nvshmem_locked(comm); }
#endif
    WM_COMM_CHECK_ALL_SAME(comm, WM_COMM_OP_DESTROY_COMM);
    communicator_map.erase(comm->comm_id);
    auto* raw_nccl_comm = comm->raw_nccl_comm;
    auto* cuda_stream   = comm->comm_stream;

    maybe_remove_temp_dir(comm);

    delete comm;
    WHOLEMEMORY_CHECK(ncclCommDestroy(raw_nccl_comm) == ncclSuccess);
    WM_CUDA_CHECK(cudaStreamDestroy(cuda_stream));

    return WHOLEMEMORY_SUCCESS;
  } catch (const wholememory::cuda_error& wce) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", wce.what());
  } catch (const wholememory::logic_error& wle) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", wle.what());
  } catch (const raft::exception& re) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", re.what());
  } catch (...) {
    WHOLEMEMORY_FAIL_NOTHROW("Unknown exception.");
  }
}

wholememory_error_code_t destroy_communicator(wholememory_comm_t comm) noexcept
{
  std::unique_lock<std::mutex> mlock(comm_mu);
  return destroy_communicator_locked(comm);
}

wholememory_error_code_t communicator_support_type_location(
  wholememory_comm_t comm,
  wholememory_memory_type_t memory_type,
  wholememory_memory_location_t memory_location) noexcept
{
  return comm->support_type_location(memory_type, memory_location) ? WHOLEMEMORY_SUCCESS
                                                                   : WHOLEMEMORY_NOT_SUPPORTED;
}

wholememory_error_code_t destroy_all_communicators() noexcept
{
  std::unique_lock<std::mutex> mlock(comm_mu);
  while (!communicator_map.empty()) {
    auto id_comm = communicator_map.begin();
    destroy_communicator_locked(id_comm->second);
  }
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t communicator_get_rank(int* rank, wholememory_comm_t comm) noexcept
{
  *rank = comm->world_rank;
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t communicator_get_size(int* size, wholememory_comm_t comm) noexcept
{
  *size = comm->world_size;
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t communicator_get_local_size(int* local_size,
                                                     wholememory_comm_t comm) noexcept
{
  *local_size = comm->intra_node_rank_num;
  return WHOLEMEMORY_SUCCESS;
}

// wholememory_error_code_t communicator_get_clique_rank(int* clique_rank,
//                                                       wholememory_comm_t comm) noexcept
// {
//   *clique_rank = comm->clique_rank;
//   return WHOLEMEMORY_SUCCESS;
// }

// wholememory_error_code_t communicator_get_clique_size(int* clique_size,
//                                                       wholememory_comm_t comm) noexcept
// {
//   *clique_size = comm->clique_rank_num;
//   return WHOLEMEMORY_SUCCESS;
// }

wholememory_error_code_t communicator_get_clique_info(clique_info_t* clique_info,
                                                      wholememory_comm_t comm) noexcept
{
  *clique_info = comm->clique_info;
  return WHOLEMEMORY_SUCCESS;
}

bool communicator_is_bind_to_nvshmem(wholememory_comm_t comm) noexcept
{
#ifdef WITH_NVSHMEM_SUPPORT
  return comm->bind_to_nvshmem;
#else
  return false;
#endif
}

wholememory_distributed_backend_t communicator_get_distributed_backend(
  wholememory_comm_t comm) noexcept
{
  return comm->distributed_backend;
}

void communicator_barrier(wholememory_comm_t comm)
{
  try {
    comm->barrier();
  } catch (const wholememory::cuda_error& wce) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", wce.what());
  } catch (const wholememory::logic_error& wle) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", wle.what());
  } catch (const raft::exception& re) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", re.what());
  } catch (...) {
    WHOLEMEMORY_FAIL_NOTHROW("Unknown exception.");
  }
}

bool is_intranode_communicator(wholememory_comm_t comm) noexcept { return comm->is_intranode(); }

bool is_intra_mnnvl_communicator(wholememory_comm_t comm) noexcept
{
  return comm->is_intra_mnnvl();
}

#ifdef WITH_NVSHMEM_SUPPORT
wholememory_error_code_t init_nvshmem_with_comm(wholememory_comm_t comm) noexcept
{
  try {
    std::unique_lock<std::mutex> mlock(comm_mu);

    WHOLEMEMORY_CHECK(comm != nullptr);
    WHOLEMEMORY_EXPECTS(comm->bind_to_nvshmem == true,
                        "The distributed_backend should be WHOLEMEMORY_DB_NVSHMEM.");
    if (nvshmem_has_initalized) return WHOLEMEMORY_SUCCESS;

    WM_COMM_CHECK_ALL_SAME(comm, nvshmem_has_initalized);

    WHOLEMEMORY_EXPECTS(nvshmem_has_initalized == false,
                        "A wholememory_comm_t has been used to init nvshmem. To init nvshmem with "
                        "other wholememory_coomm_t, call finalize_nvshmem first.");

    auto set_env_if_not_exist = [](std::string_view env_name, std::string_view value) {
      if (getenv(env_name.data()) == nullptr || strcmp(getenv(env_name.data()), "") == 0) {
        setenv(env_name.data(), value.data(), 1);
      }
      if (strcmp(getenv(env_name.data()), value.data()) != 0) {
        WHOLEMEMORY_WARN("When using wholegraph, the environment variable %s is best set to %s",
                         env_name.data(),
                         value.data());
      }
    };
    set_env_if_not_exist("NVSHMEM_BOOTSTRAP", "mpi");
    set_env_if_not_exist("NVSHMEM_BOOTSTRAP_MPI_PLUGIN", "libnvshmem_wholememory_bootstrap.so");
    nvshmemx_init_attr_t attr;
    attr.mpi_comm = &comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    WHOLEMEMORY_CHECK(nvshmem_my_pe() == comm->world_rank);
    WHOLEMEMORY_CHECK(nvshmem_n_pes() == comm->world_size);
    nvshmem_has_initalized = true;
    comm->bind_to_nvshmem  = true;

    return WHOLEMEMORY_SUCCESS;

  } catch (const wholememory::logic_error& wle) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", wle.what());
  } catch (const wholememory::cuda_error& wce) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", wce.what());
  } catch (const raft::exception& re) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", re.what());
  } catch (...) {
    WHOLEMEMORY_FAIL_NOTHROW("Unknown exception.");
  }
}

wholememory_error_code_t finalize_nvshmem_locked(wholememory_comm_t comm) noexcept
{
  try {
    WHOLEMEMORY_CHECK(comm != nullptr);

    WM_COMM_CHECK_ALL_SAME(comm, nvshmem_has_initalized);
    WHOLEMEMORY_EXPECTS(comm->bind_to_nvshmem == true,
                        "The wholememory_comm_t must be the one previously used to init nvshmem.");
    WHOLEMEMORY_EXPECTS(
      nvshmem_has_initalized == true,
      "To call finalize_nvshmem,please use wholememory_comm_t to init nvshmem first.");
    // destroy all memory allocated by nvshmem
    nvshmem_finalize();
    nvshmemi_is_nvshmem_bootstrapped = false;  // so we can bootstrap again.
    nvshmem_has_initalized           = false;
    comm->bind_to_nvshmem            = false;
    return WHOLEMEMORY_SUCCESS;

  } catch (const wholememory::logic_error& wle) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", wle.what());
  } catch (const wholememory::cuda_error& wce) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", wce.what());
  } catch (const raft::exception& re) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", re.what());
  } catch (...) {
    WHOLEMEMORY_FAIL_NOTHROW("Unknown exception.");
  }
}

#endif

wholememory_error_code_t communicator_set_distributed_backend(
  wholememory_comm_t comm, wholememory_distributed_backend_t distributed_backend) noexcept
{
  try {
    std::unique_lock<std::mutex> mlock(comm_mu);

    WHOLEMEMORY_CHECK(comm != nullptr);
    WM_COMM_CHECK_ALL_SAME(comm, distributed_backend);

    for (auto&& [id, handle] : comm->wholememory_map) {
      WHOLEMEMORY_EXPECTS(wholememory_get_memory_type(handle) != WHOLEMEMORY_MT_DISTRIBUTED,
                          "Please set distributed_backend before creating any whole_memory with "
                          "distributed memory type if need to change distriubted_backend");
    }
    comm->distributed_backend = distributed_backend;
#ifdef WITH_NVSHMEM_SUPPORT
    if (distributed_backend == WHOLEMEMORY_DB_NVSHMEM) { comm->bind_to_nvshmem = true; }
#endif
    return WHOLEMEMORY_SUCCESS;
  } catch (const wholememory::logic_error& wle) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", wle.what());
  } catch (const wholememory::cuda_error& wce) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", wce.what());
  } catch (const raft::exception& re) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", re.what());
  } catch (...) {
    WHOLEMEMORY_FAIL_NOTHROW("Unknown exception.");
  }
}
}  // namespace wholememory
