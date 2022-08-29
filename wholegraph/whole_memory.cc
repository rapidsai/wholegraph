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
#include "whole_memory.h"
#include "whole_chunked_memory.h"
#include "whole_nccl_memory.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <math.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "macros.h"
#include "whole_memory_communicator.h"

namespace whole_graph {

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct PeerMemorySharableHandle {
  int fd = -1;
};

struct WMSPHandle {
  bool is_host = false;
  CUdeviceptr whole_memory_cuptr;
  void *whole_memory_ptr = nullptr;
  size_t total_memory_size = 0;
  size_t total_aligned_memory_size = 0;

  void *host_ptr = nullptr;

  size_t granularity = 0;
  std::vector<int> dev_list;
  std::vector<int> dev_to_memory_rank;
  std::vector<CUmemGenericAllocationHandle> handles;
};

struct WmmpCommunicator;

struct WMMPHandle {
  int id = -1;
  bool is_host = false;

  CUdeviceptr whole_memory_cuptr;
  void *whole_memory_ptr = nullptr;
  size_t total_memory_size = 0;
  size_t total_aligned_memory_size = 0;
  size_t local_memory_size = 0;

  void *host_ptr = nullptr;

  std::vector<int> dev_list;

  std::vector<size_t> memory_offsets;
  CUmemGenericAllocationHandle local_alloc_handle;
  std::vector<PeerMemorySharableHandle> this_mem_handles;
  std::vector<CUmemGenericAllocationHandle> all_handles;
  std::vector<PeerMemorySharableHandle> all_mem_handles;
  WmmpCommunicator *wmmp_comm = nullptr;
};

struct IPCHandle {
  int socket = -1;
  std::string socket_name;
};

static std::mutex mu;
static bool is_wm_init = false;

static std::mutex ptr_map_mu;

typedef enum {
  PTR_TYPE_SP = 0,
  PTR_TYPE_MP = 1,
  PTR_TYPE_CK = 2,
  PTR_TYPE_NC = 3,
} PtrType;

static std::unordered_map<void *, PtrType> ptr_type_map;

static std::unordered_map<void *, std::unique_ptr<WMSPHandle>> wmsp_ptr_to_handle;

struct WmmpCommunicator {
  BootstrapCommunicator *bootstrap_communicator;
  int id = -1;
  int dev_id;
  size_t granularity;
  pid_t self_pid;
  std::vector<pid_t> local_pids;
  std::string temp_dir_for_unix_domain_sockets;
  // Temp variables below
  std::vector<IPCHandle> recv_handles;
  IPCHandle send_handle;
};

// WholeMemory Multiple Process mode data
static std::map<BootstrapCommunicator *, WmmpCommunicator *> wmmp_bcs;

static int next_id_ = 0;

static std::unordered_map<void *, std::unique_ptr<WMMPHandle>> wmmp_ptr_to_handle;

static std::unordered_set<WholeChunkedMemory_t> wmcmp_handle_set;
static std::unordered_set<WholeNCCLMemory_t> wmnmp_handle_set;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int BCRank(BootstrapCommunicator *bc_ptr) {
  return bc_ptr->Rank();
}

int BCSize(BootstrapCommunicator *bc_ptr) {
  return bc_ptr->Size();
}

void BCAllToAll(const void *send_buf, int send_size, void *recv_buf, int recv_size, BootstrapCommunicator *bc_ptr) {
  bc_ptr->AllToAll(send_buf, send_size, recv_buf, recv_size);
}
void BCAllToAllV(const void **send_bufs,
                 const int *send_sizes,
                 void **recv_bufs,
                 const int *recv_sizes,
                 BootstrapCommunicator *bc_ptr) {
  bc_ptr->AllToAllV(send_bufs, send_sizes, recv_bufs, recv_sizes);
}

void BCBarrier(BootstrapCommunicator *bc_ptr) {
  bc_ptr->Barrier();
}

void CollBroadcastString(std::string *str, int root, BootstrapCommunicator *bc_ptr) {
  auto len = CollBroadcast<size_t>(str->size(), root, bc_ptr);
  int rank_count = BCSize(bc_ptr);
  str->resize(len, ' ');
  std::string send_str;
  for (int i = 0; i < rank_count; i++) send_str.append(*str);
  std::string recv_str;
  recv_str.resize(len * rank_count);
  BCAllToAll(send_str.c_str(), len, &recv_str[0], len, bc_ptr);
  str->assign(recv_str.substr(root * len, len));
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

size_t GetGranularity(int dev_id) {
  size_t granularity = 0;
  CUmemAllocationProp prop;
  memset(&prop, 0, sizeof(prop));
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;
  CUmemAllocationGranularity_flags flags = CU_MEM_ALLOC_GRANULARITY_RECOMMENDED;
  prop.location.id = dev_id;
  WM_CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, flags));
  if (granularity < 512 * 1024 * 1024) granularity = 512 * 1024 * 1024;
  return granularity;
}

size_t GetGranularityOfDevices(const int *dev_list, int dev_count) {
  size_t granularity = GetGranularity(dev_list[0]);
  WM_CHECK(dev_list != nullptr && dev_count > 0);
  for (int i = 1; i < dev_count; i++) {
    WM_CHECK(granularity == GetGranularity(dev_list[i]));
  }
  return granularity;
}

enum WMOPTypes : int32_t {
  WM_OP_STARTING = 0xE601E,
  WM_OP_ENDING,
  WM_OP_ALLOCATING,
  WM_OP_COMPUTINT_SIZE,
  WM_OP_DEALLOCATING,
};

void StartCollOp(WMOPTypes wm_op, BootstrapCommunicator *bc_ptr) {
  CollCheckAllSame<WMOPTypes>(wm_op, bc_ptr);
}

size_t GetGranularityOfAllRanks(int local_dev_id, BootstrapCommunicator *bc_ptr) {
  size_t granularity = GetGranularity(local_dev_id);
  std::vector<size_t> gv;
  CollAllGather<size_t>(granularity, &gv, bc_ptr);
  int size = bc_ptr->Size();
  for (int i = 0; i < size; i++) {
    if (gv[i] > granularity) granularity = gv[i];
  }
  CollCheckAllSame(granularity, bc_ptr);
  //if (cc_ptr->Rank() == 0) std::cerr << "Using granularity " << granularity << std::endl;
  return granularity;
}

void DistributePages(std::vector<size_t> *page_counts, size_t total_page_count, int wg_size) {
  page_counts->resize(wg_size, 0);
  for (int i = 0; i < wg_size; i++) {
    size_t start_page = total_page_count * i / wg_size;
    size_t end_page = (total_page_count * (i + 1)) / wg_size;
    size_t rank_page_count = end_page - start_page;
    page_counts->at(i) = rank_page_count;
  }
}

// for sender, create a socket.
void IPCCreateSocket(IPCHandle *handle, const char *name) {
  int server_fd;
  struct sockaddr_un servaddr {};

  handle->socket = -1;
  handle->socket_name = "";

  // Creating socket file descriptor
  if ((server_fd = socket(AF_UNIX, SOCK_DGRAM, 0)) == 0) {
    std::cerr << "IPC failure: Socket creation failed" << std::endl;
    return;
  }

  unlink(name);
  bzero(&servaddr, sizeof(servaddr));
  servaddr.sun_family = AF_UNIX;

  size_t len = strlen(name);
  if (len > (sizeof(servaddr.sun_path) - 1)) {
    std::cerr << "IPC failure: Cannot bind provided name to socket. Name too large" << std::endl;
    abort();
    return;
  }

  strcpy(servaddr.sun_path, name);

  if (bind(server_fd, (struct sockaddr *) &servaddr, SUN_LEN(&servaddr)) < 0) {
    std::cerr << "IPC failure: Binding socket failed" << std::endl;
    abort();
    return;
  }

  handle->socket_name = name;
  handle->socket = server_fd;
}

// for receiver
void IPCOpenSocket(IPCHandle *handle, const char *name) {
  WM_CHECK(handle != nullptr);
  int sock = 0;
  struct sockaddr_un cliaddr {};

  if ((sock = socket(AF_UNIX, SOCK_DGRAM, 0)) < 0) {
    std::cerr << "IPC failure: Socket creation error" << std::endl;
    abort();
    return;
  }
  bzero(&cliaddr, sizeof(cliaddr));
  cliaddr.sun_family = AF_UNIX;

  strcpy(cliaddr.sun_path, name);
  if (bind(sock, (struct sockaddr *) &cliaddr, sizeof(cliaddr)) < 0) {
    std::cerr << "IPC failure: Binding socket failed, name=" << name << std::endl;
    abort();
    return;
  }

  handle->socket = sock;
  handle->socket_name = name;
}

// for both sender and receiver
void IPCCloseSocket(IPCHandle *handle) {
  if (!handle) {
    std::cerr << "Closing null handle" << std::endl;
    abort();
  }

  if (!handle->socket_name.empty()) {
    WM_CHECK(unlink(handle->socket_name.c_str()) == 0);
    handle->socket_name = "";
  }
  close(handle->socket);
  handle->socket = -1;
};

void IPCSendSharableHandle(IPCHandle *handle, int fd_to_send, const char *dst_name) {
  struct msghdr msg {};
  struct iovec iov[1];

  union {
    struct cmsghdr cm;
    char control[CMSG_SPACE(sizeof(int))];
  } control_un{};

  struct cmsghdr *cmptr;
  struct sockaddr_un cliaddr {};

  // Construct client address to send this Shareable handle to
  bzero(&cliaddr, sizeof(cliaddr));
  cliaddr.sun_family = AF_UNIX;
  strcpy(cliaddr.sun_path, dst_name);

  // Send corresponding shareable handle to the client
  int sendfd = fd_to_send;

  msg.msg_control = control_un.control;
  msg.msg_controllen = sizeof(control_un.control);

  cmptr = CMSG_FIRSTHDR(&msg);
  cmptr->cmsg_len = CMSG_LEN(sizeof(int));
  cmptr->cmsg_level = SOL_SOCKET;
  cmptr->cmsg_type = SCM_RIGHTS;

  memmove(CMSG_DATA(cmptr), &sendfd, sizeof(sendfd));

  msg.msg_name = (void *) &cliaddr;
  msg.msg_namelen = sizeof(struct sockaddr_un);

  iov[0].iov_base = (void *) "";
  iov[0].iov_len = 1;
  msg.msg_iov = iov;
  msg.msg_iovlen = 1;

  ssize_t sendResult = sendmsg(handle->socket, &msg, 0);
  if (sendResult <= 0) {
    std::cerr << "IPC failure: Sending data over socket failed" << std::endl;
    abort();
    return;
  }
}

// return receive_fd
int IPCRecvSharableHandle(IPCHandle *handle) {
  struct msghdr msg = {nullptr};
  struct iovec iov[1];
  // struct cmsghdr cm{};

  // Union to guarantee alignment requirements for control array
  union {
    struct cmsghdr cm;
    char control[CMSG_SPACE(sizeof(int))];
  } control_un{};

  struct cmsghdr *cmptr;
  ssize_t n;
  int receivedfd;
  char dummy_buffer[1];

  msg.msg_control = control_un.control;
  msg.msg_controllen = sizeof(control_un.control);

  iov[0].iov_base = (void *) dummy_buffer;
  iov[0].iov_len = sizeof(dummy_buffer);

  msg.msg_iov = iov;
  msg.msg_iovlen = 1;

  if ((n = recvmsg(handle->socket, &msg, 0)) <= 0) {
    std::cerr << "IPC failure: Receiving data over socket failed" << std::endl;
    abort();
    return -1;
  }

  if (((cmptr = CMSG_FIRSTHDR(&msg)) != nullptr) && (cmptr->cmsg_len == CMSG_LEN(sizeof(int)))) {
    if ((cmptr->cmsg_level != SOL_SOCKET) || (cmptr->cmsg_type != SCM_RIGHTS)) {
      std::cerr << "Non socket received." << std::endl;
      abort();
      return -1;
    }

    memmove(&receivedfd, CMSG_DATA(cmptr), sizeof(receivedfd));
    return receivedfd;
  } else {
    std::cerr << "Recv cm_ptr=" << cmptr << ", cmsg_len=" << (cmptr ? cmptr->cmsg_len : -1) << std::endl;
    abort();
    return -1;
  }
}

CUmemGenericAllocationHandle CreateCUMem(size_t size, int dev_id) {
  CUmemGenericAllocationHandle h;
  CUmemAllocationProp prop;
  memset(&prop, 0, sizeof(prop));
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;
  prop.location.id = dev_id;
  WM_CU_CHECK(cuMemCreate(&h, size, &prop, 0));
  return h;
}

PeerMemorySharableHandle CreateSharableHandle(CUmemGenericAllocationHandle h) {
  PeerMemorySharableHandle pmsh;
  WM_CU_CHECK(cuMemExportToShareableHandle(&pmsh.fd, h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
  return pmsh;
}

CUmemGenericAllocationHandle ImportCUMemHandle(PeerMemorySharableHandle pmsh) {
  CUmemGenericAllocationHandle h;
  WM_CU_CHECK(cuMemImportFromShareableHandle(&h,
                                             (void *) (uintptr_t) pmsh.fd,
                                             CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
  return h;
}

}// namespace

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

std::string GetRecverSocketName(pid_t sender_pid,
                                int src_rank,
                                pid_t recver_pid,
                                int dst_rank,
                                const std::string &temp_dir_for_unix_domain_sockets) {
  char temp[108];
  snprintf(temp,
           108,
           "%s/RP%uR%dfromP%uR%d",
           temp_dir_for_unix_domain_sockets.c_str(),
           sender_pid,
           src_rank,
           recver_pid,
           dst_rank);
  return std::string(temp);
}

std::string GetSenderSocketName(pid_t sender_pid, int src_rank, const std::string &temp_dir_for_unix_domain_sockets) {
  char temp[108];
  snprintf(temp, 108, "%s/SP%uR%dA", temp_dir_for_unix_domain_sockets.c_str(), sender_pid, src_rank);
  return std::string(temp);
}

void CreateTempDirForUnixDomainSockets(WmmpCommunicator *wmmp_communicator) {
  auto *bc_ptr = wmmp_communicator->bootstrap_communicator;
  const char *sock_prefix = getenv("WHOLEGRAPH_TMPNAME");
  std::string wholememory_prefix_str = "wmtmp";
  if (sock_prefix != nullptr) {
    wholememory_prefix_str = sock_prefix;
  }
  wholememory_prefix_str.append(".XXXXXX");
  char template_str[128] = "/tmp/";
  strcpy(&template_str[strlen(template_str)], wholememory_prefix_str.c_str());
  if (bc_ptr->Rank() == 0) {
    char *tmp_dir = mkdtemp(template_str);
    //fprintf(stderr, "Creating tmp_dir=%s\n", tmp_dir);
    WM_CHECK(tmp_dir != nullptr);
    wmmp_communicator->temp_dir_for_unix_domain_sockets = tmp_dir;
  }
  CollBroadcastString(&wmmp_communicator->temp_dir_for_unix_domain_sockets, 0, bc_ptr);
  wmmp_communicator->self_pid = getpid();
  CollAllGather<pid_t>(wmmp_communicator->self_pid, &wmmp_communicator->local_pids, bc_ptr);
}
void RemoveTempDirForUnixDomainSockets(WmmpCommunicator *wmmp_communicator) {
  auto *bc_ptr = wmmp_communicator->bootstrap_communicator;
  bc_ptr->Barrier();
  if (bc_ptr->Rank() == 0) {
    remove(wmmp_communicator->temp_dir_for_unix_domain_sockets.c_str());
  }
  bc_ptr->Barrier();
}

void OpenUnixDomainSockets(WmmpCommunicator *wmmp_communicator) {
  auto *bc_ptr = wmmp_communicator->bootstrap_communicator;
  int local_size = bc_ptr->Size();
  wmmp_communicator->recv_handles.resize(local_size);
  for (int i = 0; i < local_size; i++) {
    int src_rank = i;
    int dst_rank = bc_ptr->Rank();
    pid_t src_pid = wmmp_communicator->local_pids[i];
    pid_t dst_pid = wmmp_communicator->self_pid;
    IPCOpenSocket(&wmmp_communicator->recv_handles[i],
                  GetRecverSocketName(src_pid,
                                      src_rank,
                                      dst_pid,
                                      dst_rank,
                                      wmmp_communicator->temp_dir_for_unix_domain_sockets)
                      .c_str());
  }
  IPCCreateSocket(&wmmp_communicator->send_handle,
                  GetSenderSocketName(wmmp_communicator->self_pid,
                                      bc_ptr->Rank(),
                                      wmmp_communicator->temp_dir_for_unix_domain_sockets)
                      .c_str());
  bc_ptr->Barrier();
}
void CloseUnixDomainSockets(WmmpCommunicator *wmmp_communicator) {
  auto *bc_ptr = wmmp_communicator->bootstrap_communicator;
  bc_ptr->Barrier();
  IPCCloseSocket(&wmmp_communicator->send_handle);
  for (int i = 0; i < bc_ptr->Size(); i++) {
    IPCCloseSocket(&wmmp_communicator->recv_handles[i]);
  }
  bc_ptr->Barrier();
}

void AllToAllMemoryHandlesByIPC(const std::vector<PeerMemorySharableHandle> &memory_handles,
                                std::vector<PeerMemorySharableHandle> *all_memory_handles,
                                std::vector<size_t> *memory_offsets,
                                WmmpCommunicator *wmmp_communicator) {
  auto *bc_ptr = wmmp_communicator->bootstrap_communicator;
  int wg_size = bc_ptr->Size();
  WM_CHECK((size_t) wg_size + 1 == memory_offsets->size());
  for (int i = 0; i < wg_size; i++) {
    pid_t src_pid = wmmp_communicator->self_pid;
    int src_rank = bc_ptr->Rank();
    int dst_rank = i;
    pid_t dst_pid = wmmp_communicator->local_pids[dst_rank];
    if (memory_handles[i].fd >= 0) {
      IPCSendSharableHandle(&wmmp_communicator->send_handle,
                            memory_handles[i].fd,
                            GetRecverSocketName(src_pid,
                                                src_rank,
                                                dst_pid,
                                                dst_rank,
                                                wmmp_communicator->temp_dir_for_unix_domain_sockets)
                                .c_str());
    }
  }
  bc_ptr->Barrier();
  all_memory_handles->resize(wg_size);
  for (int i = 0; i < wg_size; i++) {
    if (memory_offsets->at(i + 1) - memory_offsets->at(i) != 0) {
      int src_local_rank = i;
      all_memory_handles->at(i).fd = IPCRecvSharableHandle(&wmmp_communicator->recv_handles[src_local_rank]);
    } else {
      all_memory_handles->at(i).fd = -1;
    }
  }
  bc_ptr->Barrier();
  for (int i = 0; i < wg_size; i++) {
    if (memory_handles[i].fd >= 0) {
      WM_CHECK(close(memory_handles[i].fd) == 0);
    }
  }
}

}// namespace

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct WholeChunkedMemory {
  WholeChunkedMemory() {
    int dev_count = 0;
    WM_CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    dev_handles.resize(dev_count, nullptr);
  }
  int id = -1;

  size_t min_granularity = 0;
  size_t raw_size = 0;
  size_t alloc_size = 0;
  std::mutex wcmmu;
  WholeChunkedMemoryHandle cpu_handle{};
  std::vector<WholeChunkedMemoryHandle *> dev_handles;

  std::vector<int> dev_list;

  void *local_mem_ptr = nullptr;
  cudaIpcMemHandle_t local_ipc_handle;
  std::vector<cudaIpcMemHandle_t> all_mem_handles;
  WmmpCommunicator *wmmp_comm = nullptr;

  WholeChunkedMemoryHandle *GetDeviceHandle(int dev_id) {
    if (dev_id == -1) return &cpu_handle;
    std::unique_lock<std::mutex> mlock(wcmmu);
    size_t idx = 0;
    for (; idx < dev_list.size(); idx++) {
      if (dev_list[idx] == dev_id) break;
    }
    WM_CHECK(idx < dev_list.size());
    if (dev_handles[dev_id] == nullptr) {
      CreateDeviceHandleLocked(dev_id);
    }
    return dev_handles[dev_id];
  }
  void CreateDeviceHandleLocked(int dev_id) {
    int dev_count = 0;
    WM_CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    WM_CHECK(dev_id >= 0 && dev_id < dev_count);
    if (dev_handles[dev_id] == nullptr) {
      int current_dev;
      WM_CUDA_CHECK(cudaGetDevice(&current_dev));
      WM_CUDA_CHECK(cudaSetDevice(dev_id));
      WholeChunkedMemoryHandle *h;
      WM_CUDA_CHECK(cudaMalloc((void **) &h, sizeof(WholeChunkedMemoryHandle)));
      WM_CUDA_CHECK(cudaMemcpy(h, &cpu_handle, sizeof(WholeChunkedMemoryHandle), cudaMemcpyHostToDevice));
      WM_CUDA_CHECK(cudaDeviceSynchronize());
      dev_handles[dev_id] = h;
      WM_CUDA_CHECK(cudaSetDevice(current_dev));
    }
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct WholeNCCLMemory {
  WholeNCCLMemory() {
  }
  int id = -1;
  size_t min_granularity = 0;
  size_t raw_size = 0;
  size_t alloc_size = 0;
  size_t local_memory_size = 0;
  void *local_memory_ptr = nullptr;
  WmmpCommunicator *wmmp_comm = nullptr;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static void WmmpFinalizeLocked();
static void WmspFreeLocked(void *ptr);
static void WmmpFreeLocked(void *ptr);
void WcmmpFreeLocked(WholeChunkedMemory_t wcmt);
void WnmmpFreeLocked(WholeNCCLMemory_t wnmt);

static const int kPrimeTableTargetCount = 128 * 1024 * 1024;
static const int kPrimeTableStepThreshold = 64 * 1024;
static const int kPrimeTableStepSize[2] = {32, 1024};
static std::vector<int> prime_tables[2];

int GetPrimeValue(int start);

static void GeneratePrimeTable() {
  std::vector<bool> prime_vector(kPrimeTableTargetCount, true);
  prime_vector[0] = prime_vector[1] = false;
  for (int i = 2; i < (int) sqrt(kPrimeTableTargetCount); i++) {
    if (prime_vector[i] == false) continue;
    for (int j = i * 2; j < kPrimeTableTargetCount; j += i) {
      prime_vector[j] = false;
    }
  }
  int max_prime = 2;
  for (int i = kPrimeTableTargetCount - 1; i >= 2; i--) {
    if (prime_vector[i] == true) {
      max_prime = i;
      break;
    }
  }
  WM_CHECK(max_prime > kPrimeTableStepThreshold);
  int large_prime_table_size = max_prime / kPrimeTableStepSize[1] + 1;
  prime_tables[0].resize(DivUp(kPrimeTableStepThreshold, kPrimeTableStepSize[0]) + 1, max_prime);
  prime_tables[1].resize(large_prime_table_size, max_prime);
  int prime_value = max_prime;
  for (int i = max_prime; i >= 0; i--) {
    if (prime_vector[i] == true) {
      prime_value = i;
    }
    if (i % kPrimeTableStepSize[1] == 0) {
      prime_tables[1][i / kPrimeTableStepSize[1]] = prime_value;
    }
  }
  prime_value = GetPrimeValue(kPrimeTableStepThreshold);
  for (int i = kPrimeTableStepThreshold; i >= 0; i--) {
    if (prime_vector[i] == true) {
      prime_value = i;
    }
    if (i % kPrimeTableStepSize[0] == 0) {
      prime_tables[0][i / kPrimeTableStepSize[0]] = prime_value;
    }
  }
}

int GetPrimeValue(int start) {
  if (start < kPrimeTableStepThreshold) {
    return prime_tables[0][DivUp(start, kPrimeTableStepSize[0])];
  } else {
    int idx = DivUp(start, kPrimeTableStepSize[1]);
    WM_CHECK(idx < (int) prime_tables[1].size());
    return prime_tables[1][idx];
  }
}

void WholeMemoryInit() {
  std::unique_lock<std::mutex> lock(mu);
  WM_CHECK(!is_wm_init);
  WM_CU_CHECK(cuInit(0));
  GeneratePrimeTable();
  is_wm_init = true;
}
void WholeMemoryFinalize() {
  std::unique_lock<std::mutex> lock(mu);
  WM_CHECK(is_wm_init);
  if (!wmmp_bcs.empty()) {
    WmmpFinalizeLocked();
  }
  // release all memory
  while (!wmsp_ptr_to_handle.empty()) {
    auto it = wmsp_ptr_to_handle.begin();
    WmspFreeLocked(it->first);
  }
  is_wm_init = false;
}
void WmspMalloc(void **ptr, size_t size, const int *dev_list, int dev_count) {
  std::unique_lock<std::mutex> lock(mu);
  int all_dev_count;
  WM_CUDA_CHECK(cudaGetDeviceCount(&all_dev_count));
  std::vector<int> dev_list_vec;
  if (dev_list == nullptr || dev_count == 0) {
    dev_list_vec.resize(all_dev_count);
    for (int i = 0; i < all_dev_count; i++) dev_list_vec[i] = i;
    dev_count = all_dev_count;
  } else {
    dev_list_vec.resize(dev_count);
    for (int i = 0; i < dev_count; i++) {
      WM_CHECK(dev_list[i] >= 0 && dev_list[i] < all_dev_count);
      dev_list_vec[i] = dev_list[i];
    }
  }
  size_t granularity = GetGranularityOfDevices(dev_list_vec.data(), dev_list_vec.size());
  WMSPHandle *h = new WMSPHandle;
  h->is_host = false;
  h->dev_list = dev_list_vec;
  h->dev_to_memory_rank.resize(all_dev_count, -1);
  h->total_memory_size = size;
  if (size >= 16LL * 1024LL * 1024LL * 1024LL) granularity = 512LL * 1024LL * 1024LL;
  h->total_aligned_memory_size = AlignUp(size, granularity);
  h->granularity = granularity;
  size_t page_count = h->total_aligned_memory_size / granularity;
  WM_CU_CHECK(cuMemAddressReserve(&h->whole_memory_cuptr, h->total_aligned_memory_size, granularity, 0, 0));
  h->handles.resize(h->dev_list.size());
  for (size_t i = 0; i < dev_list_vec.size(); i++) {
    h->dev_to_memory_rank[dev_list_vec[i]] = i;
    size_t start_page = page_count * i / dev_list_vec.size();
    size_t end_page = (page_count * (i + 1)) / dev_list_vec.size();
    size_t current_page_count = end_page - start_page;
    if (current_page_count == 0) continue;

    CUmemAllocationProp prop;
    memset(&prop, 0, sizeof(prop));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;
    prop.location.id = dev_list_vec[i];

    WM_CU_CHECK(cuMemCreate(&h->handles[i], current_page_count * granularity, &prop, 0));

    WM_CU_CHECK(cuMemMap(h->whole_memory_cuptr + start_page * granularity,
                         current_page_count * granularity,
                         0,
                         h->handles[i],
                         0));

    CUmemAccessDesc madesc[dev_list_vec.size()];
    for (size_t ii = 0; ii < dev_list_vec.size(); ii++) {
      madesc[ii].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      madesc[ii].location.id = dev_list_vec[ii];
      madesc[ii].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    WM_CU_CHECK(cuMemSetAccess(h->whole_memory_cuptr + start_page * granularity,
                               current_page_count * granularity,
                               &madesc[0],
                               dev_list_vec.size()));
  }
  h->whole_memory_ptr = (void *) h->whole_memory_cuptr;
  *ptr = h->whole_memory_ptr;
  {
    std::unique_lock<std::mutex> ptr_lock(ptr_map_mu);
    ptr_type_map.emplace(*ptr, PTR_TYPE_SP);
    wmsp_ptr_to_handle.emplace(*ptr, h);
  }
}
void WmspMallocHost(void **ptr, size_t size) {
  std::unique_lock<std::mutex> lock(mu);
  WMSPHandle *h = new WMSPHandle;
  h->is_host = true;
  h->total_memory_size = h->total_aligned_memory_size = size;
  h->host_ptr = malloc(size);
  WM_CUDA_CHECK(cudaHostRegister(h->host_ptr, size, cudaHostRegisterDefault));
  WM_CUDA_CHECK(cudaHostGetDevicePointer(&h->whole_memory_ptr, h->host_ptr, 0));
  *ptr = h->whole_memory_ptr;
  {
    std::unique_lock<std::mutex> ptr_lock(ptr_map_mu);
    ptr_type_map.emplace(*ptr, PTR_TYPE_SP);
    wmsp_ptr_to_handle.emplace(*ptr, h);
  }
}
static void WmspFreeLocked(void *ptr) {
  std::unordered_map<void *, std::unique_ptr<WMSPHandle>>::const_iterator it;
  WMSPHandle *h = nullptr;
  {
    std::unique_lock<std::mutex> ptr_lock(ptr_map_mu);
    it = wmsp_ptr_to_handle.find(ptr);
    WM_CHECK(it != wmsp_ptr_to_handle.end());
    h = it->second.get();
  }
  if (h->is_host) {
    WM_CUDA_CHECK(cudaHostUnregister(h->host_ptr));
    free(h->host_ptr);
    {
      std::unique_lock<std::mutex> ptr_lock(ptr_map_mu);
      ptr_type_map.erase(ptr);
      wmsp_ptr_to_handle.erase(ptr);
    }
    return;
  }
  size_t page_count = h->total_aligned_memory_size / h->granularity;
  for (size_t i = 0; i < h->dev_list.size(); i++) {
    size_t start_page = page_count * i / h->dev_list.size();
    size_t end_page = (page_count * (i + 1)) / h->dev_list.size();
    size_t page_count = end_page - start_page;
    if (page_count == 0) continue;
    WM_CU_CHECK(cuMemUnmap(h->whole_memory_cuptr + start_page * h->granularity, page_count * h->granularity));
    WM_CU_CHECK(cuMemRelease(h->handles[i]));
  }
  WM_CU_CHECK(cuMemAddressFree(h->whole_memory_cuptr, h->total_aligned_memory_size));
  {
    std::unique_lock<std::mutex> ptr_lock(ptr_map_mu);
    ptr_type_map.erase(ptr);
    wmsp_ptr_to_handle.erase(ptr);
  }
}
void WmspFree(void *ptr) {
  std::unique_lock<std::mutex> lock(mu);
  WmspFreeLocked(ptr);
}
static void WmspGetRankAndSizeLocked(void *ptr, int *rank, int *size) {
  auto it = wmsp_ptr_to_handle.find(ptr);
  WM_CHECK(it != wmsp_ptr_to_handle.end());
  WMSPHandle *h = it->second.get();
  int dev_id;
  WM_CUDA_CHECK(cudaGetDevice(&dev_id));
  *size = h->dev_list.size();
  *rank = h->dev_to_memory_rank[dev_id];
  WM_CHECK(*rank >= 0);
}
static int GetIdLocked() {
  return next_id_++;
}
BootstrapCommunicator *WmmpCreateCommunicator(int size, WmmpUniqueId unique_id, int rank) {
  std::unique_lock<std::mutex> lock(mu);
  auto *bc = new BootstrapCommunicator();
  bc->InitRank(size, unique_id, rank);
  auto *wmmp_comm = new WmmpCommunicator;
  wmmp_bcs.insert(std::pair<BootstrapCommunicator *, WmmpCommunicator *>(bc, wmmp_comm));
  wmmp_comm->id = GetIdLocked();
  wmmp_comm->bootstrap_communicator = bc;
  wmmp_comm->dev_id = bc->CUDADevID();
  int dev_id = -1;
  WM_CUDA_CHECK(cudaGetDevice(&dev_id));
  WM_CHECK(dev_id == wmmp_comm->dev_id);
  wmmp_comm->granularity = GetGranularityOfAllRanks(wmmp_comm->dev_id, bc);
  StartCollOp(WM_OP_STARTING, bc);
  CreateTempDirForUnixDomainSockets(wmmp_comm);
  return bc;
}
void WmmpDestroyCommunicatorLocked(BootstrapCommunicator *bootstrap_communicator) {
  StartCollOp(WM_OP_ENDING, bootstrap_communicator);
  auto it = wmmp_bcs.find(bootstrap_communicator);
  WM_CHECK(it != wmmp_bcs.end());
  auto *wmmp_comm = it->second;
  RemoveTempDirForUnixDomainSockets(it->second);
  wmmp_bcs.erase(bootstrap_communicator);
  delete bootstrap_communicator;
  delete wmmp_comm;
}
void WmmpDestroyCommunicator(BootstrapCommunicator *bootstrap_communicator) {
  std::unique_lock<std::mutex> lock(mu);
  WmmpDestroyCommunicatorLocked(bootstrap_communicator);
}
static void WmmpFinalizeLocked() {
  std::map<int, void *> remaining_handles;
  for (auto &handle : wmmp_ptr_to_handle) {
    remaining_handles.emplace(handle.second->id, handle.first);
  }
  while (!remaining_handles.empty()) {
    auto it = remaining_handles.begin();
    int id = it->first;
    WmmpFreeLocked(it->second);
    remaining_handles.erase(id);
  }
  for (auto handle_ptr : wmcmp_handle_set) {
    remaining_handles.emplace(handle_ptr->id, handle_ptr);
  }
  while (!remaining_handles.empty()) {
    auto it = remaining_handles.begin();
    int id = it->first;
    WcmmpFreeLocked((WholeChunkedMemory_t) it->second);
    remaining_handles.erase(id);
  }
  for (auto handle_ptr : wmnmp_handle_set) {
    remaining_handles.emplace(handle_ptr->id, handle_ptr);
  }
  while (!remaining_handles.empty()) {
    auto it = remaining_handles.begin();
    int id = it->first;
    WnmmpFreeLocked((WholeNCCLMemory_t) it->second);
    remaining_handles.erase(id);
  }
  for (auto &handle : wmmp_bcs) {
    remaining_handles.emplace(handle.second->id, handle.first);
  }
  while (!remaining_handles.empty()) {
    auto it = remaining_handles.begin();
    int id = it->first;
    WmmpDestroyCommunicatorLocked((BootstrapCommunicator *) it->second);
    remaining_handles.erase(id);
  }
}
void WmmpFinalize() {
  std::unique_lock<std::mutex> lock(mu);
  WmmpFinalizeLocked();
}
static std::string CollWmmpGetSharedFileName(int shmem_id, WmmpCommunicator *wmmp_comm) {
  std::string name;
  name = "wgmphs_pid_";
  name += std::to_string(wmmp_comm->local_pids[0]);
  name += "_mid_";
  name += std::to_string(shmem_id);
  return name;
}
static void WmmpMallocHostLocked(void **ptr, size_t size, BootstrapCommunicator *bc_ptr) {
  auto it = wmmp_bcs.find(bc_ptr);
  WM_CHECK(it != wmmp_bcs.end());
  WM_CHECK(bc_ptr->IsIntraNode());
  auto *wmmp_comm = it->second;
  auto *h = new WMMPHandle;
  h->wmmp_comm = it->second;
  int local_rank_index = bc_ptr->Rank();
  bool is_first_rank = (local_rank_index == 0);

  StartCollOp(WM_OP_ALLOCATING, bc_ptr);
  h->is_host = true;
  h->id = GetIdLocked();
  CollAllGather(wmmp_comm->dev_id, &h->dev_list, bc_ptr);
  h->total_memory_size = h->total_aligned_memory_size = size;
  std::string shm_name = CollWmmpGetSharedFileName(h->id, wmmp_comm);
  int shm_fd = -1;
  if (is_first_rank) {
    shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    if (shm_fd < 0) {
      std::cerr << "Rank=" << bc_ptr->Rank() << ", create shm_name=" << shm_name << ", failed." << std::endl;
      abort();
    }
    WM_CHECK(ftruncate(shm_fd, h->total_memory_size) == 0);
    bc_ptr->Barrier();
  } else {
    bc_ptr->Barrier();
    shm_fd = shm_open(shm_name.c_str(), O_RDWR, S_IRUSR | S_IWUSR);
    if (shm_fd < 0) {
      std::cerr << "Rank=" << bc_ptr->Rank() << ", open shm_name=" << shm_name << ", failed." << std::endl;
      abort();
    }
  }
  bc_ptr->Barrier();
  h->host_ptr = mmap(nullptr, h->total_memory_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  WM_CHECK(h->host_ptr != (void *) -1);
  WM_CHECK(close(shm_fd) == 0);
  WM_CUDA_CHECK(cudaHostRegister(h->host_ptr, size, cudaHostRegisterDefault));
  WM_CUDA_CHECK(cudaHostGetDevicePointer(&h->whole_memory_ptr, h->host_ptr, 0));
  WM_CHECK(h->whole_memory_ptr == h->host_ptr);
  *ptr = h->whole_memory_ptr;
  {
    std::unique_lock<std::mutex> ptr_lock(ptr_map_mu);
    ptr_type_map.emplace(*ptr, PTR_TYPE_MP);
    wmmp_ptr_to_handle.emplace(*ptr, h);
  }
  //std::cout << "Allocating Host ptr=" << *ptr << ", memory_count=" << wmmp_ptr_to_handle.size() << std::endl;
}
static void WmmpMallocLocked(void **ptr, size_t size, BootstrapCommunicator *bc_ptr) {
  auto it = wmmp_bcs.find(bc_ptr);
  WM_CHECK(it != wmmp_bcs.end());
  WM_CHECK(bc_ptr->IsIntraNode());
  auto *wmmp_comm = it->second;
  auto *h = new WMMPHandle;
  h->wmmp_comm = wmmp_comm;
  int local_rank_index = bc_ptr->Rank();

  StartCollOp(WM_OP_ALLOCATING, bc_ptr);
  h->is_host = false;
  h->id = GetIdLocked();
  CollCheckAllSame(h->id, bc_ptr);
  CollAllGather(h->wmmp_comm->dev_id, &h->dev_list, bc_ptr);
  h->total_memory_size = size;
  size_t granularity = h->wmmp_comm->granularity;
  if (size >= 16LL * 1024LL * 1024LL * 1024LL) granularity = 512LL * 1024LL * 1024LL;
  h->total_aligned_memory_size = AlignUp(size, granularity);
  size_t total_page_count = h->total_aligned_memory_size / granularity;
  std::vector<size_t> page_counts;
  int rank_count = bc_ptr->Size();
  DistributePages(&page_counts, total_page_count, rank_count);
  h->memory_offsets.resize(rank_count + 1);
  h->memory_offsets[0] = 0;
  for (int i = 0; i < rank_count; i++) {
    h->memory_offsets[i + 1] = h->memory_offsets[i] + page_counts[i] * granularity;
  }
  size_t local_memory_size = h->memory_offsets[local_rank_index + 1] - h->memory_offsets[local_rank_index];
  h->local_memory_size = local_memory_size;

  WM_CU_CHECK(cuMemAddressReserve(&h->whole_memory_cuptr, h->total_aligned_memory_size, granularity, 0, 0));
  h->this_mem_handles.resize(rank_count);
  if (local_memory_size > 0) {
    h->local_alloc_handle = CreateCUMem(local_memory_size, h->wmmp_comm->dev_id);
    for (int i = 0; i < rank_count; i++) {
      h->this_mem_handles[i] = CreateSharableHandle(h->local_alloc_handle);
    }
  } else {
    for (int i = 0; i < rank_count; i++) {
      h->this_mem_handles[i].fd = -1;
    }
  }
  OpenUnixDomainSockets(h->wmmp_comm);
  AllToAllMemoryHandlesByIPC(h->this_mem_handles, &h->all_mem_handles, &h->memory_offsets, h->wmmp_comm);
  CloseUnixDomainSockets(h->wmmp_comm);
  WM_CHECK(h->all_mem_handles.size() == (size_t) rank_count);
  h->all_handles.resize(rank_count);
  for (int i = 0; i < rank_count; i++) {
    size_t mem_size = h->memory_offsets[i + 1] - h->memory_offsets[i];
    if (mem_size > 0) {
      WM_CHECK(h->all_mem_handles[i].fd >= 0);
      h->all_handles[i] = ImportCUMemHandle(h->all_mem_handles[i]);
      WM_CU_CHECK(cuMemMap(h->whole_memory_cuptr + h->memory_offsets[i], mem_size, 0, h->all_handles[i], 0));
      WM_CHECK(close(h->all_mem_handles[i].fd) == 0);
    } else {
      WM_CHECK(h->all_mem_handles[i].fd == -1);
    }
  }
  CUmemAccessDesc madesc;
  madesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  madesc.location.id = h->wmmp_comm->dev_id;
  madesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  WM_CU_CHECK(cuMemSetAccess(h->whole_memory_cuptr, h->total_aligned_memory_size, &madesc, 1));
  h->whole_memory_ptr = (void *) h->whole_memory_cuptr;
  *ptr = h->whole_memory_ptr;
  {
    std::unique_lock<std::mutex> ptr_lock(ptr_map_mu);
    ptr_type_map.emplace(*ptr, PTR_TYPE_MP);
    wmmp_ptr_to_handle.emplace(*ptr, h);
  }
  //std::cout << "Allocating Device ptr=" << *ptr << ", memory_count=" << wmmp_ptr_to_handle.size() << std::endl;
}
size_t WmmpAggregateSize(size_t local_size, size_t *offset, BootstrapCommunicator *bc_ptr) {
  StartCollOp(WM_OP_COMPUTINT_SIZE, bc_ptr);
  std::vector<size_t> sizes;
  CollAllGather(local_size, &sizes, bc_ptr);
  size_t total_size = 0;
  int rank_idx = bc_ptr->Rank();
  for (size_t i = 0; i < sizes.size(); i++) {
    if (offset != nullptr && (size_t) rank_idx == i) {
      *offset = total_size;
    }
    total_size += sizes[i];
  }
  return total_size;
}
void WmmpMalloc(void **ptr, size_t size, BootstrapCommunicator *bc_ptr) {
  std::unique_lock<std::mutex> lock(mu);
  WmmpMallocLocked(ptr, size, bc_ptr);
}
void WmmpMallocHost(void **ptr, size_t size, BootstrapCommunicator *bc_ptr) {
  std::unique_lock<std::mutex> lock(mu);
  WmmpMallocHostLocked(ptr, size, bc_ptr);
}
static void WmmpFreeLocked(void *ptr) {
  //std::cout << "Freeing ptr=" << ptr << ", memory_count=" << wmmp_ptr_to_handle.size() << std::endl;
  std::unordered_map<void *, std::unique_ptr<WMMPHandle>>::const_iterator it;
  WMMPHandle *h = nullptr;
  {
    std::unique_lock<std::mutex> ptr_lock(ptr_map_mu);
    it = wmmp_ptr_to_handle.find(ptr);
    WM_CHECK(it != wmmp_ptr_to_handle.end());
    h = it->second.get();
  }
  auto *bc_ptr = h->wmmp_comm->bootstrap_communicator;
  StartCollOp(WM_OP_DEALLOCATING, bc_ptr);
  CollCheckAllSame(h->id, bc_ptr);

  if (h->is_host) {
    int local_rank_index = bc_ptr->Rank();
    bool is_first_rank = (local_rank_index == 0);
    WM_CUDA_CHECK(cudaHostUnregister(h->host_ptr));
    WM_CHECK(munmap(h->host_ptr, h->total_memory_size) == 0);
    bc_ptr->Barrier();
    std::string shm_name = CollWmmpGetSharedFileName(h->id, h->wmmp_comm);
    if (is_first_rank) {
      WM_CHECK(shm_unlink(shm_name.c_str()) == 0);
    }
    bc_ptr->Barrier();
    {
      std::unique_lock<std::mutex> ptr_lock(ptr_map_mu);
      ptr_type_map.erase(ptr);
      wmmp_ptr_to_handle.erase(ptr);
    }
    return;
  }

  for (size_t i = 0; i < (size_t) bc_ptr->Size(); i++) {
    size_t mem_size = h->memory_offsets[i + 1] - h->memory_offsets[i];
    if (mem_size > 0) {
      WM_CU_CHECK(cuMemUnmap(h->whole_memory_cuptr + h->memory_offsets[i], mem_size));
      WM_CU_CHECK(cuMemRelease(h->all_handles[i]));
    }
  }
  // Check all memory unmapped.
  bc_ptr->Barrier();
  if (h->local_memory_size > 0) {
    WM_CU_CHECK(cuMemRelease(h->local_alloc_handle));
  }
  WM_CU_CHECK(cuMemAddressFree(h->whole_memory_cuptr, h->total_aligned_memory_size));

  {
    std::unique_lock<std::mutex> ptr_lock(ptr_map_mu);
    ptr_type_map.erase(ptr);
    wmmp_ptr_to_handle.erase(ptr);
  }
}
void WmmpFree(void *ptr) {
  std::unique_lock<std::mutex> lock(mu);
  // skip free after lib finalize.
  if (wmmp_bcs.empty()) return;
  WmmpFreeLocked(ptr);
}

BootstrapCommunicator *WmmpGetBootstrapCommunicator(const void *ptr) {
  auto it = wmmp_ptr_to_handle.find((void *) ptr);
  WM_CHECK(it != wmmp_ptr_to_handle.end());
  WMMPHandle *h = it->second.get();
  return h->wmmp_comm->bootstrap_communicator;
}

static void WmmpGetRankAndSizeLocked(void *ptr, int *rank, int *size) {
  auto it = wmmp_ptr_to_handle.find(ptr);
  WM_CHECK(it != wmmp_ptr_to_handle.end());
  WMMPHandle *h = it->second.get();
  *size = h->wmmp_comm->bootstrap_communicator->Size();
  *rank = h->wmmp_comm->bootstrap_communicator->Rank();
  WM_CHECK(*rank >= 0);
}

void WmmpGetRankIdxAndRankSizeOfPtr(int *rank_idx, int *rank_size, void *ptr) {
  std::unique_lock<std::mutex> lock(mu);
  std::unordered_map<void *, std::unique_ptr<WMMPHandle>>::const_iterator it;
  {
    std::unique_lock<std::mutex> ptr_lock(ptr_map_mu);
    it = wmmp_ptr_to_handle.find(ptr);
    if (it == wmmp_ptr_to_handle.end()) {
      *rank_idx = 0;
      *rank_size = 0;
      return;
    }
  }
  WMMPHandle *h = it->second.get();
  BootstrapCommunicator *bc_ptr = h->wmmp_comm->bootstrap_communicator;
  *rank_idx = bc_ptr->Rank();
  *rank_size = bc_ptr->Size();
}

bool WmmpCanAccessFrom(void *ptr, int dev) {
  if (dev < -1) return false;
  std::unique_lock<std::mutex> lock(mu);
  std::unordered_map<void *, std::unique_ptr<WMMPHandle>>::const_iterator it;
  {
    std::unique_lock<std::mutex> ptr_lock(ptr_map_mu);
    it = wmmp_ptr_to_handle.find(ptr);
    if (it == wmmp_ptr_to_handle.end()) {
      return false;
    }
  }
  WMMPHandle *h = it->second.get();
  if (dev == -1) {
    return h->is_host;
  }
  for (int devid : h->dev_list) {
    if (devid == dev) return true;
  }
  return false;
}

void WmmpBarrier(BootstrapCommunicator *bc_ptr) {
  bc_ptr->Barrier();
}

int64_t WmmpGetRank(BootstrapCommunicator *bc_ptr) {
  return (int64_t) bc_ptr->Rank();
}

int64_t WmmpGetSize(BootstrapCommunicator *bc_ptr) {
  return (int64_t) bc_ptr->Size();
}

void WmmpAllToAll(const void *send_buf,
                  int send_size,
                  void *recv_buf,
                  int recv_size,
                  BootstrapCommunicator *bc_ptr) {
  bc_ptr->AllToAll(send_buf, send_size, recv_buf, recv_size);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void WcmmpMalloc(WholeChunkedMemory_t *pwcmt, size_t size, BootstrapCommunicator *bc_ptr, size_t min_granularity) {
  std::unique_lock<std::mutex> lock(mu);
  auto it = wmmp_bcs.find(bc_ptr);
  WM_CHECK(it != wmmp_bcs.end());
  WM_CHECK(bc_ptr->IsIntraNode());
  auto *wmmp_comm = it->second;
  int real_rank_count = bc_ptr->Size();

  auto *h = new WholeChunkedMemory;
  h->wmmp_comm = wmmp_comm;
  int local_rank_index = bc_ptr->Rank();

  StartCollOp(WM_OP_ALLOCATING, bc_ptr);
  const int64_t kWholeChunkedMemoryAllocating = 0xE601EC6CC8EDEEE0LL;
  CollCheckAllSame(kWholeChunkedMemoryAllocating, bc_ptr);
  h->id = GetIdLocked();
  CollCheckAllSame(h->id, bc_ptr);
  CollAllGather(h->wmmp_comm->dev_id, &h->dev_list, bc_ptr);
  CollCheckAllSame(size, bc_ptr);

  WM_CHECK(min_granularity > 0);
  h->min_granularity = min_granularity;
  h->raw_size = size;
  h->alloc_size = AlignUp(h->raw_size, min_granularity * real_rank_count);
  if (h->alloc_size == 0) h->alloc_size = min_granularity * real_rank_count;
  h->cpu_handle.chunk_count = real_rank_count;
  h->cpu_handle.chunk_size = h->alloc_size / real_rank_count;

  size_t malloc_size = h->cpu_handle.chunk_size;
  if (malloc_size >= 2LL * 1024LL * 1024LL * 1024LL) malloc_size = AlignUp(malloc_size, 512LL * 1024LL * 1024LL);
  WM_CUDA_CHECK(cudaMalloc((void **) &h->local_mem_ptr, malloc_size));
  WM_CUDA_CHECK(cudaIpcGetMemHandle(&h->local_ipc_handle, h->local_mem_ptr));
  CollAllGather(h->local_ipc_handle, &h->all_mem_handles, bc_ptr);
  for (int i = 0; i < real_rank_count; i++) {
    if (i == local_rank_index) {
      h->cpu_handle.chunked_ptrs[i] = h->local_mem_ptr;
    } else {
      WM_CUDA_CHECK(cudaIpcOpenMemHandle(&h->cpu_handle.chunked_ptrs[i],
                                         h->all_mem_handles[i],
                                         cudaIpcMemLazyEnablePeerAccess));
    }
  }
  *pwcmt = h;
  {
    std::unique_lock<std::mutex> ptr_lock(ptr_map_mu);
    ptr_type_map.emplace(h, PTR_TYPE_CK);
    wmcmp_handle_set.insert(h);
  }
}

void WcmmpFreeLocked(WholeChunkedMemory_t wcmt) {
  std::unordered_set<WholeChunkedMemory_t>::const_iterator it;
  {
    std::unique_lock<std::mutex> ptr_lock(ptr_map_mu);
    it = wmcmp_handle_set.find(wcmt);
    WM_CHECK(it != wmcmp_handle_set.end());
  }
  WholeChunkedMemory *h = wcmt;
  auto *wmmp_comm = h->wmmp_comm;
  auto *bc_ptr = wmmp_comm->bootstrap_communicator;
  StartCollOp(WM_OP_DEALLOCATING, bc_ptr);
  CollCheckAllSame(h->id, bc_ptr);

  int local_rank_index = bc_ptr->Rank();

  for (int i = 0; i < (int) bc_ptr->Size(); i++) {
    if (i != local_rank_index) {
      WM_CUDA_CHECK(cudaIpcCloseMemHandle(h->cpu_handle.chunked_ptrs[i]));
    }
  }
  // Check all memory unmapped.
  bc_ptr->Barrier();
  WM_CUDA_CHECK(cudaFree(h->local_mem_ptr));

  for (auto &dev_handle : h->dev_handles) {
    if (dev_handle) {
      WM_CUDA_CHECK(cudaFree(dev_handle));
      dev_handle = nullptr;
    }
  }

  {
    std::unique_lock<std::mutex> ptr_lock(ptr_map_mu);
    ptr_type_map.erase(wcmt);
    wmcmp_handle_set.erase(wcmt);
  }
}

void WcmmpFree(WholeChunkedMemory_t wcmt) {
  std::unique_lock<std::mutex> lock(mu);
  if (wmmp_bcs.empty()) return;
  WcmmpFreeLocked(wcmt);
}

BootstrapCommunicator *WcmmpGetBootstrapCommunicator(const WholeChunkedMemory_t wcmt) {
  auto it = wmcmp_handle_set.find((WholeChunkedMemory_t) wcmt);
  WM_CHECK(it != wmcmp_handle_set.end());
  auto *h = (WholeChunkedMemory *) wcmt;
  return h->wmmp_comm->bootstrap_communicator;
}

static void WcmmpGetRankAndSizeLocked(void *ptr, int *rank, int *size) {
  auto it = wmcmp_handle_set.find((WholeChunkedMemory_t) ptr);
  WM_CHECK(it != wmcmp_handle_set.end());
  auto *h = (WholeChunkedMemory *) ptr;
  auto *bc_ptr = h->wmmp_comm->bootstrap_communicator;
  *size = bc_ptr->Size();
  *rank = bc_ptr->Rank();
  WM_CHECK(*rank >= 0);
}

WholeChunkedMemoryHandle *GetDeviceChunkedHandle(WholeChunkedMemory_t wcmt, int dev_id) {
  if (wcmt == nullptr) return nullptr;
  std::unique_lock<std::mutex> lock(mu);
  std::unordered_set<WholeChunkedMemory_t>::const_iterator it;
  {
    std::unique_lock<std::mutex> ptr_lock(ptr_map_mu);
    it = wmcmp_handle_set.find(wcmt);
    WM_CHECK(it != wmcmp_handle_set.end());
  }
  WholeChunkedMemory *h = wcmt;
  return h->GetDeviceHandle(dev_id);
}

void WcmmpGetLocalMemory(WholeChunkedMemory_t wcmt, void **ptr, size_t *size) {
  auto *bc_ptr = wcmt->wmmp_comm->bootstrap_communicator;
  int local_rank_index = bc_ptr->Rank();
  *ptr = wcmt->cpu_handle.chunked_ptrs[local_rank_index];
  size_t total_size = (local_rank_index + 1) * wcmt->cpu_handle.chunk_size;
  size_t start_offset = local_rank_index * wcmt->cpu_handle.chunk_size;
  *size = wcmt->cpu_handle.chunk_size;
  if (total_size > wcmt->raw_size) {
    if (start_offset > wcmt->raw_size) {
      *size = 0;
    } else {
      *size = wcmt->raw_size - start_offset;
    }
  }
}

void WnmmpMalloc(WholeNCCLMemory_t *pwnmt, size_t size, BootstrapCommunicator *bc_ptr, size_t min_granularity) {
  std::unique_lock<std::mutex> lock(mu);
  auto it = wmmp_bcs.find(bc_ptr);
  WM_CHECK(it != wmmp_bcs.end());
  auto *wmmp_comm = it->second;
  int real_rank_count = bc_ptr->Size();

  auto *h = new WholeNCCLMemory;
  h->wmmp_comm = wmmp_comm;
  //int local_rank_index = bc_ptr->Rank();

  StartCollOp(WM_OP_ALLOCATING, bc_ptr);
  const int64_t kWholeNCCLMemoryAllocating = 0xE601ECCC1EEE077LL;
  CollCheckAllSame(kWholeNCCLMemoryAllocating, bc_ptr);
  h->id = GetIdLocked();
  CollCheckAllSame(h->id, bc_ptr);
  CollCheckAllSame(size, bc_ptr);

  WM_CHECK(min_granularity > 0);
  h->min_granularity = min_granularity;
  h->raw_size = size;
  h->alloc_size = AlignUp(h->raw_size, min_granularity * real_rank_count);
  if (h->alloc_size == 0) h->alloc_size = min_granularity * real_rank_count;

  h->local_memory_size = h->alloc_size / real_rank_count;
  WM_CUDA_CHECK(cudaMalloc((void **) &h->local_memory_ptr, h->local_memory_size));
  *pwnmt = h;
  {
    std::unique_lock<std::mutex> ptr_lock(ptr_map_mu);
    ptr_type_map.emplace(h, PTR_TYPE_NC);
    wmnmp_handle_set.insert(h);
  }
}

void WnmmpFreeLocked(WholeNCCLMemory_t wnmt) {
  std::unordered_set<WholeNCCLMemory_t>::const_iterator it;
  {
    std::unique_lock<std::mutex> ptr_lock(ptr_map_mu);
    it = wmnmp_handle_set.find(wnmt);
    WM_CHECK(it != wmnmp_handle_set.end());
  }
  WholeNCCLMemory *h = wnmt;
  auto *wmmp_comm = h->wmmp_comm;
  auto *bc_ptr = wmmp_comm->bootstrap_communicator;
  StartCollOp(WM_OP_DEALLOCATING, bc_ptr);
  CollCheckAllSame(h->id, bc_ptr);

  WM_CUDA_CHECK(cudaFree(h->local_memory_ptr));

  {
    std::unique_lock<std::mutex> ptr_lock(ptr_map_mu);
    ptr_type_map.erase(wnmt);
    wmnmp_handle_set.erase(wnmt);
  }
}

void WnmmpFree(WholeNCCLMemory_t wnmt) {
  std::unique_lock<std::mutex> lock(mu);
  if (wmmp_bcs.empty()) return;
  WnmmpFreeLocked(wnmt);
}

BootstrapCommunicator *WnmmpGetBootstrapCommunicator(const WholeNCCLMemory_t wnmt) {
  auto it = wmnmp_handle_set.find((WholeNCCLMemory_t) wnmt);
  WM_CHECK(it != wmnmp_handle_set.end());
  auto *h = (WholeNCCLMemory *) wnmt;
  return h->wmmp_comm->bootstrap_communicator;
}

void WnmmpGetLocalMemory(WholeNCCLMemory_t wnmt, void **ptr, size_t *size) {
  auto *bc_ptr = wnmt->wmmp_comm->bootstrap_communicator;
  int local_rank_index = bc_ptr->Rank();
  *ptr = wnmt->local_memory_ptr;
  size_t total_size = (local_rank_index + 1) * wnmt->local_memory_size;
  size_t start_offset = local_rank_index * wnmt->local_memory_size;
  *size = wnmt->local_memory_size;
  if (total_size > wnmt->raw_size) {
    if (start_offset > wnmt->raw_size) {
      *size = 0;
    } else {
      *size = wnmt->raw_size - start_offset;
    }
  }
}

size_t WnmmpGetChunkSize(WholeNCCLMemory_t wnmt) {
  return wnmt->local_memory_size;
}

static void WnmmpGetRankAndSizeLocked(void *ptr, int *rank, int *size) {
  auto it = wmnmp_handle_set.find((WholeNCCLMemory_t) ptr);
  WM_CHECK(it != wmnmp_handle_set.end());
  auto *h = (WholeNCCLMemory *) ptr;
  auto *bc_ptr = h->wmmp_comm->bootstrap_communicator;
  *size = bc_ptr->Size();
  *rank = bc_ptr->Rank();
  WM_CHECK(*rank >= 0);
}

void GetRankAndSize(void *ptr, int *rank, int *size) {
  std::unique_lock<std::mutex> ptr_lock(ptr_map_mu);
  auto it = ptr_type_map.find(ptr);
  WM_CHECK(it != ptr_type_map.end());
  if (it->second == PTR_TYPE_SP) {
    WmspGetRankAndSizeLocked(ptr, rank, size);
  } else if (it->second == PTR_TYPE_MP) {
    WmmpGetRankAndSizeLocked(ptr, rank, size);
  } else if (it->second == PTR_TYPE_CK) {
    WcmmpGetRankAndSizeLocked(ptr, rank, size);
  } else if (it->second == PTR_TYPE_NC) {
    WnmmpGetRankAndSizeLocked(ptr, rank, size);
  } else {
    abort();
  }
}

size_t WmmpCollOffsetAndSize(size_t total_size, size_t *local_size, BootstrapCommunicator *bc_ptr) {
  int local_rank_index = bc_ptr->Rank();
  int all_rank_count = bc_ptr->Size();
  size_t start = total_size * local_rank_index / all_rank_count;
  size_t end = total_size * (local_rank_index + 1) / all_rank_count;
  if (local_size != nullptr) *local_size = end - start;
  return start;
}

}// namespace whole_graph