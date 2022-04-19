#include "whole_memory.h"
#include "whole_chunked_memory.h"

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/mman.h>
#include <sys/un.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "macros.h"
#include "domain_socket_communicator.h"
#include "whole_memory_communicator.h"

namespace whole_memory {

void CollectiveCommunicator::Barrier(const int *ranks, int rank_count) {
  int rank = Rank();
  int size = Size();
  int comm_rank = rank;
  int comm_size = size;
  if (ranks != nullptr && rank_count != 0) {
    const int* rank_pos = std::find(ranks, ranks + rank_count, rank);
    comm_rank = (int)(rank_pos - ranks);
    comm_size = rank_count;
    if (comm_rank == comm_size) return;
  }
  std::vector<int> send_buf(comm_size, 0);
  std::vector<int> recv_buf(comm_size, 0);
  AllToAll(send_buf.data(), sizeof(int), &recv_buf[0], sizeof(int), ranks, rank_count);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct PeerMemorySharableHandle {
  int fd = -1;
};

struct WMSPHandle {
  bool is_host = false;
  CUdeviceptr whole_memory_cuptr;
  void* whole_memory_ptr = nullptr;
  size_t total_memory_size = 0;
  size_t total_aligned_memory_size = 0;

  void* host_ptr = nullptr;

  size_t granularity = 0;
  std::vector<int> dev_list;
  std::vector<CUmemGenericAllocationHandle> handles;
};

struct WMMPHandle {
  int id = -1;
  bool is_host = false;

  CUdeviceptr whole_memory_cuptr;
  void* whole_memory_ptr = nullptr;
  size_t total_memory_size = 0;
  size_t total_aligned_memory_size = 0;
  size_t local_memory_size = 0;

  void* host_ptr = nullptr;

  std::vector<int> ranks;
  std::vector<int> dev_list;

  std::vector<size_t> memory_offsets;
  CUmemGenericAllocationHandle local_alloc_handle;
  std::vector<PeerMemorySharableHandle> this_mem_handles;
  std::vector<CUmemGenericAllocationHandle> all_handles;
  std::vector<PeerMemorySharableHandle> all_mem_handles;
};

struct IPCHandle {
  int socket = -1;
  std::string socket_name;
};


static std::mutex mu;
static bool is_wm_init = false;

static std::unordered_map<void*, std::unique_ptr<WMSPHandle>> wmsp_ptr_to_handle;

// WholeMemory Multiple Process mode data
static bool is_wmmp_init = false;

static std::unique_ptr<CollectiveCommunicator> cc_ptr;

static int dev_id_ = 0;

static int next_id_ = 0;
static size_t granularity_ = 0;

static pid_t self_pid_;
static std::vector<pid_t> local_pids_;
static std::vector<IPCHandle> recv_handles_;
static IPCHandle send_handle_;
static std::string temp_dir_for_unix_domain_sockets_;
static std::unordered_map<void*, std::unique_ptr<WMMPHandle>> wmmp_ptr_to_handle;

static std::unordered_set<WholeChunkedMemory_t> wmcmp_handle_set;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int CCRank() {
  return cc_ptr->Rank();
}

int CCSize() {
  return cc_ptr->Size();
}

void CCAllToAll(const void* send_buf, int send_size, void* recv_buf, int recv_size, const int* ranks, int rank_count) {
  cc_ptr->AllToAll(send_buf, send_size, recv_buf, recv_size, ranks, rank_count);
}
void CCAllToAllV(const void** send_bufs, const int* send_sizes, void** recv_bufs, const int* recv_sizes, const int* ranks, int rank_count) {
  cc_ptr->AllToAllV(send_bufs, send_sizes, recv_bufs, recv_sizes, ranks, rank_count);
}

void CCBarrier(const int* ranks, int rank_count) {
  cc_ptr->Barrier(ranks, rank_count);
}

void CheckRanksValid(const int* ranks, int rank_count) {
  int rank = CCRank();
  int size = CCSize();
  if (rank < 0 || rank >= size) {
    std::cerr << "[GetRankOffsetInRanks] rank=" << rank << ", not in range [0, " << size << ")" << std::endl;
    abort();
  }
  if (ranks == nullptr || rank_count == 0) return;
  std::set<int> rank_set;
  for (int i = 0; i < rank_count; i++) {
    if (ranks[i] < 0 || ranks[i] >= size) {
      std::cerr << "[CheckRanksValid] ranks[" << i << "]=" << ranks[i] << ", not in range [0, " << size << ")" << std::endl;
      abort();
    }
    rank_set.insert(ranks[i]);
  }
  if (rank_set.size() != (size_t)rank_count) {
    std::cerr << "[CheckRanksValid] unique rank count=" << rank_set.size() << " but rank_count=" << rank_count << std::endl;
    abort();
  }
  bool has_rank = false;
  for (int i = 0; i < rank_count; i++) {
    if (ranks[i] == rank) {
      has_rank = true;
      break;
    }
  }
  if (!has_rank) {
    std::cerr << "[CheckRanksValid] rank=" << rank << "not in ranks" << std::endl;
    abort();
  }
}

int GetRankOffsetInRanks(int rank, const int* ranks, int rank_count) {
  int size = CCSize();
  if (rank < 0 || rank >= size) {
    std::cerr << "[GetRankOffsetInRanks] rank=" << rank << ", not in range [0, " << size << ")" << std::endl;
    abort();
  }
  if (ranks == nullptr || rank_count == 0) {
    return rank;
  }
  for (int i = 0; i < rank_count; i++) {
    if (ranks[i] == rank) return i;
  }
  std::cerr << "[GetRankOffsetInRanks] rank=" << rank << ", not in ranks" << std::endl;
  abort();
  return -1;
}

int GetSizeInRanks(const int* ranks, int rank_count) {
  if (ranks == nullptr || rank_count == 0) {
    return CCSize();
  }
  return rank_count;
}

void CollBroadcastString(std::string* str, int root, const int* ranks, int rank_count) {
  auto len = CollBroadcast<size_t>(str->size(), root, ranks, rank_count);
  int size = CCSize();
  if (ranks == nullptr || rank_count == 0) rank_count = size;
  str->resize(len, ' ');
  std::string send_str;
  for (int i = 0; i < rank_count; i++) send_str.append(*str);
  std::string recv_str;
  recv_str.resize(len * rank_count);
  CCAllToAll(send_str.c_str(), len, &recv_str[0], len, ranks, rank_count);
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

size_t GetGranularityOfDevices(const int* dev_list, int dev_count) {
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

void StartCollOp(WMOPTypes wm_op) {
  CollCheckAllSame<WMOPTypes>(wm_op, nullptr, 0);
}

size_t GetGranularityOfAllRanks(int local_dev_id) {
  size_t granularity = GetGranularity(local_dev_id);
  std::vector<size_t> gv;
  CollAllGather<size_t>(granularity, &gv, nullptr, 0);
  int size = cc_ptr->Size();
  for (int i = 0; i < size; i++) {
    if (gv[i] > granularity) granularity = gv[i];
  }
  CollCheckAllSame(granularity, nullptr, 0);
  //if (cc_ptr->Rank() == 0) std::cerr << "Using granularity " << granularity << std::endl;
  return granularity;
}

void DistributePages(std::vector<size_t>* page_counts, size_t total_page_count, int wg_size) {
  page_counts->resize(wg_size, 0);
  for (int i = 0; i < wg_size; i++) {
    size_t start_page = total_page_count * i / wg_size;
    size_t end_page = (total_page_count * (i + 1)) / wg_size;
    size_t rank_page_count = end_page - start_page;
    page_counts->at(i) = rank_page_count;
  }
}

// for sender, create a socket.
void IPCCreateSocket(IPCHandle* handle, const char* name) {
  int server_fd;
  struct sockaddr_un servaddr{};

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

  if (bind(server_fd, (struct sockaddr *)&servaddr, SUN_LEN(&servaddr)) < 0) {
    std::cerr << "IPC failure: Binding socket failed" << std::endl;
    abort();
    return;
  }

  handle->socket_name = name;
  handle->socket = server_fd;
}

// for receiver
void IPCOpenSocket(IPCHandle* handle, const char* name) {
  WM_CHECK(handle != nullptr);
  int sock = 0;
  struct sockaddr_un cliaddr{};

  if ((sock = socket(AF_UNIX, SOCK_DGRAM, 0)) < 0) {
    std::cerr << "IPC failure: Socket creation error" << std::endl;
    abort();
    return;
  }
  bzero(&cliaddr, sizeof(cliaddr));
  cliaddr.sun_family = AF_UNIX;

  strcpy(cliaddr.sun_path, name);
  if (bind(sock, (struct sockaddr *)&cliaddr, sizeof(cliaddr)) < 0) {
    std::cerr << "IPC failure: Binding socket failed, name=" << name << std::endl;
    abort();
    return;
  }

  handle->socket = sock;
  handle->socket_name = name;
}

// for both sender and receiver
void IPCCloseSocket(IPCHandle* handle) {
  if (!handle) {
    std::cerr << "Closing null handle" << std::endl;
    abort();
  }

  if (!handle->socket_name.empty()) {
    assert(unlink(handle->socket_name.c_str()) == 0);
    handle->socket_name = "";
  }
  close(handle->socket);
  handle->socket = -1;
};

void IPCSendSharableHandle(IPCHandle* handle, int fd_to_send, const char* dst_name) {
  struct msghdr msg{};
  struct iovec iov[1];

  union {
    struct cmsghdr cm;
    char control[CMSG_SPACE(sizeof(int))];
  } control_un{};

  struct cmsghdr *cmptr;
  struct sockaddr_un cliaddr{};

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

  msg.msg_name = (void *)&cliaddr;
  msg.msg_namelen = sizeof(struct sockaddr_un);

  iov[0].iov_base = (void *)"";
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
int IPCRecvSharableHandle(IPCHandle* handle) {
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

  iov[0].iov_base = (void *)dummy_buffer;
  iov[0].iov_len = sizeof(dummy_buffer);

  msg.msg_iov = iov;
  msg.msg_iovlen = 1;

  if ((n = recvmsg(handle->socket, &msg, 0)) <= 0) {
    std::cerr << "IPC failure: Receiving data over socket failed" << std::endl;
    abort();
    return -1;
  }

  if (((cmptr = CMSG_FIRSTHDR(&msg)) != nullptr) &&
      (cmptr->cmsg_len == CMSG_LEN(sizeof(int)))) {
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
  WM_CU_CHECK(cuMemImportFromShareableHandle(&h, (void *)(uintptr_t)pmsh.fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
  return h;
}

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

std::string GetRecverSocketName(pid_t sender_pid, int src_rank, pid_t recver_pid, int dst_rank) {
  char temp[108];
  snprintf(temp,
           108,
           "%s/RP%uR%dfromP%uR%d",
           temp_dir_for_unix_domain_sockets_.c_str(),
           sender_pid,
           src_rank,
           recver_pid,
           dst_rank);
  return std::string(temp);
}

std::string GetSenderSocketName(pid_t sender_pid, int src_rank) {
  char temp[108];
  snprintf(temp, 108, "%s/SP%uR%dA", temp_dir_for_unix_domain_sockets_.c_str(), sender_pid, src_rank);
  return std::string(temp);
}

void CreateTempDirForUnixDomainSockets() {
  const char* sock_prefix = getenv("WHOLEMEMORY_TMPNAME");
  std::string wholememory_prefix_str = "wmtmp";
  if (sock_prefix != nullptr) {
    wholememory_prefix_str = sock_prefix;
  }
  wholememory_prefix_str.append(".XXXXXX");
  char template_str[128] = "/tmp/";
  strcpy(&template_str[strlen(template_str)], wholememory_prefix_str.c_str());
  if (cc_ptr->Rank() == 0) {
    char *tmp_dir = mkdtemp(template_str);
    //fprintf(stderr, "Creating tmp_dir=%s\n", tmp_dir);
    WM_CHECK(tmp_dir != nullptr);
    temp_dir_for_unix_domain_sockets_ = tmp_dir;
  }
  CollBroadcastString(&temp_dir_for_unix_domain_sockets_, 0, nullptr, 0);
}
void RemoveTempDirForUnixDomainSockets() {
  cc_ptr->Barrier();
  if (cc_ptr->Rank() == 0) {
    remove(temp_dir_for_unix_domain_sockets_.c_str());
  }
  cc_ptr->Barrier();
}

void OpenUnixDomainSockets() {
  int local_size = cc_ptr->Size();
  self_pid_ = getpid();
  CollAllGather<pid_t>(self_pid_, &local_pids_, nullptr, 0);
  recv_handles_.resize(local_size);
  for (int i = 0; i < local_size; i++) {
    int src_rank = i;
    int dst_rank = cc_ptr->Rank();
    pid_t src_pid = local_pids_[i];
    pid_t dst_pid = self_pid_;
    IPCOpenSocket(&recv_handles_[i], GetRecverSocketName(src_pid, src_rank, dst_pid, dst_rank).c_str());
  }
  IPCCreateSocket(&send_handle_, GetSenderSocketName(self_pid_, cc_ptr->Rank()).c_str());
  cc_ptr->Barrier();
}
void CloseUnixDomainSockets() {
  cc_ptr->Barrier();
  IPCCloseSocket(&send_handle_);
  for (int i = 0; i < cc_ptr->Size(); i++) {
    IPCCloseSocket(&recv_handles_[i]);
  }
  cc_ptr->Barrier();
}

void AllToAllMemoryHandlesByIPC(const std::vector<PeerMemorySharableHandle>& memory_handles,
                                std::vector<PeerMemorySharableHandle> *all_memory_handles,
                                std::vector<size_t> *memory_sizes,
                                const int *ranks,
                                int rank_count) {
  int wg_size = (ranks == nullptr || rank_count == 0) ? cc_ptr->Size() : rank_count;
  WM_CHECK((size_t) wg_size + 1 == memory_sizes->size());
  for (int i = 0; i < wg_size; i++) {
    pid_t src_pid = self_pid_;
    int src_rank = cc_ptr->Rank();
    int dst_rank = (ranks == nullptr || rank_count == 0) ? i : ranks[i];
    pid_t dst_pid = local_pids_[dst_rank];
    if (memory_handles[i].fd >= 0) {
      IPCSendSharableHandle(&send_handle_,
                            memory_handles[i].fd,
                            GetRecverSocketName(src_pid, src_rank, dst_pid, dst_rank).c_str());
    }
  }
  cc_ptr->Barrier();
  all_memory_handles->resize(wg_size);
  for (int i = 0; i < wg_size; i++) {
    if (memory_sizes->at(i + 1) - memory_sizes->at(i) != 0) {
      int src_local_rank = (ranks == nullptr || rank_count == 0) ? i : ranks[i];
      all_memory_handles->at(i).fd = IPCRecvSharableHandle(&recv_handles_[src_local_rank]);
    } else {
      all_memory_handles->at(i).fd = -1;
    }
  }
  cc_ptr->Barrier();
  for (int i = 0; i < wg_size; i++) {
    if (memory_handles[i].fd >= 0) {
      WM_CHECK(close(memory_handles[i].fd) == 0);
    }
  }
}

}

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
  std::vector<WholeChunkedMemoryHandle*> dev_handles;

  std::vector<int> ranks;
  std::vector<int> dev_list;

  void* local_mem_ptr = nullptr;
  cudaIpcMemHandle_t local_ipc_handle;
  std::vector<cudaIpcMemHandle_t> all_mem_handles;

  WholeChunkedMemoryHandle* GetDeviceHandle(int dev_id) {
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
      WholeChunkedMemoryHandle* h;
      WM_CUDA_CHECK(cudaMalloc((void**)&h, sizeof(WholeChunkedMemoryHandle)));
      WM_CUDA_CHECK(cudaMemcpy(h, &cpu_handle, sizeof(WholeChunkedMemoryHandle), cudaMemcpyHostToDevice));
      WM_CUDA_CHECK(cudaDeviceSynchronize());
      dev_handles[dev_id] = h;
      WM_CUDA_CHECK(cudaSetDevice(current_dev));
    }
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static void WmmpFinalizeLocked();
static void WmspFreeLocked(void *ptr);
static void WmmpFreeLocked(void *ptr);
void WcmmpFreeLocked(WholeChunkedMemory_t wcmt);

void WholeMemoryInit() {
  std::unique_lock<std::mutex> lock(mu);
  WM_CHECK(!is_wm_init);
  WM_CU_CHECK(cuInit(0));
  is_wm_init = true;
}
void WholeMemoryFinalize() {
  std::unique_lock<std::mutex> lock(mu);
  WM_CHECK(is_wm_init);
  if (is_wmmp_init) {
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
  WMSPHandle* h = new WMSPHandle;
  h->is_host = false;
  h->dev_list = dev_list_vec;
  h->total_memory_size = size;
  if (size >= 16LL * 1024LL * 1024LL * 1024LL) granularity = 512LL * 1024LL * 1024LL;
  h->total_aligned_memory_size = AlignUp(size, granularity);
  h->granularity = granularity;
  size_t page_count = h->total_aligned_memory_size / granularity;
  WM_CU_CHECK(cuMemAddressReserve(&h->whole_memory_cuptr, h->total_aligned_memory_size, granularity, 0, 0));
  h->handles.resize(h->dev_list.size());
  for (size_t i = 0; i < dev_list_vec.size(); i++) {
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

    WM_CU_CHECK(cuMemMap(h->whole_memory_cuptr + start_page * granularity, current_page_count * granularity, 0, h->handles[i], 0));

    CUmemAccessDesc madesc[dev_list_vec.size()];
    for (size_t ii = 0; ii < dev_list_vec.size(); ii++) {
      madesc[ii].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      madesc[ii].location.id = dev_list_vec[ii];
      madesc[ii].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    WM_CU_CHECK(cuMemSetAccess(h->whole_memory_cuptr + start_page * granularity, current_page_count * granularity, &madesc[0], dev_list_vec.size()));
  }
  h->whole_memory_ptr = (void*) h->whole_memory_cuptr;
  *ptr = h->whole_memory_ptr;
  wmsp_ptr_to_handle.emplace(*ptr, h);
}
void WmspMallocHost(void** ptr, size_t size) {
  std::unique_lock<std::mutex> lock(mu);
  WMSPHandle* h = new WMSPHandle;
  h->is_host = true;
  h->total_memory_size = h->total_aligned_memory_size = size;
  h->host_ptr = malloc(size);
  WM_CUDA_CHECK(cudaHostRegister(h->host_ptr, size, cudaHostRegisterDefault));
  WM_CUDA_CHECK(cudaHostGetDevicePointer(&h->whole_memory_ptr, h->host_ptr, 0));
  *ptr = h->whole_memory_ptr;
  wmsp_ptr_to_handle.emplace(*ptr, h);
}
static void WmspFreeLocked(void *ptr) {
  auto it = wmsp_ptr_to_handle.find(ptr);
  WM_CHECK(it != wmsp_ptr_to_handle.end());
  WMSPHandle* h = it->second.get();
  if (h->is_host) {
    WM_CUDA_CHECK(cudaHostUnregister(h->host_ptr));
    free(h->host_ptr);
    wmsp_ptr_to_handle.erase(ptr);
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
  wmsp_ptr_to_handle.erase(ptr);
}
void WmspFree(void *ptr) {
  std::unique_lock<std::mutex> lock(mu);
  WmspFreeLocked(ptr);
}
void WmmpInit(int group_rank, int group_size, CollectiveCommunicator *collective_communicator) {
  std::unique_lock<std::mutex> lock(mu);
  WM_CHECK(!is_wmmp_init);
  WM_CHECK(is_wm_init);
  if (collective_communicator == nullptr) {
    cc_ptr = std::make_unique<DomainSocketCommunicator>();
  } else {
    cc_ptr.reset(collective_communicator);
  }
  cc_ptr->SetRankAndSize(group_rank, group_size);
  StartCollOp(WM_OP_STARTING);
  CreateTempDirForUnixDomainSockets();
  WM_CUDA_CHECK(cudaGetDevice(&dev_id_));
  granularity_ = GetGranularityOfAllRanks(dev_id_);
  is_wmmp_init = true;
}
static void WmmpFinalizeLocked() {
  StartCollOp(WM_OP_ENDING);
  std::map<int, void*> remaining_handles;
  for (auto & handle : wmmp_ptr_to_handle) {
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
    WcmmpFreeLocked((WholeChunkedMemory_t)it->second);
    remaining_handles.erase(id);
  }
  RemoveTempDirForUnixDomainSockets();
  cc_ptr.reset();
  is_wmmp_init = false;
}
void WmmpFinalize() {
  std::unique_lock<std::mutex> lock(mu);
  WmmpFinalizeLocked();
}
static int GetIdLocked() {
  return next_id_++;
}
static std::string CollWmmpGetSharedFileName(int shmem_id, const int* ranks, int rank_count) {
  std::string name;
  pid_t pid = getpid();
  pid = CollBroadcast(pid, 0, ranks, rank_count);
  name = "wgmphs_pid_";
  name += std::to_string(pid);
  name += "_mid_";
  name += std::to_string(shmem_id);
  return name;
}
static void WmmpMallocHostLocked(void **ptr, size_t size, const int *ranks, int rank_count) {
  WMMPHandle* h = new WMMPHandle;
  int local_rank_index = GetRankOffsetInRanks(cc_ptr->Rank(), ranks, rank_count);
  bool is_first_rank = (local_rank_index == 0);
  if (ranks == nullptr || rank_count == 0) {
    rank_count = cc_ptr->Size();
    h->ranks.resize(rank_count);
    for (int i = 0; i < rank_count; i++) h->ranks[i] = i;
  } else {
    h->ranks.resize(rank_count);
    for (int i = 0; i < rank_count; i++) {
      h->ranks[i] = ranks[i];
    }
  }
  StartCollOp(WM_OP_ALLOCATING);
  h->is_host = true;
  h->id = GetIdLocked();
  CollCheckAllSame(h->id, ranks, rank_count);
  h->total_memory_size = h->total_aligned_memory_size = size;
  std::string shm_name = CollWmmpGetSharedFileName(h->id, ranks, rank_count);
  int shm_fd = -1;
  if (is_first_rank) {
    shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    if (shm_fd < 0) {
      std::cerr << "Rank=" << cc_ptr->Rank() << ", create shm_name=" << shm_name << ", failed." << std::endl;
      abort();
    }
    WM_CHECK(ftruncate(shm_fd, h->total_memory_size) == 0);
    cc_ptr->Barrier(h->ranks.data(), h->ranks.size());
  } else {
    cc_ptr->Barrier(h->ranks.data(), h->ranks.size());
    shm_fd = shm_open(shm_name.c_str(), O_RDWR, S_IRUSR | S_IWUSR);
    if (shm_fd < 0) {
      std::cerr << "Rank=" << cc_ptr->Rank() << ", open shm_name=" << shm_name << ", failed." << std::endl;
      abort();
    }
  }
  cc_ptr->Barrier(h->ranks.data(), h->ranks.size());
  h->host_ptr = mmap(nullptr, h->total_memory_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  WM_CHECK(h->host_ptr != (void*)-1);
  WM_CHECK(close(shm_fd) == 0);
  WM_CUDA_CHECK(cudaHostRegister(h->host_ptr, size, cudaHostRegisterDefault));
  WM_CUDA_CHECK(cudaHostGetDevicePointer(&h->whole_memory_ptr, h->host_ptr, 0));
  WM_CHECK(h->whole_memory_ptr == h->host_ptr);
  *ptr = h->whole_memory_ptr;
  wmmp_ptr_to_handle.emplace(*ptr, h);
  //std::cout << "Allocating Host ptr=" << *ptr << ", memory_count=" << wmmp_ptr_to_handle.size() << std::endl;
}
static void WmmpMallocLocked(void **ptr, size_t size, const int *ranks, int rank_count) {
  WMMPHandle* h = new WMMPHandle;
  int local_rank_index = GetRankOffsetInRanks(cc_ptr->Rank(), ranks, rank_count);
  if (ranks == nullptr || rank_count == 0) {
    rank_count = cc_ptr->Size();
    h->ranks.resize(rank_count);
    for (int i = 0; i < rank_count; i++) h->ranks[i] = i;
  } else {
    h->ranks.resize(rank_count);
    for (int i = 0; i < rank_count; i++) {
      h->ranks[i] = ranks[i];
    }
  }

  StartCollOp(WM_OP_ALLOCATING);
  h->is_host = false;
  h->id = GetIdLocked();
  CollCheckAllSame(h->id, ranks, rank_count);
  CollAllGather(dev_id_, &h->dev_list, ranks, rank_count);
  h->total_memory_size = size;
  size_t granularity = granularity_;
  if (size >= 16LL * 1024LL * 1024LL * 1024LL) granularity = 512LL * 1024LL * 1024LL;
  h->total_aligned_memory_size = AlignUp(size, granularity);
  size_t total_page_count = h->total_aligned_memory_size / granularity;
  std::vector<size_t> page_counts;
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
    h->local_alloc_handle = CreateCUMem(local_memory_size, dev_id_);
    for (int i = 0; i < rank_count; i++) {
      h->this_mem_handles[i] = CreateSharableHandle(h->local_alloc_handle);
    }
  } else {
    for (int i = 0; i < rank_count; i++) {
      h->this_mem_handles[i].fd = -1;
    }
  }
  OpenUnixDomainSockets();
  AllToAllMemoryHandlesByIPC(h->this_mem_handles, &h->all_mem_handles, &h->memory_offsets, ranks, rank_count);
  CloseUnixDomainSockets();
  WM_CHECK(h->all_mem_handles.size() == (size_t)rank_count);
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
  madesc.location.id = dev_id_;
  madesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  WM_CU_CHECK(cuMemSetAccess(h->whole_memory_cuptr, h->total_aligned_memory_size, &madesc, 1));
  h->whole_memory_ptr = (void*)h->whole_memory_cuptr;
  *ptr = h->whole_memory_ptr;
  wmmp_ptr_to_handle.emplace(*ptr, h);
  //std::cout << "Allocating Device ptr=" << *ptr << ", memory_count=" << wmmp_ptr_to_handle.size() << std::endl;
}
size_t WmmpAggregateSize(size_t local_size, size_t* offset, const int* ranks, int rank_count) {
  StartCollOp(WM_OP_COMPUTINT_SIZE);
  std::vector<size_t> sizes;
  CollAllGather(local_size, &sizes, ranks, rank_count);
  size_t total_size = 0;
  int rank_idx = GetRankOffsetInRanks(cc_ptr->Rank(), ranks, rank_count);
  for (size_t i = 0; i < sizes.size(); i++) {
    if (offset != nullptr && (size_t)rank_idx == i) {
      *offset = total_size;
    }
    total_size += sizes[i];
  }
  return total_size;
}
void WmmpMalloc(void **ptr, size_t size, const int *ranks, int rank_count) {
  std::unique_lock<std::mutex> lock(mu);
  CheckRanksValid(ranks, rank_count);
  WmmpMallocLocked(ptr, size, ranks, rank_count);
}
void WmmpMallocHost(void **ptr, size_t size, const int *ranks, int rank_count) {
  std::unique_lock<std::mutex> lock(mu);
  CheckRanksValid(ranks, rank_count);
  WmmpMallocHostLocked(ptr, size, ranks, rank_count);
}
static void WmmpFreeLocked(void *ptr) {
  //std::cout << "Freeing ptr=" << ptr << ", memory_count=" << wmmp_ptr_to_handle.size() << std::endl;
  auto it = wmmp_ptr_to_handle.find(ptr);
  WM_CHECK(it != wmmp_ptr_to_handle.end());
  WMMPHandle* h = it->second.get();
  StartCollOp(WM_OP_DEALLOCATING);
  CollCheckAllSame(h->id, h->ranks.data(), h->ranks.size());

  if (h->is_host) {
    int local_rank_index = GetRankOffsetInRanks(cc_ptr->Rank(), h->ranks.data(), h->ranks.size());
    bool is_first_rank = (local_rank_index == 0);
    WM_CUDA_CHECK(cudaHostUnregister(h->host_ptr));
    WM_CHECK(munmap(h->host_ptr, h->total_memory_size) == 0);
    cc_ptr->Barrier(h->ranks.data(), h->ranks.size());
    std::string shm_name = CollWmmpGetSharedFileName(h->id, h->ranks.data(), h->ranks.size());
    if (is_first_rank) {
      WM_CHECK(shm_unlink(shm_name.c_str()) == 0);
    }
    cc_ptr->Barrier(h->ranks.data(), h->ranks.size());
    wmmp_ptr_to_handle.erase(ptr);
    return;
  }

  for (size_t i = 0; i < h->ranks.size(); i++) {
    size_t mem_size = h->memory_offsets[i + 1] - h->memory_offsets[i];
    if (mem_size > 0) {
      WM_CU_CHECK(cuMemUnmap(h->whole_memory_cuptr + h->memory_offsets[i], mem_size));
      WM_CU_CHECK(cuMemRelease(h->all_handles[i]));
    }
  }
  // Check all memory unmapped.
  cc_ptr->Barrier(h->ranks.data(), h->ranks.size());
  if (h->local_memory_size > 0) {
    WM_CU_CHECK(cuMemRelease(h->local_alloc_handle));
  }
  WM_CU_CHECK(cuMemAddressFree(h->whole_memory_cuptr, h->total_aligned_memory_size));

  wmmp_ptr_to_handle.erase(ptr);
}
void WmmpFree(void *ptr) {
  std::unique_lock<std::mutex> lock(mu);
  // skip free after lib finalize.
  if (!is_wmmp_init) return;
  WmmpFreeLocked(ptr);
}

bool WmmpCanAccessFrom(void* ptr, int dev) {
  if (dev < -1) return false;
  std::unique_lock<std::mutex> lock(mu);
  auto it = wmmp_ptr_to_handle.find(ptr);
  if (it == wmmp_ptr_to_handle.end()) {
    return false;
  }
  WMMPHandle* h = it->second.get();
  if (dev == -1) {
    return h->is_host;
  }
  for (int devid : h->dev_list) {
    if (devid == dev) return true;
  }
  return false;
}

void WmmpBarrier(const int* ranks, int rank_count) {
  cc_ptr->Barrier(ranks, rank_count);
}

void WmmpAllToAll(const void* send_buf, int send_size, void* recv_buf, int recv_size, const int* ranks, int rank_count) {
  cc_ptr->AllToAll(send_buf, send_size, recv_buf, recv_size, ranks, rank_count);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void WcmmpMalloc(WholeChunkedMemory_t* pwcmt, size_t size, size_t min_granularity, const int* ranks, int rank_count) {
  std::unique_lock<std::mutex> lock(mu);
  CheckRanksValid(ranks, rank_count);
  int real_rank_count = rank_count;

  WholeChunkedMemory* h = new WholeChunkedMemory;
  int local_rank_index = GetRankOffsetInRanks(cc_ptr->Rank(), ranks, rank_count);
  if (ranks == nullptr || rank_count == 0) {
    real_rank_count = cc_ptr->Size();
    h->ranks.resize(real_rank_count);
    for (int i = 0; i < real_rank_count; i++) h->ranks[i] = i;
  } else {
    h->ranks.resize(real_rank_count);
    for (int i = 0; i < real_rank_count; i++) {
      h->ranks[i] = ranks[i];
    }
  }

  StartCollOp(WM_OP_ALLOCATING);
  const int64_t kWholeChunkedMemoryAllocating = 0xE601EC6CC8EDEEE0LL;
  CollCheckAllSame(kWholeChunkedMemoryAllocating, ranks, rank_count);
  h->id = GetIdLocked();
  CollCheckAllSame(h->id, ranks, rank_count);
  CollAllGather(dev_id_, &h->dev_list, ranks, rank_count);
  CollCheckAllSame(size, ranks, rank_count);;

  if (min_granularity == 0) min_granularity = 256;
  if (min_granularity < 256 && 256 % min_granularity == 0) min_granularity = 256;
  while (min_granularity % 16 != 0) {
    min_granularity *= 2;
  }
  h->min_granularity = min_granularity;
  h->raw_size = size;
  h->alloc_size = AlignUp(h->raw_size, min_granularity * real_rank_count);
  if (h->alloc_size == 0) h->alloc_size = min_granularity * real_rank_count;
  h->cpu_handle.chunk_count = real_rank_count;
  h->cpu_handle.chunk_size = h->alloc_size / real_rank_count;

  size_t malloc_size = h->cpu_handle.chunk_size;
  if (malloc_size >= 2LL * 1024LL * 1024LL * 1024LL) malloc_size = AlignUp(malloc_size, 512LL * 1024LL * 1024LL);
  WM_CUDA_CHECK(cudaMalloc((void**)&h->local_mem_ptr, malloc_size));
  WM_CUDA_CHECK(cudaIpcGetMemHandle(&h->local_ipc_handle, h->local_mem_ptr));
  CollAllGather(h->local_ipc_handle, &h->all_mem_handles, ranks, rank_count);
  for (int i = 0; i < real_rank_count; i++) {
    if (i == local_rank_index) {
      h->cpu_handle.chunked_ptrs[i] = h->local_mem_ptr;
    } else {
      WM_CUDA_CHECK(cudaIpcOpenMemHandle(&h->cpu_handle.chunked_ptrs[i], h->all_mem_handles[i], cudaIpcMemLazyEnablePeerAccess));
    }
  }
  *pwcmt = h;
  wmcmp_handle_set.insert(h);
}

void WcmmpFreeLocked(WholeChunkedMemory_t wcmt) {
  auto it = wmcmp_handle_set.find(wcmt);
  WM_CHECK(it != wmcmp_handle_set.end());
  WholeChunkedMemory* h = wcmt;
  StartCollOp(WM_OP_DEALLOCATING);
  CollCheckAllSame(h->id, h->ranks.data(), h->ranks.size());

  int local_rank_index = GetRankOffsetInRanks(cc_ptr->Rank(), h->ranks.data(), h->ranks.size());

  for (int i = 0; i < (int)h->ranks.size(); i++) {
    if (i != local_rank_index) {
      WM_CUDA_CHECK(cudaIpcCloseMemHandle(h->cpu_handle.chunked_ptrs[i]));
    }
  }
  // Check all memory unmapped.
  cc_ptr->Barrier(h->ranks.data(), h->ranks.size());
  WM_CUDA_CHECK(cudaFree(h->local_mem_ptr));

  wmcmp_handle_set.erase(wcmt);
}

void WcmmpFree(WholeChunkedMemory_t wcmt) {
  std::unique_lock<std::mutex> lock(mu);
  if (!is_wmmp_init) return;
  WcmmpFreeLocked(wcmt);
}

WholeChunkedMemoryHandle* GetDeviceChunkedHandle(WholeChunkedMemory_t wcmt, int dev_id) {
  std::unique_lock<std::mutex> lock(mu);
  auto it = wmcmp_handle_set.find(wcmt);
  WM_CHECK(it != wmcmp_handle_set.end());
  WholeChunkedMemory* h = wcmt;
  return h->GetDeviceHandle(dev_id);
}

size_t WmmpCollOffsetAndSize(size_t total_size, size_t* local_size, const int* ranks, int rank_count) {
  int local_rank_index = GetRankOffsetInRanks(cc_ptr->Rank(), ranks, rank_count);
  int all_rank_count = rank_count;
  if (ranks == nullptr || rank_count == 0) all_rank_count = cc_ptr->Size();
  size_t start = total_size * local_rank_index / all_rank_count;
  size_t end = total_size * (local_rank_index + 1) / all_rank_count;
  if (local_size != nullptr) * local_size = end - start;
  return start;
}

}