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
#include "memory_handle.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <mutex>
#include <vector>

#include "cuda_macros.hpp"
#include "error.hpp"
#include "integer_utils.hpp"
#include "logger.hpp"
#include "wholememory/communicator.hpp"
#include "wholememory/global_reference.h"
#include "wholememory/wholememory.h"

#include "system_info.hpp"
#ifdef WITH_NVSHMEM_SUPPORT
#include "nvshmem.h"
#include "nvshmemx.h"

#endif
namespace wholememory {

enum wm_memory_op : int32_t {
  WM_MEM_OP_CREATE = 0xEEEEE,
  WM_MEM_OP_EXCHANGE_ID,
  WM_MEM_OP_DESTROY,
};

class wholememory_impl {
 public:
  wholememory_impl(wholememory_handle_t wholememory_handle,
                   size_t total_size,
                   wholememory_comm_t comm,
                   wholememory_memory_type_t memory_type,
                   wholememory_memory_location_t memory_location,
                   size_t data_granularity,
                   size_t* rank_entry_partition)
    : handle_(wholememory_handle),
      comm_(comm),
      type_(memory_type),
      location_(memory_location),
      total_size_(total_size),
      data_granularity_(data_granularity)
  {
    if (rank_entry_partition != nullptr) {
      rank_partition_strategy_.partition_sizes_.resize(comm_->world_size, 0);
      rank_partition_strategy_.partition_offsets_.resize(comm_->world_size + 1, 0);
      for (int i = 0; i < comm_->world_size; i++) {
        rank_partition_strategy_.partition_sizes_[i] = rank_entry_partition[i] * data_granularity_;
        rank_partition_strategy_.partition_offsets_[i + 1] =
          rank_partition_strategy_.partition_offsets_[i] +
          rank_entry_partition[i] * data_granularity_;
      }
    }
    distrubuted_backend_ = WHOLEMEMORY_DB_NCCL;
  }
  wholememory_impl()                         = delete;
  wholememory_impl(const wholememory_impl&)  = delete;
  wholememory_impl(const wholememory_impl&&) = delete;

  virtual ~wholememory_impl() = default;

  [[nodiscard]] wholememory_memory_type_t get_type() const { return type_; }
  [[nodiscard]] wholememory_memory_location_t get_location() const { return location_; }
  [[nodiscard]] wholememory_comm_t get_comm() const { return comm_; }
  [[nodiscard]] wholememory_distributed_backend_t get_distributed_backend() const
  {
    return distrubuted_backend_;
  }

  [[nodiscard]] size_t total_size() const { return total_size_; }
  [[nodiscard]] size_t data_granularity() const { return data_granularity_; }
  virtual void create_memory()           = 0;
  virtual void destroy_memory() noexcept = 0;
  [[nodiscard]] virtual void* get_continuous_mapping_pointer() const noexcept { return nullptr; }
  [[nodiscard]] virtual wholememory_gref_t get_global_reference() const noexcept
  {
    wholememory_gref_t gref{};
    gref.pointer    = nullptr;
    gref.stride     = 0;
    gref.world_size = comm_->world_size;
    return gref;
  }
  virtual bool contains_pointer(const void* ptr) const = 0;
  virtual void get_local_memory(void** local_ptr, size_t* local_size, size_t* local_offset) const
  {
    if (local_ptr != nullptr) *local_ptr = local_partition_memory_pointer_;
    if (local_size != nullptr) *local_size = get_local_size();
    if (local_offset != nullptr) *local_offset = get_local_offset();
    if (location_ == WHOLEMEMORY_ML_HOST && (type_ == WHOLEMEMORY_MT_CONTINUOUS) &&
        (!(comm_->is_intranode()))) {
      WHOLEMEMORY_WARN(
        " Multi-node continuous type wholememory can only be accessed by GPU threads but not CPU "
        "threads, regardless of whether the location of wholememory is host.");
    }
  }
  virtual bool get_rank_memory(void** rank_memory_ptr,
                               size_t* rank_memory_size,
                               size_t* rank_memory_offset,
                               int rank) const noexcept
  {
    *rank_memory_ptr    = nullptr;
    *rank_memory_size   = 0;
    *rank_memory_offset = 0;
    return false;
  }
  [[nodiscard]] virtual size_t get_partition_stride() const
  {
    return rank_partition_strategy_.partition_mem_stride;
  }
  [[nodiscard]] size_t get_local_size() const
  {
    return rank_partition_strategy_.partition_sizes_[comm_->world_rank];
  }
  [[nodiscard]] size_t get_local_offset() const
  {
    return rank_partition_strategy_.partition_offsets_[comm_->world_rank];
  }
  std::vector<size_t> get_rank_sizes() const { return rank_partition_strategy_.partition_sizes_; }
  std::vector<size_t> get_rank_offsets() const
  {
    return rank_partition_strategy_.partition_offsets_;
  }

 protected:
  // In WholeMemory, memory is first allocated by one or all ranks, and then partition the whole
  // memory to each rank. Each rank can direct access its partition of memory and is response for
  // memory operations on that. In some WholeMemory types like CONTINUOUS or CHUNKED, memory of
  // other ranks may also be mapped to current rank. So current rank may access the whole memory,
  // but in a parallel loading case, it is not its response to process memory other than the
  // partition of memory determined by memory partition strategy.
  //
  // Memory partitioning is decoupled with memory allocation, memory allocation may have different
  // strategies.
  //
  // The following 3 functions are for memory allocation strategies
  // first rank responsible for all memory allocation, continuous or chunked host shared memory may
  // use this mode.
  void first_rank_allocate_all_strategy();
  // each rank allocate different size, chunked device memory or nccl memory may use this
  // mode. If rank_entry_partition isn't set, each rank allocate exactly the same size.
  void each_rank_different_chunk_strategy();
  // each rank allocate a multiple of pages, and map the whole memory by page, continuous device
  // memory use this mode.
  void each_rank_multiple_page_strategy();

  // For now, memory rank partitioning strategy is the same for all WholeMemory types.
  // Each rank is response for memory of size local_mem_size starting from local_mem_offset.
  // Local_mem_size can be got by calling get_local_size(), and local_mem_offset can be got
  // by calling get_local_offset(). rank_partition_strategy_.partition_sizes_ and
  // rank_partition_strategy_.partition_offsets_ record the memory size and memory offset of
  // all ranks.
  void generate_rank_partition_strategy();

  /*
   *  ++---------------------------------------------------------------------------------------++
   *  ||     Type     ||     CONTINUOUS      ||      CHUNKED        ||       DISTRIBUTED       ||
   *  ++--------------+------------------------------------------------------------------------++
   *  ||   Location   ||  DEVICE  |   HOST   ||  DEVICE  |   HOST   ||   DEVICE   |    HOST    ||
   *  ++---------------------------------------------------------------------------------------++
   *  || Allocated by ||   EACH   |  FIRST   ||   EACH   |  FIRST   ||    EACH    |    EACH    ||
   *  ++---------------------------------------------------------------------------------------++
   *  || Allocate API ||  Driver  |   Host   ||  Runtime |   Host   ||   Runtime  |   Runtime  ||
   *  ++---------------------------------------------------------------------------------------++
   *  ||  IPC Mapping || Unix fd  |   mmap   ||  cudaIpc |   mmap   || No IPC map | No IPC map ||
   *  ++---------------------------------------------------------------------------------------++
   */

  wholememory_handle_t handle_;
  wholememory_comm_t comm_;
  wholememory_memory_type_t type_;
  wholememory_memory_location_t location_;
  wholememory_distributed_backend_t distrubuted_backend_;
  // raw user input size, real allocation may be larger than this.
  size_t total_size_;
  size_t data_granularity_;

  struct alloc_strategy {
    size_t total_alloc_size = 0;
    size_t local_alloc_size = 0;
    size_t alignment        = 0;
    std::vector<size_t> alloc_offsets;
    std::vector<size_t> alloc_sizes;
  } alloc_strategy_;

  struct partition_strategy {
    std::vector<size_t> partition_sizes_;
    std::vector<size_t> partition_offsets_;
    size_t partition_mem_stride = 0;
    bool same_chunk;
  } rank_partition_strategy_;

  void* local_partition_memory_pointer_ = nullptr;

  void get_rank_partition_info(size_t* rank_mem_size,
                               size_t* rank_mem_start,
                               int rank) const noexcept
  {
    WHOLEMEMORY_CHECK_NOTHROW(rank >= 0 && rank <= comm_->world_size);
    if (rank_mem_size != nullptr) *rank_mem_size = rank_partition_strategy_.partition_sizes_[rank];
    if (rank_mem_start != nullptr)
      *rank_mem_start = rank_partition_strategy_.partition_offsets_[rank];
  }

  static constexpr size_t HUGE_PAGE_THRESHOLD = 16UL * 1024UL * 1024UL * 1024UL;
  static constexpr size_t HUGE_PAGE_SIZE      = 512UL * 1024UL * 1024UL;
};

struct wholememory_vma_data {
  wholememory_handle_t wholememory_handle;
  const void* start_ptr;
  size_t mem_block_size;
};
// mutex to protect wholememory_vma_map
static std::mutex wholememory_vma_mu;
// map to store memory regions that are in wholememory.
// Key is the tail of a valid memory, the byte of the key is not in wholememory.
// The reason to use tail is that we can check if a pointer is in wholememory by upper_bound.
static std::map<uint64_t, wholememory_vma_data> wholememory_vma_map;

wholememory_handle_t wholememory_get_handle(const void* ptr)
{
  std::unique_lock<std::mutex> vma_lock(wholememory_vma_mu);
  uint64_t int_ptr = reinterpret_cast<uint64_t>(ptr);
  auto it          = wholememory_vma_map.upper_bound(int_ptr);
  if (it == wholememory_vma_map.end()) return nullptr;
  wholememory_handle_t wm_h = it->second.wholememory_handle;
  if (wm_h->impl->contains_pointer(ptr)) { return wm_h; }
  return nullptr;
}

static void register_wholememory_vma_range_locked(const void* ptr,
                                                  size_t mem_block_size,
                                                  wholememory_handle_t wm_h)
{
  WHOLEMEMORY_CHECK(ptr != nullptr);
  WHOLEMEMORY_CHECK(wm_h != nullptr);
  WHOLEMEMORY_CHECK(wm_h->impl->contains_pointer(ptr));
  uint64_t int_start_ptr = reinterpret_cast<uint64_t>(ptr);
  uint64_t int_tail_ptr  = int_start_ptr + mem_block_size;
  WHOLEMEMORY_CHECK(wm_h->impl->contains_pointer(reinterpret_cast<void*>(int_tail_ptr - 1)));
  WHOLEMEMORY_CHECK(wholememory_vma_map.find(int_tail_ptr) == wholememory_vma_map.end());
  wholememory_vma_data vma_data;
  vma_data.wholememory_handle = wm_h;
  vma_data.start_ptr          = ptr;
  vma_data.mem_block_size     = mem_block_size;
  wholememory_vma_map.insert(std::pair<uint64_t, wholememory_vma_data>(int_tail_ptr, vma_data));
  // No overlap with previous block
  auto it1 = wholememory_vma_map.find(int_tail_ptr);
  if (it1 != wholememory_vma_map.begin()) {
    --it1;
    WHOLEMEMORY_CHECK(reinterpret_cast<uint64_t>(it1->second.start_ptr) +
                        it1->second.mem_block_size <=
                      int_start_ptr);
  }
  // No overlap with next block
  auto it2 = wholememory_vma_map.find(int_tail_ptr);
  if (it2 != wholememory_vma_map.find(wholememory_vma_map.rbegin()->first)) {
    ++it2;
    WHOLEMEMORY_CHECK(reinterpret_cast<uint64_t>(it2->second.start_ptr) >= int_tail_ptr);
  }
}

static void unregister_wholememory_vma_range_locked(const void* ptr,
                                                    size_t mem_block_size,
                                                    wholememory_handle_t wm_h) noexcept
{
  try {
    WHOLEMEMORY_CHECK(wm_h != nullptr);
    WHOLEMEMORY_CHECK(wm_h->impl->contains_pointer(ptr));
    uint64_t int_start_ptr = reinterpret_cast<uint64_t>(ptr);
    uint64_t int_tail_ptr  = int_start_ptr + mem_block_size;
    WHOLEMEMORY_CHECK(wm_h->impl->contains_pointer(reinterpret_cast<void*>(int_tail_ptr - 1)));
    auto it = wholememory_vma_map.find(int_tail_ptr);
    WHOLEMEMORY_CHECK(it != wholememory_vma_map.end());
    WHOLEMEMORY_CHECK(it->second.wholememory_handle == wm_h);
    WHOLEMEMORY_CHECK(it->second.start_ptr == ptr);
    WHOLEMEMORY_CHECK(it->second.mem_block_size == mem_block_size);
    wholememory_vma_map.erase(int_tail_ptr);
  } catch (const wholememory::logic_error& le) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", le.what());
  }
}

// Implementation for distributed memory that don't need global map.
// Each rank allocate for its own memory and don't need communication during creation.
// for DISTRIBUTED memory type with DEVICE or HOST location
class distributed_wholememory_impl : public wholememory_impl {
 public:
  distributed_wholememory_impl(wholememory_handle_t wholememory_handle,
                               size_t total_size,
                               wholememory_comm_t comm,
                               wholememory_memory_type_t memory_type,
                               wholememory_memory_location_t memory_location,
                               size_t data_granularity,
                               size_t* rank_entry_partition)
    : wholememory_impl(wholememory_handle,
                       total_size,
                       comm,
                       memory_type,
                       memory_location,
                       data_granularity,
                       rank_entry_partition)
  {
    WHOLEMEMORY_CHECK(type_ == WHOLEMEMORY_MT_DISTRIBUTED || type_ == WHOLEMEMORY_MT_HIERARCHY);
  }
  void create_memory() override
  {
    generate_rank_partition_strategy();
    each_rank_different_chunk_strategy();
    create_local_cuda_runtime_memory();
    register_private_memory();
  }
  void destroy_memory() noexcept override
  {
    unregister_private_memory();
    destroy_local_cuda_runtime_memory();
  }
  bool contains_pointer(const void* ptr) const override
  {
    uint64_t int_ptr       = reinterpret_cast<uint64_t>(ptr);
    uint64_t int_start_ptr = reinterpret_cast<uint64_t>(no_ipc_handle_.local_alloc_mem_ptr);
    return int_ptr >= int_start_ptr && int_ptr < int_start_ptr + alloc_strategy_.local_alloc_size;
  }

 protected:
  void register_private_memory()
  {
    std::unique_lock<std::mutex> vma_lock(wholememory_vma_mu);
    if (alloc_strategy_.local_alloc_size > 0) {
      register_wholememory_vma_range_locked(
        no_ipc_handle_.local_alloc_mem_ptr, alloc_strategy_.local_alloc_size, handle_);
    }
  }
  void unregister_private_memory() noexcept
  {
    std::unique_lock<std::mutex> vma_lock(wholememory_vma_mu);
    if (alloc_strategy_.local_alloc_size > 0) {
      unregister_wholememory_vma_range_locked(
        no_ipc_handle_.local_alloc_mem_ptr, alloc_strategy_.local_alloc_size, handle_);
    }
  }
  void create_local_cuda_runtime_memory()
  {
    bool on_device    = location_ == WHOLEMEMORY_ML_DEVICE;
    void* dev_ptr     = nullptr;
    size_t alloc_size = alloc_strategy_.local_alloc_size;
    if (alloc_size == 0) {
      no_ipc_handle_.local_alloc_mem_ptr = nullptr;
      return;
    }

    if (on_device) {
      WM_CUDA_CHECK(cudaMalloc(&dev_ptr, alloc_size));
    } else {
      WM_CUDA_CHECK(cudaMallocHost(&dev_ptr, alloc_size));
    }
    no_ipc_handle_.local_alloc_mem_ptr = dev_ptr;
    local_partition_memory_pointer_    = dev_ptr;
  }
  void destroy_local_cuda_runtime_memory() noexcept
  {
    try {
      void* ptr = no_ipc_handle_.local_alloc_mem_ptr;
      if (no_ipc_handle_.local_alloc_mem_ptr == nullptr) return;
      bool on_device = location_ == WHOLEMEMORY_ML_DEVICE;
      if (on_device) {
        WM_CUDA_CHECK(cudaFree(ptr));
      } else {
        WM_CUDA_CHECK(cudaFreeHost(ptr));
      }
      no_ipc_handle_.local_alloc_mem_ptr = nullptr;
    } catch (const wholememory::cuda_error& wce) {
      WHOLEMEMORY_FAIL_NOTHROW("%s", wce.what());
    } catch (const raft::exception& re) {
      WHOLEMEMORY_FAIL_NOTHROW("%s", re.what());
    }
  }

  struct no_ipc_handle {
    void* local_alloc_mem_ptr = nullptr;
  } no_ipc_handle_;
};

// Implementation for host wholememory that need global map.
// Rank 0 allocate all host memory and share between all ranks.
// for CONTINUOUS or CHUNKED type with HOST location
class global_mapped_host_wholememory_impl : public wholememory_impl {
 public:
  global_mapped_host_wholememory_impl(wholememory_handle_t wholememory_handle,
                                      size_t total_size,
                                      wholememory_comm_t comm,
                                      wholememory_memory_type_t memory_type,
                                      wholememory_memory_location_t memory_location,
                                      size_t data_granularity,
                                      size_t* rank_entry_partition)
    : wholememory_impl(wholememory_handle,
                       total_size,
                       comm,
                       memory_type,
                       memory_location,
                       data_granularity,
                       rank_entry_partition)
  {
    WHOLEMEMORY_CHECK(type_ == WHOLEMEMORY_MT_CONTINUOUS || type_ == WHOLEMEMORY_MT_CHUNKED);
    WHOLEMEMORY_CHECK(location_ == WHOLEMEMORY_ML_HOST);
  }
  void create_memory() override
  {
    generate_rank_partition_strategy();
    first_rank_allocate_all_strategy();
    create_and_map_shared_host_memory();
    register_host_memory();
  }
  void destroy_memory() noexcept override
  {
    unregister_host_memory();
    unmap_and_destroy_shared_host_memory();
  }
  [[nodiscard]] void* get_continuous_mapping_pointer() const noexcept override
  {
    return shared_host_handle_.shared_host_memory_ptr;
  }
  [[nodiscard]] wholememory_gref_t get_global_reference() const noexcept override
  {
    wholememory_gref_t gref{};
    gref.pointer    = get_continuous_mapping_pointer();
    gref.stride     = 0;
    gref.world_size = comm_->world_size;
    return gref;
  }
  bool contains_pointer(const void* ptr) const override
  {
    uint64_t int_ptr       = reinterpret_cast<uint64_t>(ptr);
    uint64_t int_start_ptr = reinterpret_cast<uint64_t>(shared_host_handle_.shared_host_memory_ptr);
    return int_ptr >= int_start_ptr && int_ptr < int_start_ptr + total_size_;
  }

  bool get_rank_memory(void** rank_memory_ptr,
                       size_t* rank_memory_size,
                       size_t* rank_memory_offset,
                       int rank) const noexcept override
  {
    size_t mem_size, mem_start;
    get_rank_partition_info(&mem_size, &mem_start, rank);
    if (rank_memory_ptr != nullptr)
      *rank_memory_ptr = (char*)get_continuous_mapping_pointer() + mem_start;
    if (rank_memory_size != nullptr) *rank_memory_size = mem_size;
    if (rank_memory_offset != nullptr) *rank_memory_offset = mem_start;
    return true;
  }

 protected:
  void register_host_memory()
  {
    std::unique_lock<std::mutex> vma_lock(wholememory_vma_mu);
    register_wholememory_vma_range_locked(
      shared_host_handle_.shared_host_memory_ptr, total_size_, handle_);
  }
  void unregister_host_memory() noexcept
  {
    std::unique_lock<std::mutex> vma_lock(wholememory_vma_mu);
    unregister_wholememory_vma_range_locked(
      shared_host_handle_.shared_host_memory_ptr, total_size_, handle_);
  }
  static std::string get_host_memory_full_path(wholememory_comm_t wm_comm, int tensor_id)
  {
    std::string host_memory_full_path = get_shm_prefix(wm_comm);
    host_memory_full_path.append("_").append("wm_host_").append(std::to_string(tensor_id));
    return host_memory_full_path;
  }

#define SYSTEMV_SHM_PROJ_ID (0xE601EEEE)
  void create_and_map_shared_host_memory()
  {
    WHOLEMEMORY_CHECK(is_intranode_communicator(comm_));
    const char* shm_env_var = std::getenv("WG_USE_POSIX_SHM");
    if (shm_env_var == nullptr || shm_env_var[0] == '0') {
      use_systemv_shm_ = true;
    } else {
      use_systemv_shm_ = false;
    }
    std::string shm_full_path;
    if (use_systemv_shm_) {
      shm_full_path = "/tmp/";
      shm_full_path.append(get_host_memory_full_path(comm_, handle_->handle_id));
      FILE* shm_fp = fopen(shm_full_path.c_str(), "w");
      WHOLEMEMORY_CHECK(shm_fp != nullptr);
      WHOLEMEMORY_CHECK(fclose(shm_fp) == 0);
    } else {
      shm_full_path = get_host_memory_full_path(comm_, handle_->handle_id);
    }
    int shm_id = -1;
    int shm_fd = -1;
    if (comm_->world_rank == 0) {
      if (use_systemv_shm_) {
        auto shm_key = ftok(shm_full_path.c_str(), SYSTEMV_SHM_PROJ_ID);
        WHOLEMEMORY_CHECK(shm_key != (key_t)-1);
        shm_id = shmget(shm_key, alloc_strategy_.local_alloc_size, 0644 | IPC_CREAT | IPC_EXCL);
        if (shm_id == -1) {
          WHOLEMEMORY_FATAL("Create host shared memory from IPC key %d failed, Reason=%s",
                            shm_key,
                            strerror(errno));
        }
      } else {
        shm_fd = shm_open(shm_full_path.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
        if (shm_fd < 0) {
          WHOLEMEMORY_FATAL("Create host shared memory from file %s failed, Reason=%s.",
                            shm_full_path.c_str(),
                            strerror(errno));
        }
        WHOLEMEMORY_CHECK(ftruncate(shm_fd, alloc_strategy_.local_alloc_size) == 0);
      }
      communicator_barrier(comm_);
    } else {
      communicator_barrier(comm_);
      if (use_systemv_shm_) {
        auto shm_key = ftok(shm_full_path.c_str(), SYSTEMV_SHM_PROJ_ID);
        WHOLEMEMORY_CHECK(shm_key != (key_t)-1);
        shm_id = shmget(shm_key, alloc_strategy_.local_alloc_size, 0644);
        if (shm_id == -1) {
          WHOLEMEMORY_FATAL(
            "Get host shared memory from IPC key %d failed, Reason=%s", shm_key, strerror(errno));
        }
      } else {
        shm_fd = shm_open(shm_full_path.c_str(), O_RDWR, S_IRUSR | S_IWUSR);
        if (shm_fd < 0) {
          WHOLEMEMORY_FATAL("Rank=%d open host shared memory from file %s failed.",
                            comm_->world_rank,
                            shm_full_path.c_str());
        }
      }
    }
    communicator_barrier(comm_);
    if (!use_systemv_shm_ && comm_->world_rank == 0) {
      WHOLEMEMORY_CHECK(shm_unlink(shm_full_path.c_str()) == 0);
    }
    void* mmap_ptr = nullptr;
    if (use_systemv_shm_) {
      mmap_ptr = shmat(shm_id, nullptr, 0);
      WHOLEMEMORY_CHECK(mmap_ptr != (void*)-1);
    } else {
      mmap_ptr = mmap(
        nullptr, alloc_strategy_.total_alloc_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
      WHOLEMEMORY_CHECK(mmap_ptr != (void*)-1);
    }
    memset(static_cast<char*>(mmap_ptr) + get_local_offset(), 0, get_local_size());
    WM_CUDA_CHECK_NO_THROW(
      cudaHostRegister(mmap_ptr, alloc_strategy_.total_alloc_size, cudaHostRegisterDefault));
    if (!use_systemv_shm_) WHOLEMEMORY_CHECK(close(shm_fd) == 0);
    void* dev_ptr = nullptr;
    WM_CUDA_CHECK_NO_THROW(cudaHostGetDevicePointer(&dev_ptr, mmap_ptr, 0));
    WHOLEMEMORY_CHECK(dev_ptr == mmap_ptr);
    shared_host_handle_.shared_host_memory_ptr = dev_ptr;
    local_partition_memory_pointer_            = static_cast<char*>(dev_ptr) + get_local_offset();
  }

  void unmap_and_destroy_shared_host_memory() noexcept
  {
    try {
      void* ptr = shared_host_handle_.shared_host_memory_ptr;
      if (ptr == nullptr) return;
      WM_CUDA_CHECK(cudaHostUnregister(ptr));
      std::string shm_full_path;
      int shm_id = -1;
      if (use_systemv_shm_) {
        shm_full_path = "/tmp/";
        shm_full_path.append(get_host_memory_full_path(comm_, handle_->handle_id));
        auto shm_key = ftok(shm_full_path.c_str(), SYSTEMV_SHM_PROJ_ID);
        WHOLEMEMORY_CHECK(shm_key != (key_t)-1);
        shm_id = shmget(shm_key, alloc_strategy_.local_alloc_size, 0644);
        if (shm_id == -1) {
          WHOLEMEMORY_FATAL("Get host shared memory from IPC key %d for delete failed, Reason=%s",
                            shm_key,
                            strerror(errno));
        }
        WHOLEMEMORY_CHECK(shmdt(ptr) == 0);
      } else {
        shm_full_path = get_host_memory_full_path(comm_, handle_->handle_id);
        WHOLEMEMORY_CHECK(munmap(ptr, alloc_strategy_.total_alloc_size) == 0);
      }
      communicator_barrier(comm_);
      if (use_systemv_shm_ && comm_->world_rank == 0) {
        WHOLEMEMORY_CHECK(shmctl(shm_id, IPC_RMID, nullptr) == 0);
        WHOLEMEMORY_CHECK(unlink(shm_full_path.c_str()) == 0);
      }

      communicator_barrier(comm_);
      shared_host_handle_.shared_host_memory_ptr = nullptr;
    } catch (const wholememory::logic_error& wle) {
      WHOLEMEMORY_FAIL_NOTHROW("%s", wle.what());
    } catch (const wholememory::cuda_error& wce) {
      WHOLEMEMORY_FAIL_NOTHROW("%s", wce.what());
    } catch (const raft::exception& re) {
      WHOLEMEMORY_FAIL_NOTHROW("%s", re.what());
    }
  }

  struct shared_host_handle {
    void* shared_host_memory_ptr = nullptr;
  } shared_host_handle_;

  bool use_systemv_shm_;
};

// Implementation for continuous device wholememory that need global map.
// Each rank allocate multiple pages and share pages with other ranks.
// for CONTINUOUS type with DEVICE location
class continuous_device_wholememory_impl : public wholememory_impl {
 public:
  continuous_device_wholememory_impl(wholememory_handle_t wholememory_handle,
                                     size_t total_size,
                                     wholememory_comm_t comm,
                                     wholememory_memory_type_t memory_type,
                                     wholememory_memory_location_t memory_location,
                                     size_t data_granularity,
                                     size_t* rank_entry_partition)
    : wholememory_impl(wholememory_handle,
                       total_size,
                       comm,
                       memory_type,
                       memory_location,
                       data_granularity,
                       rank_entry_partition)
  {
    // printf(
    //   "while in continuous device wholememory creation, the memory_type (%d) and memory_location
    //   "
    //   "(%d).\n",
    //   (int)memory_type,
    //   (int)memory_location);
    WHOLEMEMORY_CHECK(type_ == WHOLEMEMORY_MT_CONTINUOUS);
  }
  void create_memory() override
  {
    WHOLEMEMORY_CHECK(location_ == WHOLEMEMORY_ML_DEVICE);
    generate_rank_partition_strategy();
    each_rank_multiple_page_strategy();
    create_and_map_driver_device_memory();
    register_continuous_device_memory();
  }
  void destroy_memory() noexcept override
  {
    unregister_continuous_device_memory();
    unmap_and_destroy_driver_device_memory();
  }
  [[nodiscard]] void* get_continuous_mapping_pointer() const noexcept override
  {
    return cu_alloc_handle_.mapped_whole_memory;
  }
  [[nodiscard]] wholememory_gref_t get_global_reference() const noexcept override
  {
    wholememory_gref_t gref{};
    gref.pointer    = get_continuous_mapping_pointer();
    gref.stride     = 0;
    gref.world_size = comm_->world_size;
    return gref;
  }
  bool contains_pointer(const void* ptr) const override
  {
    uint64_t int_ptr       = reinterpret_cast<uint64_t>(ptr);
    uint64_t int_start_ptr = reinterpret_cast<uint64_t>(cu_alloc_handle_.mapped_whole_memory);
    return int_ptr >= int_start_ptr && int_ptr < int_start_ptr + total_size_;
  }
  bool get_rank_memory(void** rank_memory_ptr,
                       size_t* rank_memory_size,
                       size_t* rank_memory_offset,
                       int rank) const noexcept override
  {
    size_t mem_size, mem_start;
    get_rank_partition_info(&mem_size, &mem_start, rank);
    if (rank_memory_ptr != nullptr)
      *rank_memory_ptr = (char*)get_continuous_mapping_pointer() + mem_start;
    if (rank_memory_size != nullptr) *rank_memory_size = mem_size;
    if (rank_memory_offset != nullptr) *rank_memory_offset = mem_start;
    return true;
  }

 protected:
  void register_continuous_device_memory()
  {
    std::unique_lock<std::mutex> vma_lock(wholememory_vma_mu);
    register_wholememory_vma_range_locked(
      cu_alloc_handle_.mapped_whole_memory, total_size_, handle_);
  }
  void unregister_continuous_device_memory() noexcept
  {
    std::unique_lock<std::mutex> vma_lock(wholememory_vma_mu);
    unregister_wholememory_vma_range_locked(
      cu_alloc_handle_.mapped_whole_memory, total_size_, handle_);
  }

  struct ipc_sharable_cu_handle {
    int fd = -1;
  };

  static CUmemGenericAllocationHandle create_cu_mem(size_t mem_size, int dev_id)
  {
    CUmemGenericAllocationHandle h;
    CUmemAllocationProp prop;
    memset(&prop, 0, sizeof(prop));
    prop.type                       = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.requestedHandleTypes       = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    prop.location.type              = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;
    prop.location.id                = dev_id;
    WM_CU_CHECK(cuMemCreate(&h, mem_size, &prop, 0));
    return h;
  }

  static ipc_sharable_cu_handle create_sharable_handle(CUmemGenericAllocationHandle h)
  {
    ipc_sharable_cu_handle sharable_cu_handle;
    sharable_cu_handle.fd = -1;
    if (h != 0) {
      WM_CU_CHECK(cuMemExportToShareableHandle(
        &sharable_cu_handle.fd, h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
    }
    return sharable_cu_handle;
  }

  static int ipc_open_socket(const std::string& name)
  {
    int sock = -1;
    struct sockaddr_un skt_addr {
      0
    };
    if ((sock = socket(AF_UNIX, SOCK_DGRAM, 0)) < 0) {
      WHOLEMEMORY_FATAL("IPC failure: Socket creation error.");
    }
    bzero(&skt_addr, sizeof(skt_addr));
    skt_addr.sun_family = AF_UNIX;
    if (name.length() >= sizeof(skt_addr.sun_path)) {
      WHOLEMEMORY_FATAL(
        "IPC socket path length (%zu) larger than sockaddr_un.sun_path length (%lu), full_path: %s",
        name.length(),
        sizeof(skt_addr.sun_path),
        name.c_str());
    }
    strcpy(skt_addr.sun_path, name.c_str());
    if (bind(sock, reinterpret_cast<struct sockaddr*>(&skt_addr), sizeof(skt_addr)) < 0) {
      WHOLEMEMORY_FATAL("IPC failure: Binding socket failed, name=%s", name.c_str());
    }
    return sock;
  }

  static void ipc_close_socket(int fd, const std::string& name)
  {
    WHOLEMEMORY_CHECK(fd >= 0);
    WHOLEMEMORY_CHECK(!name.empty());
    WHOLEMEMORY_CHECK(unlink(name.c_str()) == 0);
    WHOLEMEMORY_CHECK(close(fd) == 0);
  };

  [[nodiscard]] std::string get_sender_fd_name() const
  {
    std::string name = get_temporary_directory_path(comm_);
    name.append("/sender_").append(std::to_string(comm_->world_rank)).append(".sock");
    return name;
  }
  [[nodiscard]] std::string get_recver_fd_name(int src_id) const
  {
    std::string name = get_temporary_directory_path(comm_);
    name.append("/recver_")
      .append(std::to_string(src_id))
      .append("_to_")
      .append(std::to_string(comm_->world_rank))
      .append(".sock");
    return name;
  }
  [[nodiscard]] std::string get_target_recver_fd_name(int dst_id) const
  {
    std::string name = get_temporary_directory_path(comm_);
    name.append("/recver_")
      .append(std::to_string(comm_->world_rank))
      .append("_to_")
      .append(std::to_string(dst_id))
      .append(".sock");
    return name;
  }

  void open_unix_domain_sockets()
  {
    communicator_barrier(comm_);
    cu_alloc_handle_.recv_fds.clear();
    cu_alloc_handle_.recv_fds.resize(comm_->world_size, -1);
    for (int i = 0; i < comm_->world_size; i++) {
      cu_alloc_handle_.recv_fds[i] = ipc_open_socket(get_recver_fd_name(i));
    }
    cu_alloc_handle_.send_fd = ipc_open_socket(get_sender_fd_name());
    communicator_barrier(comm_);
  }
  void close_unix_domain_sockets()
  {
    communicator_barrier(comm_);
    ipc_close_socket(cu_alloc_handle_.send_fd, get_sender_fd_name());
    cu_alloc_handle_.send_fd = -1;
    WHOLEMEMORY_CHECK(cu_alloc_handle_.recv_fds.size() == comm_->world_size);
    for (int i = 0; i < comm_->world_size; i++) {
      ipc_close_socket(cu_alloc_handle_.recv_fds[i], get_recver_fd_name(i));
    }
    communicator_barrier(comm_);
  }

  static void ipc_send_sharable_handle(int sock_fd,
                                       const ipc_sharable_cu_handle& sent_handle,
                                       const std::string& dst_name)
  {
    struct msghdr message_header {};
    struct iovec iov[1];

    union {
      struct cmsghdr cm;
      char control[CMSG_SPACE(sizeof(int))];
    } control_un{};

    struct cmsghdr* cmptr;
    struct sockaddr_un cliaddr {};

    // Construct client address to send this Shareable handle to
    bzero(&cliaddr, sizeof(cliaddr));
    cliaddr.sun_family = AF_UNIX;
    strcpy(cliaddr.sun_path, dst_name.c_str());

    // Send corresponding shareable handle to the client
    int sendfd = sent_handle.fd;

    message_header.msg_control    = control_un.control;
    message_header.msg_controllen = sizeof(control_un.control);

    cmptr             = CMSG_FIRSTHDR(&message_header);
    cmptr->cmsg_len   = CMSG_LEN(sizeof(int));
    cmptr->cmsg_level = SOL_SOCKET;
    cmptr->cmsg_type  = SCM_RIGHTS;

    memmove(CMSG_DATA(cmptr), &sendfd, sizeof(sendfd));

    message_header.msg_name    = static_cast<void*>(&cliaddr);
    message_header.msg_namelen = sizeof(struct sockaddr_un);

    iov[0].iov_base           = const_cast<void*>(static_cast<const void*>(""));
    iov[0].iov_len            = 1;
    message_header.msg_iov    = iov;
    message_header.msg_iovlen = 1;

    ssize_t send_result = sendmsg(sock_fd, &message_header, 0);
    if (send_result <= 0) {
      WHOLEMEMORY_FATAL("IPC failure: Sending data over socket failed send_result=%ld.",
                        send_result);
    }
  }
  static ipc_sharable_cu_handle ipc_recv_sharable_handle(int recv_sock_fd)
  {
    struct msghdr message_header = {nullptr};
    struct iovec iov[1];

    // Union to guarantee alignment requirements for control array
    union {
      struct cmsghdr cm;
      char control[CMSG_SPACE(sizeof(int))];
    } control_un{};

    struct cmsghdr* cmptr;
    ssize_t n;
    int received_fd = -1;
    char dummy_buffer[1];

    message_header.msg_control    = control_un.control;
    message_header.msg_controllen = sizeof(control_un.control);

    iov[0].iov_base = static_cast<void*>(&dummy_buffer[0]);
    iov[0].iov_len  = sizeof(dummy_buffer);

    message_header.msg_iov    = iov;
    message_header.msg_iovlen = 1;

    if ((n = recvmsg(recv_sock_fd, &message_header, 0)) <= 0) {
      WHOLEMEMORY_FATAL("IPC failure: Receiving data over socket failed, recvmsg returned %ld", n);
    }

    if (((cmptr = CMSG_FIRSTHDR(&message_header)) != nullptr) &&
        (cmptr->cmsg_len == CMSG_LEN(sizeof(int)))) {
      if ((cmptr->cmsg_level != SOL_SOCKET) || (cmptr->cmsg_type != SCM_RIGHTS)) {
        WHOLEMEMORY_FATAL("Non socket received");
      }

      memmove(&received_fd, CMSG_DATA(cmptr), sizeof(received_fd));
    } else {
      WHOLEMEMORY_FATAL("Recv cm_ptr=%p, cmsg_len=%ld", cmptr, (cmptr ? cmptr->cmsg_len : -1));
    }
    ipc_sharable_cu_handle sharable_cu_handle;
    sharable_cu_handle.fd = received_fd;
    return sharable_cu_handle;
  }

  void exchange_driver_device_memory_handles(
    std::vector<ipc_sharable_cu_handle>* recv_ipc_sharable_cu_handles,
    std::vector<ipc_sharable_cu_handle>* send_ipc_sharable_cu_handles)
  {
    for (int r = 0; r < comm_->world_size; r++) {
      if ((*send_ipc_sharable_cu_handles)[r].fd >= 0) {
        ipc_send_sharable_handle(cu_alloc_handle_.send_fd,
                                 (*send_ipc_sharable_cu_handles)[r],
                                 get_target_recver_fd_name(r));
      }
    }
    communicator_barrier(comm_);
    recv_ipc_sharable_cu_handles->resize(comm_->world_size);
    for (int r = 0; r < comm_->world_size; r++) {
      if (alloc_strategy_.alloc_sizes[r] > 0) {
        (*recv_ipc_sharable_cu_handles)[r] = ipc_recv_sharable_handle(cu_alloc_handle_.recv_fds[r]);
      }
    }
    communicator_barrier(comm_);
    if (cu_alloc_handle_.local_ipc_handle.fd >= 0) {
      WHOLEMEMORY_CHECK(close(cu_alloc_handle_.local_ipc_handle.fd) == 0);
    }
    send_ipc_sharable_cu_handles->clear();
  }
  static CUmemGenericAllocationHandle import_cu_mem_handle(
    ipc_sharable_cu_handle sharable_cu_handle)
  {
    CUmemGenericAllocationHandle h;
    WM_CU_CHECK(cuMemImportFromShareableHandle(
      &h, (void*)(uintptr_t)sharable_cu_handle.fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    return h;
  }
  void map_driver_device_memory_handles(
    std::vector<ipc_sharable_cu_handle>* recv_ipc_sharable_cu_handles)
  {
    cu_alloc_handle_.all_cu_handles.resize(comm_->world_size);
    for (int i = 0; i < comm_->world_size; i++) {
      size_t mem_size = alloc_strategy_.alloc_sizes[i];
      if (mem_size > 0) {
        WHOLEMEMORY_CHECK((*recv_ipc_sharable_cu_handles)[i].fd >= 0);
        cu_alloc_handle_.all_cu_handles[i] =
          import_cu_mem_handle((*recv_ipc_sharable_cu_handles)[i]);

        WM_CU_CHECK(cuMemMap(reinterpret_cast<CUdeviceptr>(cu_alloc_handle_.mapped_whole_memory) +
                               alloc_strategy_.alloc_offsets[i],
                             mem_size,
                             0,
                             cu_alloc_handle_.all_cu_handles[i],
                             0));
        WHOLEMEMORY_CHECK(close((*recv_ipc_sharable_cu_handles)[i].fd) == 0);
      } else {
        WHOLEMEMORY_CHECK((*recv_ipc_sharable_cu_handles)[i].fd == -1);
      }
    }
    recv_ipc_sharable_cu_handles->clear();
    CUmemAccessDesc madesc;
    madesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    madesc.location.id   = comm_->dev_id;
    madesc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    WM_CU_CHECK(cuMemSetAccess(reinterpret_cast<CUdeviceptr>(cu_alloc_handle_.mapped_whole_memory),
                               alloc_strategy_.total_alloc_size,
                               &madesc,
                               1));
  }
  void create_and_map_driver_device_memory()
  {
    WM_CU_CHECK(
      cuMemAddressReserve(reinterpret_cast<CUdeviceptr*>(&cu_alloc_handle_.mapped_whole_memory),
                          alloc_strategy_.total_alloc_size,
                          alloc_strategy_.alignment,
                          0,
                          0));
    cu_alloc_handle_.all_cu_handles.resize(comm_->world_size, 0);
    std::vector<ipc_sharable_cu_handle> send_ipc_sharable_cu_handles(comm_->world_size);
    std::vector<ipc_sharable_cu_handle> recv_ipc_sharable_cu_handles;
    cu_alloc_handle_.local_cu_handle = 0;
    if (alloc_strategy_.local_alloc_size > 0) {
      cu_alloc_handle_.local_cu_handle =
        create_cu_mem(alloc_strategy_.local_alloc_size, comm_->dev_id);
    }
    cu_alloc_handle_.local_ipc_handle = create_sharable_handle(cu_alloc_handle_.local_cu_handle);
    for (int i = 0; i < comm_->world_size; i++) {
      send_ipc_sharable_cu_handles[i] = cu_alloc_handle_.local_ipc_handle;
    }
    open_unix_domain_sockets();
    exchange_driver_device_memory_handles(&recv_ipc_sharable_cu_handles,
                                          &send_ipc_sharable_cu_handles);
    close_unix_domain_sockets();
    map_driver_device_memory_handles(&recv_ipc_sharable_cu_handles);
    communicator_barrier(comm_);
    local_partition_memory_pointer_ =
      static_cast<char*>(cu_alloc_handle_.mapped_whole_memory) + get_local_offset();
  }
  void unmap_and_destroy_driver_device_memory() noexcept
  {
    try {
      communicator_barrier(comm_);
      for (int i = 0; i < comm_->world_size; i++) {
        size_t mem_size = alloc_strategy_.alloc_sizes[i];
        if (mem_size > 0) {
          WM_CU_CHECK(
            cuMemUnmap(reinterpret_cast<CUdeviceptr>(cu_alloc_handle_.mapped_whole_memory) +
                         alloc_strategy_.alloc_offsets[i],
                       mem_size));
          WM_CU_CHECK(cuMemRelease(cu_alloc_handle_.all_cu_handles[i]));
        }
      }
      communicator_barrier(comm_);
      if (alloc_strategy_.local_alloc_size > 0) {
        WM_CU_CHECK(cuMemRelease(cu_alloc_handle_.local_cu_handle));
      }
      WM_CU_CHECK(
        cuMemAddressFree(reinterpret_cast<CUdeviceptr>(cu_alloc_handle_.mapped_whole_memory),
                         alloc_strategy_.total_alloc_size));

      communicator_barrier(comm_);
    } catch (const wholememory::cu_error& wce) {
      WHOLEMEMORY_FAIL_NOTHROW("%s", wce.what());
    } catch (const raft::exception& re) {
      WHOLEMEMORY_FAIL_NOTHROW("%s", re.what());
    }
  }

  struct cu_alloc_handle {
    CUmemGenericAllocationHandle local_cu_handle = 0;
    std::vector<CUmemGenericAllocationHandle> all_cu_handles;
    void* mapped_whole_memory = nullptr;
    ipc_sharable_cu_handle local_ipc_handle;
#if CUDA_VERSION >= 12030
    CUmemFabricHandle local_ipc_fabric_handle;
#endif

    int send_fd = -1;
    std::vector<int> recv_fds;
  } cu_alloc_handle_;
};

// Implementation for chunked device wholememory that need global map.
// Each rank allocate same size memory chunk and share with other ranks.
// for CHUNKED type with DEVICE location
class chunked_device_wholememory_impl : public wholememory_impl {
 public:
  chunked_device_wholememory_impl(wholememory_handle_t wholememory_handle,
                                  size_t total_size,
                                  wholememory_comm_t comm,
                                  wholememory_memory_type_t memory_type,
                                  wholememory_memory_location_t memory_location,
                                  size_t data_granularity,
                                  size_t* rank_entry_partition)
    : wholememory_impl(wholememory_handle,
                       total_size,
                       comm,
                       memory_type,
                       memory_location,
                       data_granularity,
                       rank_entry_partition)
  {
    WHOLEMEMORY_CHECK(type_ == WHOLEMEMORY_MT_CHUNKED);
    WHOLEMEMORY_CHECK(location_ == WHOLEMEMORY_ML_DEVICE);
  }
  void create_memory() override
  {
    generate_rank_partition_strategy();
    each_rank_different_chunk_strategy();
    create_and_map_runtime_device_memory();
    register_chunked_device_memory();
  }
  void destroy_memory() noexcept override
  {
    unregister_chunked_device_memory();
    unmap_and_destroy_runtime_device_memory();
  }
  [[nodiscard]] wholememory_gref_t get_global_reference() const noexcept override { return gref_; }
  bool contains_pointer(const void* ptr) const override
  {
    uint64_t int_ptr = reinterpret_cast<uint64_t>(ptr);
    size_t acc_size  = 0;
    for (int i = 0; i < comm_->world_size; i++) {
      size_t mem_size_of_this_rank_and_after = total_size_ - acc_size;
      size_t mem_size_for_current_rank =
        std::min(mem_size_of_this_rank_and_after, rank_partition_strategy_.partition_sizes_[i]);
      uint64_t int_start_ptr = reinterpret_cast<uint64_t>(cuda_ipc_handle_.mapped_ptrs[i]);
      if (int_ptr >= int_start_ptr && int_ptr < int_start_ptr + mem_size_for_current_rank) {
        return true;
      }
      acc_size += mem_size_for_current_rank;
    }
    return false;
  }
  bool get_rank_memory(void** rank_memory_ptr,
                       size_t* rank_memory_size,
                       size_t* rank_memory_offset,
                       int rank) const noexcept override
  {
    size_t mem_size, mem_start;
    get_rank_partition_info(&mem_size, &mem_start, rank);
    if (rank_memory_ptr != nullptr) *rank_memory_ptr = cuda_ipc_handle_.mapped_ptrs[rank];
    if (rank_memory_size != nullptr) *rank_memory_size = mem_size;
    if (rank_memory_offset != nullptr) *rank_memory_offset = mem_start;
    return true;
  }

 protected:
  void register_chunked_device_memory()
  {
    std::unique_lock<std::mutex> vma_lock(wholememory_vma_mu);
    size_t acc_size = 0;
    for (int i = 0; i < comm_->world_size; i++) {
      size_t mem_size_of_this_rank_and_after = total_size_ - acc_size;
      size_t mem_size_for_current_rank =
        std::min(mem_size_of_this_rank_and_after, rank_partition_strategy_.partition_sizes_[i]);
      if (mem_size_for_current_rank > 0) {
        register_wholememory_vma_range_locked(
          cuda_ipc_handle_.mapped_ptrs[i], mem_size_for_current_rank, handle_);
      }
      acc_size += mem_size_for_current_rank;
    }
  }
  void unregister_chunked_device_memory() noexcept
  {
    std::unique_lock<std::mutex> vma_lock(wholememory_vma_mu);
    size_t acc_size = 0;
    for (int i = 0; i < comm_->world_size; i++) {
      size_t mem_size_of_this_rank_and_after = total_size_ - acc_size;
      size_t mem_size_for_current_rank =
        std::min(mem_size_of_this_rank_and_after, rank_partition_strategy_.partition_sizes_[i]);
      if (mem_size_for_current_rank > 0) {
        unregister_wholememory_vma_range_locked(
          cuda_ipc_handle_.mapped_ptrs[i], mem_size_for_current_rank, handle_);
      }
      acc_size += mem_size_for_current_rank;
    }
  }
  void create_and_map_runtime_device_memory()
  {
    cuda_ipc_handle_.mapped_ptrs.resize(comm_->world_size, nullptr);
    cuda_ipc_handle_.all_mem_handles.resize(comm_->world_size);
    WM_CUDA_CHECK(
      cudaMalloc((void**)&cuda_ipc_handle_.local_mem_ptr, alloc_strategy_.local_alloc_size));
    WM_CUDA_CHECK(
      cudaIpcGetMemHandle(&cuda_ipc_handle_.local_ipc_handle, cuda_ipc_handle_.local_mem_ptr));
    comm_->host_allgather(&cuda_ipc_handle_.local_ipc_handle,
                          cuda_ipc_handle_.all_mem_handles.data(),
                          sizeof(cuda_ipc_handle_.local_ipc_handle),
                          WHOLEMEMORY_DT_INT8);
    for (int i = 0; i < comm_->world_size; i++) {
      if (i == comm_->world_rank) {
        cuda_ipc_handle_.mapped_ptrs[i] = cuda_ipc_handle_.local_mem_ptr;
      } else {
        WM_CUDA_CHECK(cudaIpcOpenMemHandle(&cuda_ipc_handle_.mapped_ptrs[i],
                                           cuda_ipc_handle_.all_mem_handles[i],
                                           cudaIpcMemLazyEnablePeerAccess));
      }
    }
    local_partition_memory_pointer_ = cuda_ipc_handle_.local_mem_ptr;
    WM_CUDA_CHECK(cudaMalloc(&gref_.pointer, sizeof(void*) * comm_->world_size));
    WM_CUDA_CHECK(cudaMemcpy(gref_.pointer,
                             cuda_ipc_handle_.mapped_ptrs.data(),
                             sizeof(void*) * comm_->world_size,
                             cudaMemcpyHostToDevice));
    WM_CUDA_CHECK(cudaMalloc(&gref_.rank_memory_offsets, sizeof(size_t) * (comm_->world_size + 1)));
    WM_CUDA_CHECK(cudaMemcpy(gref_.rank_memory_offsets,
                             get_rank_offsets().data(),
                             sizeof(size_t) * (comm_->world_size + 1),
                             cudaMemcpyHostToDevice));
    gref_.world_size = comm_->world_size;
    gref_.stride     = rank_partition_strategy_.partition_mem_stride;
    gref_.same_chunk = rank_partition_strategy_.same_chunk;
  }
  void unmap_and_destroy_runtime_device_memory() noexcept
  {
    try {
      WM_CUDA_CHECK(cudaFree(gref_.pointer));
      WM_CUDA_CHECK(cudaFree(gref_.rank_memory_offsets));
      gref_.pointer = nullptr;
      for (int i = 0; i < comm_->world_size; i++) {
        if (i != comm_->world_rank) {
          WM_CUDA_CHECK(cudaIpcCloseMemHandle(cuda_ipc_handle_.mapped_ptrs[i]));
        }
      }
      // Check all memory unmapped.
      communicator_barrier(comm_);
      WM_CUDA_CHECK(cudaFree(cuda_ipc_handle_.local_mem_ptr));
    } catch (const wholememory::cuda_error& wce) {
      WHOLEMEMORY_FAIL_NOTHROW("%s", wce.what());
    } catch (const raft::exception& re) {
      WHOLEMEMORY_FAIL_NOTHROW("%s", re.what());
    }
  }

  struct cuda_ipc_handle {
    cudaIpcMemHandle_t local_ipc_handle;
    std::vector<cudaIpcMemHandle_t> all_mem_handles;
    std::vector<void*> mapped_ptrs;
    void* local_mem_ptr;
  } cuda_ipc_handle_;

  wholememory_gref_t gref_;
};

#ifdef WITH_NVSHMEM_SUPPORT
class nvshmem_device_wholememory_impl : public wholememory_impl {
 public:
  nvshmem_device_wholememory_impl(wholememory_handle_t wholememory_handle,
                                  size_t total_size,
                                  wholememory_comm_t comm,
                                  wholememory_memory_type_t memory_type,
                                  wholememory_memory_location_t memory_location,
                                  size_t data_granularity,
                                  size_t* rank_entry_partition)
    : wholememory_impl(wholememory_handle,
                       total_size,
                       comm,
                       memory_type,
                       memory_location,
                       data_granularity,
                       rank_entry_partition)
  {
    WHOLEMEMORY_CHECK(type_ == WHOLEMEMORY_MT_DISTRIBUTED);
    WHOLEMEMORY_CHECK(location_ == WHOLEMEMORY_ML_DEVICE);
    WHOLEMEMORY_CHECK(comm->distributed_backend == WHOLEMEMORY_DB_NVSHMEM);
    distrubuted_backend_ = WHOLEMEMORY_DB_NVSHMEM;
    check_or_set_nvshmem_heap_kind();
    init_nvshmem_with_comm(comm);
    WHOLEMEMORY_CHECK(comm->bind_to_nvshmem);
  }

  void create_memory() override
  {
    generate_rank_partition_strategy();
    each_rank_different_chunk_strategy();
    nvshmem_malloc_device_memory();
    register_nvshmem_device_memory();
  }

  void destroy_memory() noexcept override
  {
    unregister_nvshmem_device_memory();
    nvshmem_free_device_memory();
  }

  bool contains_pointer(const void* ptr) const override
  {
    uint64_t int_ptr = reinterpret_cast<uint64_t>(ptr);
    size_t acc_size  = 0;
    for (int i = 0; i < comm_->world_size; i++) {
      size_t mem_size_of_this_rank_and_after = total_size_ - acc_size;
      size_t mem_size_for_current_rank =
        std::min(mem_size_of_this_rank_and_after, rank_partition_strategy_.partition_sizes_[i]);
      acc_size += mem_size_for_current_rank;
      uint64_t int_start_ptr =
        reinterpret_cast<uint64_t>(nvshmem_ptr(nvshmem_memory_handle_.local_alloc_mem_ptr, i));
      if (int_start_ptr == 0) continue;
      if (int_ptr >= int_start_ptr && int_ptr < int_start_ptr + mem_size_for_current_rank) {
        return true;
      }
    }
    return false;
  }

  bool get_rank_memory(void** rank_memory_ptr,
                       size_t* rank_memory_size,
                       size_t* rank_memory_offset,
                       int rank) const noexcept override
  {
    size_t mem_size, mem_start;
    get_rank_partition_info(&mem_size, &mem_start, rank);
    void* peer_ptr = nvshmem_ptr(nvshmem_memory_handle_.local_alloc_mem_ptr, rank);
    if (peer_ptr == nullptr) return false;

    if (rank_memory_ptr != nullptr) *rank_memory_ptr = peer_ptr;
    if (rank_memory_size != nullptr) *rank_memory_size = mem_size;
    if (rank_memory_offset != nullptr) *rank_memory_offset = mem_start;
    return true;
  }

  [[nodiscard]] wholememory_gref_t get_global_reference() const noexcept override { return gref_; }

 protected:
  void register_nvshmem_device_memory()
  {
    std::unique_lock<std::mutex> vma_lock(wholememory_vma_mu);
    size_t acc_size = 0;
    for (int i = 0; i < comm_->world_size; i++) {
      size_t mem_size_of_this_rank_and_after = total_size_ - acc_size;
      size_t mem_size_for_current_rank =
        std::min(mem_size_of_this_rank_and_after, rank_partition_strategy_.partition_sizes_[i]);
      if (mem_size_for_current_rank > 0) {
        void* ptr = nvshmem_ptr(nvshmem_memory_handle_.local_alloc_mem_ptr, i);
        if (ptr != nullptr) {
          register_wholememory_vma_range_locked(ptr, mem_size_for_current_rank, handle_);
        }
      }
      acc_size += mem_size_for_current_rank;
    }
  }
  void unregister_nvshmem_device_memory()
  {
    std::unique_lock<std::mutex> vma_lock(wholememory_vma_mu);
    size_t acc_size = 0;
    for (int i = 0; i < comm_->world_size; i++) {
      size_t mem_size_of_this_rank_and_after = total_size_ - acc_size;
      size_t mem_size_for_current_rank =
        std::min(mem_size_of_this_rank_and_after, rank_partition_strategy_.partition_sizes_[i]);
      if (mem_size_for_current_rank > 0) {
        void* ptr = nvshmem_ptr(nvshmem_memory_handle_.local_alloc_mem_ptr, i);
        if (ptr != nullptr) {
          unregister_wholememory_vma_range_locked(ptr, mem_size_for_current_rank, handle_);
        }
      }
      acc_size += mem_size_for_current_rank;
    }
  }
  void nvshmem_malloc_device_memory()
  {
    WHOLEMEMORY_EXPECTS(
      comm_->bind_to_nvshmem == true,
      "nvshmem_malloc_device_memory  should be called with the comm which used to init nvshmem.");
    size_t alloc_size                          = alloc_strategy_.local_alloc_size;
    nvshmem_memory_handle_.local_alloc_mem_ptr = nvshmem_malloc(alloc_size);
    local_partition_memory_pointer_            = nvshmem_memory_handle_.local_alloc_mem_ptr;
    distrubuted_backend_                       = WHOLEMEMORY_DB_NVSHMEM;

    WM_CUDA_CHECK(cudaMalloc(&gref_.rank_memory_offsets, sizeof(size_t) * (comm_->world_size + 1)));
    WM_CUDA_CHECK(cudaMemcpy(gref_.rank_memory_offsets,
                             get_rank_offsets().data(),
                             sizeof(size_t) * (comm_->world_size + 1),
                             cudaMemcpyHostToDevice));
    gref_.pointer    = local_partition_memory_pointer_;
    gref_.world_size = comm_->world_size;
    gref_.stride     = rank_partition_strategy_.partition_mem_stride;
    gref_.same_chunk = rank_partition_strategy_.same_chunk;
  }

  void nvshmem_free_device_memory()
  {
    WM_CUDA_CHECK(cudaFree(gref_.rank_memory_offsets));
    gref_.pointer = nullptr;
    if (nvshmem_memory_handle_.local_alloc_mem_ptr) {
      nvshmem_free(nvshmem_memory_handle_.local_alloc_mem_ptr);

      nvshmem_memory_handle_.local_alloc_mem_ptr = nullptr;
    }
  }

  void check_or_set_nvshmem_heap_kind()
  {
    if (!has_set_nvshmem_heap) {
      if (location_ == WHOLEMEMORY_ML_HOST) {
        setenv("NVSHMEM_HEAP_KIND", "SYSMEM", 1);
      } else if (location_ == WHOLEMEMORY_ML_DEVICE) {
        setenv("NVSHMEM_HEAP_KIND", "DEVICE", 1);
      }
      has_set_nvshmem_heap = true;
      return;
    }
    const char* sys_heap_location = std::getenv("NVSHMEM_HEAP_KIND");

    if (location_ == WHOLEMEMORY_ML_HOST) {
      WHOLEMEMORY_CHECK((sys_heap_location != nullptr) &&
                        (strcmp(sys_heap_location, "SYSMEM") == 0));
    } else if (location_ == WHOLEMEMORY_ML_DEVICE) {
      WHOLEMEMORY_CHECK((sys_heap_location != nullptr) &&
                        (strcmp(sys_heap_location, "DEVICE") == 0));
    }
  }

  struct nvshmem_memory_handle {
    void* local_alloc_mem_ptr = nullptr;
  } nvshmem_memory_handle_;
  inline static bool has_set_nvshmem_heap = false;

  wholememory_gref_t gref_;
};
#endif
// Implementation for MNNVL wholememory that use cuda driver api.
// Each rank allocate multiple pages and share pages with other ranks.
// for CONTINUOUS type with HOST or DEVICE location
#if CUDA_VERSION >= 12030
class continuous_mnnvl_wholememory_impl : public continuous_device_wholememory_impl {
 public:
  continuous_mnnvl_wholememory_impl(wholememory_handle_t wholememory_handle,
                                    size_t total_size,
                                    wholememory_comm_t comm,
                                    wholememory_memory_type_t memory_type,
                                    wholememory_memory_location_t memory_location,
                                    size_t data_granularity,
                                    size_t* rank_entry_partition)
    : continuous_device_wholememory_impl(wholememory_handle,
                                         total_size,
                                         comm,
                                         memory_type,
                                         memory_location,
                                         data_granularity,
                                         rank_entry_partition)
  {
    WHOLEMEMORY_INFO("Using continuous_mnnvl_wholememory_impl");
    WHOLEMEMORY_CHECK_NOTHROW(type_ == WHOLEMEMORY_MT_CONTINUOUS);
  }
  void check_valid()
  {
    if (location_ == WHOLEMEMORY_ML_HOST) { WHOLEMEMORY_CHECK_NOTHROW(SupportEGM()); }
    WHOLEMEMORY_CHECK_NOTHROW(comm_->is_intra_mnnvl());
  }
  void create_memory() override
  {
    check_valid();
    generate_rank_partition_strategy();
    each_rank_multiple_page_strategy();
    create_and_map_driver_memory();
    register_continuous_mnnvl_memory();
  }
  void destroy_memory() noexcept override
  {
    unregister_continuous_mnnvl_memory();
    unmap_and_destroy_driver_host_memory();
  }

 protected:
  void register_continuous_mnnvl_memory()
  {
    std::unique_lock<std::mutex> vma_lock(wholememory_vma_mu);
    register_wholememory_vma_range_locked(
      cu_alloc_handle_.mapped_whole_memory, total_size_, handle_);
  }
  void unregister_continuous_mnnvl_memory() noexcept
  {
    std::unique_lock<std::mutex> vma_lock(wholememory_vma_mu);
    unregister_wholememory_vma_range_locked(
      cu_alloc_handle_.mapped_whole_memory, total_size_, handle_);
  }

  static CUmemGenericAllocationHandle create_cu_mem(size_t mem_size,
                                                    int dev_id,
                                                    wholememory_memory_location_t location)
  {
    CUmemGenericAllocationHandle h;
    CUmemAllocationProp prop;
    memset(&prop, 0, sizeof(prop));
    if (location == WHOLEMEMORY_ML_HOST) {
      int numa_id;
      cuDeviceGetAttribute(&numa_id, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, dev_id);

      prop.type                       = CU_MEM_ALLOCATION_TYPE_PINNED;
      prop.requestedHandleTypes       = CU_MEM_HANDLE_TYPE_FABRIC;
      prop.location.type              = CU_MEM_LOCATION_TYPE_HOST_NUMA;
      prop.location.id                = numa_id;
      prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;
    } else {
      WHOLEMEMORY_CHECK_NOTHROW(location == WHOLEMEMORY_ML_DEVICE);
      prop.type                       = CU_MEM_ALLOCATION_TYPE_PINNED;
      prop.requestedHandleTypes       = CU_MEM_HANDLE_TYPE_FABRIC;
      prop.location.type              = CU_MEM_LOCATION_TYPE_DEVICE;
      prop.location.id                = dev_id;
      prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;
    }
    WM_CU_CHECK_NO_THROW(cuMemCreate(&h, mem_size, &prop, 0));
    return h;
  }

  static CUmemFabricHandle create_sharable_fabric_handle(CUmemGenericAllocationHandle h)
  {
    CUmemFabricHandle fabric_handle;
    if (h != 0) {
      WM_CU_CHECK_NO_THROW(
        cuMemExportToShareableHandle(&fabric_handle, h, CU_MEM_HANDLE_TYPE_FABRIC, 0));
    }
    return fabric_handle;
  }

  void exchange_driver_host_memory_handles(
    std::vector<CUmemFabricHandle>* recv_ipc_sharable_cu_fabric_handles,
    CUmemFabricHandle* send_ipc_sharable_cu_fabric_handle)
  {
    communicator_barrier(comm_);
    recv_ipc_sharable_cu_fabric_handles->resize(comm_->world_size);
    comm_->host_allgather(static_cast<const void*>(send_ipc_sharable_cu_fabric_handle),
                          static_cast<void*>(recv_ipc_sharable_cu_fabric_handles->data()),
                          sizeof(CUmemFabricHandle),
                          WHOLEMEMORY_DT_INT8);
    communicator_barrier(comm_);
  }
  CUmemGenericAllocationHandle import_cu_mem_handle(CUmemFabricHandle sharable_cu_fabric_handle,
                                                    bool same_rank)
  {
    CUmemGenericAllocationHandle h;
    if (!same_rank) {
      WM_CU_CHECK_NO_THROW(
        cuMemImportFromShareableHandle(&h, &sharable_cu_fabric_handle, CU_MEM_HANDLE_TYPE_FABRIC));
    } else {
      h = cu_alloc_handle_.local_cu_handle;
    }
    return h;
  }
  void map_driver_memory_handles(
    std::vector<CUmemFabricHandle>* recv_ipc_sharable_cu_fabric_handles)
  {
    cu_alloc_handle_.all_cu_handles.resize(comm_->world_size);
    for (int i = 0; i < comm_->world_size; i++) {
      size_t mem_size = alloc_strategy_.alloc_sizes[i];
      if (mem_size > 0) {
        cu_alloc_handle_.all_cu_handles[i] =
          import_cu_mem_handle((*recv_ipc_sharable_cu_fabric_handles)[i], i == comm_->world_rank);

        WM_CU_CHECK_NO_THROW(
          cuMemMap(reinterpret_cast<CUdeviceptr>(cu_alloc_handle_.mapped_whole_memory) +
                     alloc_strategy_.alloc_offsets[i],
                   mem_size,
                   0,
                   cu_alloc_handle_.all_cu_handles[i],
                   0));
      }
    }
    recv_ipc_sharable_cu_fabric_handles->clear();
    CUmemAccessDesc madesc;
    madesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    madesc.location.id   = comm_->dev_id;
    madesc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    WM_CU_CHECK_NO_THROW(
      cuMemSetAccess(reinterpret_cast<CUdeviceptr>(cu_alloc_handle_.mapped_whole_memory),
                     alloc_strategy_.total_alloc_size,
                     &madesc,
                     1));
  }
  void create_and_map_driver_memory()
  {
    WM_CU_CHECK(
      cuMemAddressReserve(reinterpret_cast<CUdeviceptr*>(&cu_alloc_handle_.mapped_whole_memory),
                          alloc_strategy_.total_alloc_size,
                          alloc_strategy_.alignment,
                          0,
                          0));
    cu_alloc_handle_.all_cu_handles.resize(comm_->world_size, 0);
    std::vector<CUmemFabricHandle> recv_ipc_sharable_cu_fabric_handles;
    cu_alloc_handle_.local_cu_handle = 0;
    if (alloc_strategy_.local_alloc_size > 0) {
      cu_alloc_handle_.local_cu_handle =
        create_cu_mem(alloc_strategy_.local_alloc_size, comm_->dev_id, location_);
    }
    cu_alloc_handle_.local_ipc_fabric_handle =
      create_sharable_fabric_handle(cu_alloc_handle_.local_cu_handle);

    exchange_driver_host_memory_handles(&recv_ipc_sharable_cu_fabric_handles,
                                        &cu_alloc_handle_.local_ipc_fabric_handle);

    map_driver_memory_handles(&recv_ipc_sharable_cu_fabric_handles);
    local_partition_memory_pointer_ =
      static_cast<char*>(cu_alloc_handle_.mapped_whole_memory) + get_local_offset();
  }
  void unmap_and_destroy_driver_host_memory() noexcept
  {
    try {
      communicator_barrier(comm_);
      for (int i = 0; i < comm_->world_size; i++) {
        size_t mem_size = alloc_strategy_.alloc_sizes[i];
        if (mem_size > 0) {
          WM_CU_CHECK(
            cuMemUnmap(reinterpret_cast<CUdeviceptr>(cu_alloc_handle_.mapped_whole_memory) +
                         alloc_strategy_.alloc_offsets[i],
                       mem_size));
          if (i != comm_->world_rank) {
            WM_CU_CHECK(cuMemRelease(cu_alloc_handle_.all_cu_handles[i]));
          }
        }
      }
      communicator_barrier(comm_);
      if (alloc_strategy_.local_alloc_size > 0) {
        WM_CU_CHECK(cuMemRelease(cu_alloc_handle_.local_cu_handle));
      }
      WM_CU_CHECK(
        cuMemAddressFree(reinterpret_cast<CUdeviceptr>(cu_alloc_handle_.mapped_whole_memory),
                         alloc_strategy_.total_alloc_size));

      communicator_barrier(comm_);
    } catch (const wholememory::cu_error& wce) {
      WHOLEMEMORY_FAIL_NOTHROW("%s", wce.what());
    } catch (const raft::exception& re) {
      WHOLEMEMORY_FAIL_NOTHROW("%s", re.what());
    }
  }
};
#endif

void wholememory_impl::generate_rank_partition_strategy()
{
  if (!rank_partition_strategy_.partition_sizes_.empty()) {
    rank_partition_strategy_.partition_mem_stride = total_size_ / comm_->world_size;
    bool check_same                               = true;
    for (int i = 0; i < comm_->world_size - 2; i++) {  // ignore the last rank
      if (rank_partition_strategy_.partition_sizes_[i] !=
          rank_partition_strategy_.partition_sizes_[i + 1]) {
        check_same = false;
        break;
      }
    }
    rank_partition_strategy_.same_chunk = check_same;
    return;
  }
  size_t data_slot_count = total_size_ / data_granularity_;

  size_t data_slot_per_rank = 0;
  equal_partition_plan(&data_slot_per_rank, data_slot_count, comm_->world_size);

  rank_partition_strategy_.partition_sizes_.resize(comm_->world_size, 0);
  rank_partition_strategy_.partition_offsets_.resize(comm_->world_size + 1, 0);
  for (int i = 0; i < comm_->world_size; i++) {
    size_t tmp_slot_start = std::min(i * data_slot_per_rank, data_slot_count);
    size_t tmp_slot_end   = std::min((i + 1) * data_slot_per_rank, data_slot_count);
    rank_partition_strategy_.partition_sizes_[i] =
      (tmp_slot_end - tmp_slot_start) * data_granularity_;
    rank_partition_strategy_.partition_offsets_[i] = tmp_slot_start * data_granularity_;
  }
  rank_partition_strategy_.partition_offsets_[comm_->world_size] =
    data_slot_count * data_granularity_;
  rank_partition_strategy_.partition_mem_stride = data_slot_per_rank * data_granularity_;
  rank_partition_strategy_.same_chunk           = true;
}

void wholememory_impl::first_rank_allocate_all_strategy()
{
  alloc_strategy_.total_alloc_size = total_size_;
  // only first rank allocate memory
  alloc_strategy_.local_alloc_size =
    (comm_->world_rank == 0) ? alloc_strategy_.total_alloc_size : 0;
  alloc_strategy_.alignment = comm_->alloc_granularity;

  alloc_strategy_.alloc_offsets.clear();
  alloc_strategy_.alloc_offsets.resize(comm_->world_size, alloc_strategy_.total_alloc_size);
  alloc_strategy_.alloc_offsets[0] = 0;

  alloc_strategy_.alloc_sizes.clear();
  alloc_strategy_.alloc_sizes.resize(comm_->world_size, 0);
  alloc_strategy_.alloc_sizes[0] = alloc_strategy_.total_alloc_size;
}

void wholememory_impl::each_rank_different_chunk_strategy()
{
  alloc_strategy_.alloc_offsets.clear();
  alloc_strategy_.alloc_offsets.resize(comm_->world_size, 0);
  alloc_strategy_.alloc_sizes.clear();
  alloc_strategy_.alloc_sizes.resize(comm_->world_size, 0);

  size_t rank_local_alloc_offset = 0;
  for (int i = 0; i < comm_->world_size; i++) {
    size_t rank_local_alloc_size = rank_partition_strategy_.partition_sizes_[i];
    size_t rank_alignment;
    if (total_size_ > HUGE_PAGE_THRESHOLD) {
      rank_local_alloc_size = round_up_unsafe(rank_local_alloc_size, HUGE_PAGE_SIZE);
      rank_alignment        = HUGE_PAGE_SIZE;
    } else {
      rank_local_alloc_size = round_up_unsafe(rank_local_alloc_size, comm_->alloc_granularity);
      rank_alignment        = comm_->alloc_granularity;
    }
    if (i == comm_->world_rank) {
      alloc_strategy_.local_alloc_size = rank_local_alloc_size;
      alloc_strategy_.alignment        = rank_alignment;
    }
    alloc_strategy_.alloc_offsets[i] = rank_local_alloc_offset;
    alloc_strategy_.alloc_sizes[i]   = rank_local_alloc_size;
    rank_local_alloc_offset += rank_local_alloc_size;
  }
  alloc_strategy_.total_alloc_size = rank_local_alloc_offset;
}

void wholememory_impl::each_rank_multiple_page_strategy()
{
  size_t page_size = comm_->alloc_granularity;
  if (total_size_ >= HUGE_PAGE_THRESHOLD) page_size = HUGE_PAGE_SIZE;
  alloc_strategy_.alignment        = page_size;
  alloc_strategy_.total_alloc_size = round_up_unsafe(total_size_, page_size);
  size_t total_alloc_page_count    = alloc_strategy_.total_alloc_size / page_size;
  size_t rank_page_start           = comm_->world_rank * total_alloc_page_count / comm_->world_size;
  size_t rank_page_end = (comm_->world_rank + 1) * total_alloc_page_count / comm_->world_size;
  size_t page_count    = rank_page_end - rank_page_start;
  alloc_strategy_.local_alloc_size = page_count * page_size;

  alloc_strategy_.alloc_offsets.resize(comm_->world_size, 0);
  alloc_strategy_.alloc_sizes.resize(comm_->world_size, 0);
  for (int i = 0; i < comm_->world_size; i++) {
    size_t rank_i_page_start         = i * total_alloc_page_count / comm_->world_size;
    size_t rank_i_page_end           = (i + 1) * total_alloc_page_count / comm_->world_size;
    alloc_strategy_.alloc_offsets[i] = rank_i_page_start * page_size;
    alloc_strategy_.alloc_sizes[i]   = (rank_i_page_end - rank_i_page_start) * page_size;
  }
}

int negotiate_handle_id_with_comm_locked(wholememory_comm_t wm_comm)
{
  WM_COMM_CHECK_ALL_SAME(wm_comm, WM_MEM_OP_EXCHANGE_ID);
  int id        = 0;
  bool all_same = false;
  std::vector<int> rank_ids(wm_comm->world_size);
  auto& id_handle_map = wm_comm->wholememory_map;
  while (!all_same) {
    while (id_handle_map.find(id) != id_handle_map.end())
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
  return id;
}

struct wholememory_create_param {
  wholememory_create_param()                                           = default;
  wholememory_create_param(const struct wholememory_create_param&)     = default;
  wholememory_create_param(struct wholememory_create_param&&)          = default;
  wholememory_create_param& operator=(const wholememory_create_param&) = default;
  wholememory_create_param& operator=(wholememory_create_param&&)      = default;
  wholememory_create_param(size_t ts,
                           wholememory_memory_type_t mt,
                           wholememory_memory_location_t ml,
                           size_t mg)
  {
    total_size      = ts;
    memory_type     = mt;
    memory_location = ml;
    min_granularity = mg;
  }
  bool operator==(const wholememory_create_param& rhs) const
  {
    return total_size == rhs.total_size && memory_type == rhs.memory_type &&
           memory_location == rhs.memory_location && min_granularity == rhs.min_granularity;
  }
  bool operator!=(const wholememory_create_param& rhs) const { return !(*this == rhs); }
  size_t total_size;
  wholememory_memory_type_t memory_type;
  wholememory_memory_location_t memory_location;
  size_t min_granularity;
};

class hierarchy_wholememory_impl : public distributed_wholememory_impl {
 public:
  hierarchy_wholememory_impl(wholememory_handle_t wholememory_handle,
                             size_t total_size,
                             wholememory_comm_t global_comm,
                             wholememory_comm_t local_comm,
                             wholememory_memory_type_t memory_type,
                             wholememory_memory_location_t memory_location,
                             size_t data_granularity,
                             size_t* rank_entry_partition)
    : distributed_wholememory_impl(wholememory_handle,
                                   total_size,
                                   global_comm,
                                   memory_type,
                                   memory_location,
                                   data_granularity,
                                   rank_entry_partition)
  {
    WHOLEMEMORY_CHECK(memory_type == WHOLEMEMORY_MT_HIERARCHY);
    local_comm_    = local_comm;
    int world_rank = -1, world_size = -1, local_size = -1;
    wholememory_communicator_get_rank(&world_rank, global_comm);
    wholememory_communicator_get_size(&world_size, global_comm);
    wholememory_communicator_get_size(&local_size, local_comm);
    WHOLEMEMORY_CHECK(world_size % local_size == 0);
    wholememory_split_communicator(
      &cross_comm_, global_comm, world_rank % local_size, world_rank / local_size);
  }

  [[nodiscard]] wholememory_comm_t get_local_comm() const { return local_comm_; }
  [[nodiscard]] wholememory_comm_t get_cross_comm() const { return cross_comm_; }

 protected:
  wholememory_comm_t local_comm_;
  wholememory_comm_t cross_comm_;
};

wholememory_error_code_t create_wholememory(wholememory_handle_t* wholememory_handle_ptr,
                                            size_t total_size,
                                            wholememory_comm_t comm,
                                            wholememory_memory_type_t memory_type,
                                            wholememory_memory_location_t memory_location,
                                            size_t data_granularity,
                                            size_t* rank_entry_partition) noexcept
{
  try {
    if (total_size % data_granularity != 0) return WHOLEMEMORY_INVALID_VALUE;
    if (rank_entry_partition != nullptr) {
      int64_t total_slot_count = 0;
      for (int i = 0; i < comm->world_size; i++) {
        WM_COMM_CHECK_ALL_SAME(comm, rank_entry_partition[i]);
        if (rank_entry_partition[i] <= 0) { return WHOLEMEMORY_INVALID_VALUE; }
        total_slot_count += rank_entry_partition[i];
      }
      if (total_slot_count * data_granularity != total_size) {
        WHOLEMEMORY_ERROR("total slot count * data granularity (%ld*%ld) != total size (%ld)",
                          total_slot_count,
                          data_granularity,
                          total_size);
        return WHOLEMEMORY_INVALID_VALUE;
      }
    }
    *wholememory_handle_ptr = nullptr;
    std::unique_lock<std::mutex> mlock(comm->mu);
    auto* whole_memory_handle = new wholememory_handle_();

    whole_memory_handle->handle_id = negotiate_handle_id_with_comm_locked(comm);
    WM_COMM_CHECK_ALL_SAME(comm, WM_MEM_OP_CREATE);
    wholememory_create_param wcp(total_size, memory_type, memory_location, data_granularity);
    WM_COMM_CHECK_ALL_SAME(comm, wcp);

    if (memory_type == WHOLEMEMORY_MT_DISTRIBUTED) {
#ifdef WITH_NVSHMEM_SUPPORT
      if (comm->bind_to_nvshmem) {
        whole_memory_handle->impl = new nvshmem_device_wholememory_impl(whole_memory_handle,
                                                                        total_size,
                                                                        comm,
                                                                        memory_type,
                                                                        memory_location,
                                                                        data_granularity,
                                                                        rank_entry_partition);
      } else
#endif
      {
        whole_memory_handle->impl = new distributed_wholememory_impl(whole_memory_handle,
                                                                     total_size,
                                                                     comm,
                                                                     memory_type,
                                                                     memory_location,
                                                                     data_granularity,
                                                                     rank_entry_partition);
      }
    } else if (memory_type == WHOLEMEMORY_MT_CONTINUOUS) {
      if (is_intranode_communicator(comm) || !SupportEGM()) {
        if (memory_location == WHOLEMEMORY_ML_HOST) {
          whole_memory_handle->impl = new global_mapped_host_wholememory_impl(whole_memory_handle,
                                                                              total_size,
                                                                              comm,
                                                                              memory_type,
                                                                              memory_location,
                                                                              data_granularity,
                                                                              rank_entry_partition);
        } else {
          whole_memory_handle->impl = new continuous_device_wholememory_impl(whole_memory_handle,
                                                                             total_size,
                                                                             comm,
                                                                             memory_type,
                                                                             memory_location,
                                                                             data_granularity,
                                                                             rank_entry_partition);
        }
      } else {
#if CUDA_VERSION >= 12030
        whole_memory_handle->impl = new continuous_mnnvl_wholememory_impl(whole_memory_handle,
                                                                          total_size,
                                                                          comm,
                                                                          memory_type,
                                                                          memory_location,
                                                                          data_granularity,
                                                                          rank_entry_partition);
#else
        WHOLEMEMORY_FAIL_NOTHROW("Multinode CONTINUOUS is only supported on CUDA Version >= 12.3");
#endif
      }
    } else if (memory_type == WHOLEMEMORY_MT_CHUNKED) {
      WHOLEMEMORY_CHECK_NOTHROW(is_intranode_communicator(comm));
      if (memory_location == WHOLEMEMORY_ML_HOST) {
        whole_memory_handle->impl = new global_mapped_host_wholememory_impl(whole_memory_handle,
                                                                            total_size,
                                                                            comm,
                                                                            memory_type,
                                                                            memory_location,
                                                                            data_granularity,
                                                                            rank_entry_partition);
      } else {
        whole_memory_handle->impl = new chunked_device_wholememory_impl(whole_memory_handle,
                                                                        total_size,
                                                                        comm,
                                                                        memory_type,
                                                                        memory_location,
                                                                        data_granularity,
                                                                        rank_entry_partition);
      }
    } else if (memory_type == WHOLEMEMORY_MT_HIERARCHY) {
      wholememory_comm_t local_comm;
      int world_rank = -1, local_size = -1;
      wholememory_communicator_get_rank(&world_rank, comm);
      wholememory_communicator_get_local_size(&local_size, comm);
      wholememory_split_communicator(
        &local_comm, comm, world_rank / local_size, world_rank % local_size);
      whole_memory_handle->impl = new hierarchy_wholememory_impl(whole_memory_handle,
                                                                 total_size,
                                                                 comm,
                                                                 local_comm,
                                                                 memory_type,
                                                                 memory_location,
                                                                 data_granularity,
                                                                 rank_entry_partition);
    } else {
      WHOLEMEMORY_FATAL("Unsupported memory_type (%d) and memory_location (%d).",
                        (int)memory_type,
                        (int)memory_location);
    }
    whole_memory_handle->impl->create_memory();

    comm->wholememory_map.insert(
      std::pair<int, wholememory_handle_t>(whole_memory_handle->handle_id, whole_memory_handle));

    *wholememory_handle_ptr = whole_memory_handle;
    return WHOLEMEMORY_SUCCESS;
  } catch (const wholememory::cuda_error& wce) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", wce.what());
  } catch (const raft::logic_error& rle) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", rle.what());
  } catch (const wholememory::logic_error& wle) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", wle.what());
  } catch (const raft::exception& re) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", re.what());
  } catch (...) {
    WHOLEMEMORY_FAIL_NOTHROW("Unknown exception.");
  }
}

wholememory_error_code_t destroy_wholememory_with_comm_locked(
  wholememory_handle_t wholememory_handle) noexcept
{
  try {
    if (wholememory_handle == nullptr) return WHOLEMEMORY_INVALID_INPUT;
    if (wholememory_handle->impl == nullptr) return WHOLEMEMORY_INVALID_INPUT;
    auto* comm = wholememory_handle->impl->get_comm();
    if (comm == nullptr) return WHOLEMEMORY_INVALID_INPUT;

    if (comm->wholememory_map.find(wholememory_handle->handle_id) == comm->wholememory_map.end()) {
      return WHOLEMEMORY_INVALID_VALUE;
    }

    WM_COMM_CHECK_ALL_SAME(comm, WM_MEM_OP_DESTROY);
    WM_COMM_CHECK_ALL_SAME(comm, wholememory_handle->handle_id);

    comm->wholememory_map.erase(wholememory_handle->handle_id);
    delete wholememory_handle;

    return WHOLEMEMORY_SUCCESS;
  } catch (const wholememory::cuda_error& wce) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", wce.what());
  } catch (const raft::logic_error& rle) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", rle.what());
  } catch (const wholememory::logic_error& wle) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", wle.what());
  } catch (const raft::exception& re) {
    WHOLEMEMORY_FAIL_NOTHROW("%s", re.what());
  } catch (...) {
    WHOLEMEMORY_FAIL_NOTHROW("Unknown exception.");
  }
}

wholememory_error_code_t destroy_wholememory(wholememory_handle_t wholememory_handle) noexcept
{
  wholememory_comm_t comm = wholememory_handle->impl->get_comm();
  std::unique_lock<std::mutex> mlock(comm->mu);
  return destroy_wholememory_with_comm_locked(wholememory_handle);
}

wholememory_error_code_t get_communicator_from_handle(
  wholememory_comm_t* comm, wholememory_handle_t wholememory_handle) noexcept
{
  if (wholememory_handle == nullptr || wholememory_handle->impl == nullptr) {
    return WHOLEMEMORY_INVALID_INPUT;
  }
  *comm = wholememory_handle->impl->get_comm();
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t get_local_communicator_from_handle(
  wholememory_comm_t* comm, wholememory_handle_t wholememory_handle) noexcept
{
  if (wholememory_handle == nullptr || wholememory_handle->impl == nullptr) {
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (get_memory_type(wholememory_handle) != WHOLEMEMORY_MT_HIERARCHY) {
    return WHOLEMEMORY_NOT_SUPPORTED;
  }
  hierarchy_wholememory_impl* hierarchy_impl =
    dynamic_cast<hierarchy_wholememory_impl*>(wholememory_handle->impl);
  *comm = hierarchy_impl->get_local_comm();
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t get_cross_communicator_from_handle(
  wholememory_comm_t* comm, wholememory_handle_t wholememory_handle) noexcept
{
  if (wholememory_handle == nullptr || wholememory_handle->impl == nullptr) {
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (get_memory_type(wholememory_handle) != WHOLEMEMORY_MT_HIERARCHY) {
    return WHOLEMEMORY_NOT_SUPPORTED;
  }
  hierarchy_wholememory_impl* hierarchy_impl =
    dynamic_cast<hierarchy_wholememory_impl*>(wholememory_handle->impl);
  *comm = hierarchy_impl->get_cross_comm();
  return WHOLEMEMORY_SUCCESS;
}

wholememory_memory_type_t get_memory_type(wholememory_handle_t wholememory_handle) noexcept
{
  return wholememory_handle->impl->get_type();
}

wholememory_memory_location_t get_memory_location(wholememory_handle_t wholememory_handle) noexcept
{
  return wholememory_handle->impl->get_location();
}

wholememory_distributed_backend_t get_distributed_backend_t(
  wholememory_handle_t wholememory_handle) noexcept
{
  return wholememory_handle->impl->get_distributed_backend();
}

size_t get_total_size(wholememory_handle_t wholememory_handle) noexcept
{
  return wholememory_handle->impl->total_size();
}

size_t get_data_granularity(wholememory_handle_t wholememory_handle) noexcept
{
  return wholememory_handle->impl->data_granularity();
}

wholememory_error_code_t get_local_memory_from_handle(
  void** local_ptr,
  size_t* local_size,
  size_t* local_offset,
  wholememory_handle_t wholememory_handle) noexcept
{
  if (wholememory_handle == nullptr || wholememory_handle->impl == nullptr) {
    return WHOLEMEMORY_INVALID_INPUT;
  }
  wholememory_handle->impl->get_local_memory(local_ptr, local_size, local_offset);
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t get_rank_memory_from_handle(
  void** rank_memory_ptr,
  size_t* rank_memory_size,
  size_t* rank_memory_offset,
  int rank,
  wholememory_handle_t wholememory_handle) noexcept
{
  if (wholememory_handle == nullptr || wholememory_handle->impl == nullptr) {
    return WHOLEMEMORY_INVALID_INPUT;
  }
  auto* comm = wholememory_handle->impl->get_comm();
  if (rank < 0 || rank >= comm->world_size) { return WHOLEMEMORY_INVALID_INPUT; }
  if (wholememory_handle->impl->get_rank_memory(
        rank_memory_ptr, rank_memory_size, rank_memory_offset, rank) == false) {
    return WHOLEMEMORY_INVALID_INPUT;
  }
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t get_global_pointer_from_handle(
  void** global_ptr, wholememory_handle_t wholememory_handle) noexcept
{
  if (wholememory_handle == nullptr || wholememory_handle->impl == nullptr) {
    return WHOLEMEMORY_INVALID_INPUT;
  }
  *global_ptr = wholememory_handle->impl->get_continuous_mapping_pointer();
  return (*global_ptr) == nullptr ? WHOLEMEMORY_INVALID_INPUT : WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t get_global_reference_from_handle(
  wholememory_gref_t* wholememory_gref, wholememory_handle_t wholememory_handle) noexcept
{
  if (wholememory_handle == nullptr || wholememory_handle->impl == nullptr) {
    return WHOLEMEMORY_INVALID_INPUT;
  }
  *wholememory_gref = wholememory_handle->impl->get_global_reference();
  return (wholememory_gref->pointer == nullptr) ? WHOLEMEMORY_INVALID_INPUT : WHOLEMEMORY_SUCCESS;
}

#ifdef WITH_NVSHMEM_SUPPORT

wholememory_error_code_t get_nvshmem_reference_frome_handle(
  wholememory_nvshmem_ref_t* wholememory_nvshmem_ref,
  wholememory_handle_t wholememory_handle) noexcept
{
  if (wholememory_handle == nullptr || wholememory_handle->impl == nullptr ||
      wholememory_handle->impl->get_type() != WHOLEMEMORY_MT_DISTRIBUTED ||
      (wholememory_handle->impl->get_distributed_backend() != WHOLEMEMORY_DB_NVSHMEM)) {
    return WHOLEMEMORY_INVALID_INPUT;
  }
  wholememory_gref_t wholememory_gref_tmp      = wholememory_handle->impl->get_global_reference();
  *wholememory_nvshmem_ref                     = wholememory_nvshmem_ref_t{};
  wholememory_nvshmem_ref->pointer             = wholememory_gref_tmp.pointer;
  wholememory_nvshmem_ref->rank_memory_offsets = wholememory_gref_tmp.rank_memory_offsets;
  wholememory_nvshmem_ref->world_size          = wholememory_gref_tmp.world_size;
  wholememory_nvshmem_ref->world_rank          = wholememory_handle->impl->get_comm()->world_rank;
  wholememory_nvshmem_ref->stride              = wholememory_gref_tmp.stride;
  wholememory_nvshmem_ref->same_chunk          = wholememory_gref_tmp.same_chunk;
  return (wholememory_nvshmem_ref->pointer == nullptr) ? WHOLEMEMORY_INVALID_INPUT
                                                       : WHOLEMEMORY_SUCCESS;
}

#endif

wholememory_error_code_t equal_partition_plan(size_t* entry_per_rank,
                                              size_t total_entry_count,
                                              int world_size) noexcept
{
  *entry_per_rank = div_rounding_up_safe<size_t>(total_entry_count, world_size);
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t get_rank_partition_sizes_from_handle(
  size_t* rank_sizes, wholememory_handle_t wholememory_handle) noexcept
{
  if (wholememory_handle == nullptr || wholememory_handle->impl == nullptr) {
    return WHOLEMEMORY_INVALID_INPUT;
  }
  std::vector<size_t> rank_sizes_ = wholememory_handle->impl->get_rank_sizes();
  for (int i = 0; i < rank_sizes_.size(); i++)
    rank_sizes[i] = rank_sizes_[i];
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t get_rank_partition_offsets_from_handle(
  size_t* rank_offsets, wholememory_handle_t wholememory_handle) noexcept
{
  if (wholememory_handle == nullptr || wholememory_handle->impl == nullptr) {
    return WHOLEMEMORY_INVALID_INPUT;
  }
  std::vector<size_t> rank_offsets_ = wholememory_handle->impl->get_rank_offsets();
  for (int i = 0; i < rank_offsets_.size(); i++)
    rank_offsets[i] = rank_offsets_[i];
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t get_local_size_from_handle(
  size_t* rank_size, wholememory_handle_t wholememory_handle) noexcept
{
  if (wholememory_handle == nullptr || wholememory_handle->impl == nullptr) {
    return WHOLEMEMORY_INVALID_INPUT;
  }
  *rank_size = wholememory_handle->impl->get_local_size();
  return WHOLEMEMORY_SUCCESS;
}

wholememory_error_code_t get_local_offset_from_handle(
  size_t* local_offset, wholememory_handle_t wholememory_handle) noexcept
{
  if (wholememory_handle == nullptr || wholememory_handle->impl == nullptr) {
    return WHOLEMEMORY_INVALID_INPUT;
  }
  *local_offset = wholememory_handle->impl->get_local_offset();
  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory

wholememory_handle_::~wholememory_handle_()
{
  if (impl != nullptr) {
    impl->destroy_memory();
    delete impl;
    impl = nullptr;
  }
}
