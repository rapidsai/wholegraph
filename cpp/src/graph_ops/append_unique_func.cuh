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

#include "cuda_macros.hpp"
#include "error.hpp"
#include "wholememory_ops/output_memory_handle.hpp"
#include "wholememory_ops/temp_memory_handle.hpp"
#include "wholememory_ops/thrust_allocator.hpp"
#include <cooperative_groups.h>
#include <raft/util/integer_utils.hpp>
#include <thrust/scan.h>
#include <wholememory/env_func_ptrs.h>
#include <wholememory/tensor_description.h>

namespace graph_ops {

static constexpr int kAssignBucketSize      = 32;  // it is not adjustable
static constexpr int kAssignThreadBlockSize = 8 * 32;

template <typename KeyT, int BucketSize>
class AppendUniqueHash;

template <typename KeyT, int BucketSize, bool IsTarget = true>
__global__ void InsertKeysKernel(AppendUniqueHash<KeyT, BucketSize> auh);

template <typename KeyT, int BucketSize, bool IsTarget = false>
__global__ void RetrieveKeysKernel(AppendUniqueHash<KeyT, BucketSize> auh, int* output);

template <typename T>
__device__ __forceinline__ T atomicCASSigned(T* ptr, T cmp, T val)
{
  return atomicCAS(ptr, cmp, val);
}

template <>
__device__ __forceinline__ int64_t atomicCASSigned<int64_t>(int64_t* ptr, int64_t cmp, int64_t val)
{
  return (int64_t)atomicCAS((unsigned long long*)ptr, cmp, val);
}

template <typename KeyT, int BucketSize = kAssignBucketSize / sizeof(KeyT)>
class AppendUniqueHash {
 public:
  AppendUniqueHash(int target_count, int neighbor_count, const KeyT* targets, const KeyT* neighbors)
    : target_count_(target_count),
      neighbor_count_(neighbor_count),
      targets_(targets),
      neighbors_(neighbors)
  {
    int total_slots_needed = (target_count + neighbor_count) * 2;
    total_slots_needed =
      raft::div_rounding_up_safe<int>(total_slots_needed, kAssignBucketSize) * kAssignBucketSize;
    bucket_count_ = raft::div_rounding_up_safe<int>(total_slots_needed, BucketSize) + 1;
  }
  ~AppendUniqueHash() {}
  void AllocateMemoryAndInit(wholememory_ops::temp_memory_handle& hash_teable_keys_tmh,
                             wholememory_ops::temp_memory_handle& hash_teable_values_tmh,
                             cudaStream_t stream)
  {
    // compute bucket_count_ and allocate memory.
    size_t total_alloc_slots =
      raft::div_rounding_up_safe<int>(bucket_count_ * BucketSize, kAssignThreadBlockSize) *
      kAssignThreadBlockSize;
    wholememory_dtype_t table_key_wholememory_dtype = WHOLEMEMORY_DT_INT;
    if (sizeof(KeyT) == 8) { table_key_wholememory_dtype = WHOLEMEMORY_DT_INT64; }
    table_keys_ =
      (KeyT*)hash_teable_keys_tmh.device_malloc(total_alloc_slots, table_key_wholememory_dtype);
    value_id_ = (int*)hash_teable_values_tmh.device_malloc(total_alloc_slots, WHOLEMEMORY_DT_INT);

    // init key to -1
    WM_CUDA_CHECK(cudaMemsetAsync(table_keys_, -1, total_alloc_slots * sizeof(KeyT), stream));
    // init value_id to -1
    WM_CUDA_CHECK(cudaMemsetAsync(value_id_, -1, total_alloc_slots * sizeof(int), stream));
  }

  void InsertKeys(cudaStream_t stream)
  {
    const int thread_count = 512;
    int target_block_count =
      raft::div_rounding_up_safe<int>(target_count_ * BucketSize, thread_count);
    InsertKeysKernel<KeyT, BucketSize, true>
      <<<target_block_count, thread_count, 0, stream>>>(*this);
    WM_CUDA_CHECK(cudaStreamSynchronize(stream));
    int neighbor_block_count =
      raft::div_rounding_up_safe<int>(neighbor_count_ * BucketSize, thread_count);
    InsertKeysKernel<KeyT, BucketSize, false>
      <<<neighbor_block_count, thread_count, 0, stream>>>(*this);
  }
  void RetrieveNeighborKeysForValueIDs(cudaStream_t stream, int* value_ids)
  {
    const int thread_count = 512;
    int target_block_count =
      raft::div_rounding_up_safe<int>(neighbor_count_ * BucketSize, thread_count);
    RetrieveKeysKernel<KeyT, BucketSize>
      <<<target_block_count, thread_count, 0, stream>>>(*this, value_ids);
  }
  __host__ __device__ __forceinline__ int TargetCount() { return target_count_; }
  __host__ __device__ __forceinline__ int NeighborCount() { return neighbor_count_; }
  __host__ __device__ __forceinline__ const KeyT* Targets() { return targets_; }
  __host__ __device__ __forceinline__ const KeyT* Neighbors() { return neighbors_; }
  __host__ __device__ __forceinline__ KeyT* TableKeys() { return table_keys_; }
  __host__ __device__ __forceinline__ int32_t* ValueID() { return value_id_; }

  size_t SlotCount() { return bucket_count_ * BucketSize; }
  void GetBucketLayout(int* bucket_count, int* bucket_size)
  {
    *bucket_count = bucket_count_;
    *bucket_size  = BucketSize;
  }
  static constexpr KeyT kInvalidKey       = -1LL;
  static constexpr int kInvalidValueID    = -1;
  static constexpr int kNeedAssignValueID = -2;

  __device__ __forceinline__ int retrieve_key(
    const KeyT& key, cooperative_groups::thread_block_tile<BucketSize>& group)
  {
    // On find, return global slot offset
    // On not find, return new slot and set key. Not find and don't need new
    // slot case should not happen.
    int base_bucket_id = bucket_for_key(key);
    int bucket_id;
    int local_slot_offset = -1;
    int try_idx           = 0;
    do {
      bucket_id         = bucket_id_on_conflict(base_bucket_id, try_idx);
      local_slot_offset = key_in_bucket(key, bucket_id, group);
      try_idx++;
    } while (local_slot_offset < 0);
    return bucket_id * BucketSize + local_slot_offset;
  }
  __device__ __forceinline__ void insert_key(
    const KeyT& key, const int id, cooperative_groups::thread_block_tile<BucketSize>& group)
  {
    int slot_offset   = retrieve_key(key, group);
    int* value_id_ptr = value_id_ + slot_offset;
    if (group.thread_rank() == 0) {
      if (id == kNeedAssignValueID) {
        // neighbor
        atomicCAS(value_id_ptr, kInvalidValueID, id);
      } else {
        // target
        *value_id_ptr = id;
      }
    }
  }

 private:
  __device__ __forceinline__ int bucket_for_key(const KeyT& key)
  {
    const uint32_t hash_value =
      ((uint32_t)((uint64_t)key >> 32ULL)) * 0x85ebca6b + (uint32_t)((uint64_t)key & 0xFFFFFFFFULL);
    return hash_value % bucket_count_;
  }
  __device__ __forceinline__ int bucket_id_on_conflict(int base_bucket_id, int try_idx)
  {
    return (base_bucket_id + try_idx * try_idx) % bucket_count_;
  }
  __device__ __forceinline__ int key_in_bucket(
    const KeyT& key, int bucket_id, cooperative_groups::thread_block_tile<BucketSize>& group)
  {
    // On find or inserted(no work thread should not do insertion), return local
    // slot offset. On not find and bucket is full, return -1. Should do CAS
    // loop cooperative_groups::thread_block_tile<BucketSize> g =
    // cooperative_groups::tiled_partition<BucketSize>(cooperative_groups::this_thread_block());
    KeyT* key_ptr          = table_keys_ + bucket_id * BucketSize + group.thread_rank();
    KeyT old_key           = *key_ptr;
    unsigned int match_key = group.ballot(old_key == key);
    int match_lane_id      = __ffs(match_key) - 1;
    if (match_lane_id >= 0) { return match_lane_id; }
    unsigned int empty_key = group.ballot(old_key == AppendUniqueHash<KeyT>::kInvalidKey);
    while (empty_key != 0) {
      int leader = __ffs((int)empty_key) - 1;
      KeyT old;
      if (group.thread_rank() == leader) { old = atomicCASSigned(key_ptr, old_key, key); }
      old     = group.shfl(old, leader);
      old_key = group.shfl(old_key, leader);
      if (old == old_key || old == key) {
        // success and duplicate.
        return leader;
      }
      empty_key ^= (1UL << (unsigned)leader);
    }
    return -1;
  }

  int bucket_count_;

  KeyT* table_keys_  = nullptr;  // -1 invalid
  int32_t* value_id_ = nullptr;  // -1 invalid, -2 need assign final neighbor id

  const KeyT* targets_   = nullptr;
  const KeyT* neighbors_ = nullptr;
  int target_count_;
  int neighbor_count_;
};

template <typename KeyT, int BucketSize, bool IsTarget>
__global__ void InsertKeysKernel(AppendUniqueHash<KeyT, BucketSize> auh)
{
  int input_key_count       = IsTarget ? auh.TargetCount() : auh.NeighborCount();
  const KeyT* input_key_ptr = IsTarget ? auh.Targets() : auh.Neighbors();
  int key_idx               = (blockIdx.x * blockDim.x + threadIdx.x) / BucketSize;
  cooperative_groups::thread_block_tile<BucketSize> group =
    cooperative_groups::tiled_partition<BucketSize>(cooperative_groups::this_thread_block());
  if (key_idx >= input_key_count) return;
  KeyT key = input_key_ptr[key_idx];
  int id   = IsTarget ? key_idx : AppendUniqueHash<KeyT, BucketSize>::kNeedAssignValueID;
  auh.insert_key(key, id, group);
}

template <typename KeyT, int BucketSize, bool IsTarget>
__global__ void RetrieveKeysKernel(AppendUniqueHash<KeyT, BucketSize> auh, int* output)
{
  int input_key_count         = IsTarget ? auh.TargetCount() : auh.NeighborCount();
  const KeyT* input_key_ptr   = IsTarget ? auh.Targets() : auh.Neighbors();
  const int* output_value_ptr = auh.ValueID();
  int key_idx                 = (blockIdx.x * blockDim.x + threadIdx.x) / BucketSize;
  cooperative_groups::thread_block_tile<BucketSize> group =
    cooperative_groups::tiled_partition<BucketSize>(cooperative_groups::this_thread_block());
  if (key_idx >= input_key_count) return;
  KeyT key   = input_key_ptr[key_idx];
  int offset = auh.retrieve_key(key, group);
  if (group.thread_rank() == 0) { output[key_idx] = output_value_ptr[offset]; }
}

template <typename KeyT>
__global__ void CountBucketKernel(const int* value_id, int* bucket_count_ptr)
{
  __shared__ int count_buffer[kAssignThreadBlockSize / kAssignBucketSize];
  int idx   = blockIdx.x * blockDim.x + threadIdx.x;
  int value = value_id[idx];
  unsigned int assign_mask =
    __ballot_sync(0xffffffff, value == AppendUniqueHash<KeyT>::kNeedAssignValueID);
  if (threadIdx.x % 32 == 0) {
    int assign_count               = __popc((int)assign_mask);
    count_buffer[threadIdx.x / 32] = assign_count;
  }
  __syncthreads();
  if (threadIdx.x < kAssignThreadBlockSize / kAssignBucketSize) {
    bucket_count_ptr[kAssignThreadBlockSize / kAssignBucketSize * blockIdx.x + threadIdx.x] =
      count_buffer[threadIdx.x];
  }
}

template <typename KeyT>
__global__ void AssignValueKernel(int* value_id, const int* bucket_prefix_sum_ptr, int target_count)
{
  int idx                  = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_start           = bucket_prefix_sum_ptr[idx / 32];
  int value                = value_id[idx];
  unsigned int thread_mask = (1UL << (threadIdx.x % 32)) - 1;
  unsigned int assign_mask =
    __ballot_sync(0xffffffff, value == AppendUniqueHash<KeyT>::kNeedAssignValueID);
  assign_mask &= thread_mask;
  int idx_in_warp  = __popc((int)assign_mask);
  int assigned_idx = idx_in_warp + warp_start;
  if (value == AppendUniqueHash<KeyT>::kNeedAssignValueID) {
    value_id[idx] = assigned_idx + target_count;
  }
}

template <typename KeyT>
__global__ void ComputeOutputUniqueNeighborAndCountKernel(const KeyT* table_keys,
                                                          const int* value_ids,
                                                          int target_count,
                                                          KeyT* unique_total_output)
{
  int idx      = blockIdx.x * blockDim.x + threadIdx.x;
  KeyT key     = table_keys[idx];
  int value_id = value_ids[idx];
  if (value_id >= target_count) { unique_total_output[value_id] = key; }
}

template <typename KeyT>
void graph_append_unique_func(void* target_nodes_ptr,
                              wholememory_array_description_t target_nodes_desc,
                              void* neighbor_nodes_ptr,
                              wholememory_array_description_t neighbor_nodes_desc,
                              void* output_unique_node_memory_context,
                              int* output_neighbor_raw_to_unique_mapping_ptr,
                              wholememory_env_func_t* p_env_fns,
                              cudaStream_t stream)
{
  int target_count   = target_nodes_desc.size;
  int neighbor_count = neighbor_nodes_desc.size;
  AppendUniqueHash<KeyT> auh(
    target_count, neighbor_count, (const KeyT*)target_nodes_ptr, (const KeyT*)neighbor_nodes_ptr);

  wholememory_ops::temp_memory_handle hash_teable_keys_tmh(p_env_fns);
  wholememory_ops::temp_memory_handle hash_teable_values_tmh(p_env_fns);
  auh.AllocateMemoryAndInit(hash_teable_keys_tmh, hash_teable_values_tmh, stream);
  auh.InsertKeys(stream);
  wholememory_ops::temp_memory_handle bucket_count_tm(p_env_fns), bucket_prefix_sum_tm(p_env_fns);
  int num_bucket_count  = raft::div_rounding_up_safe<int>(auh.SlotCount(), kAssignBucketSize) + 1;
  int* bucket_count_ptr = (int*)bucket_count_tm.device_malloc(num_bucket_count, WHOLEMEMORY_DT_INT);
  int* bucket_prefix_sum_ptr =
    (int*)bucket_prefix_sum_tm.device_malloc(num_bucket_count, WHOLEMEMORY_DT_INT);
  KeyT* table_keys = auh.TableKeys();
  int* value_id    = auh.ValueID();
  int num_blocks   = raft::div_rounding_up_safe<int>(auh.SlotCount(), kAssignThreadBlockSize);
  CountBucketKernel<KeyT>
    <<<num_blocks, kAssignThreadBlockSize, 0, stream>>>(value_id, bucket_count_ptr);
  WM_CUDA_CHECK(cudaGetLastError());
  wholememory_ops::wm_thrust_allocator thrust_allocator(p_env_fns);
  thrust::exclusive_scan(thrust::cuda::par_nosync(thrust_allocator).on(stream),
                         bucket_count_ptr,
                         bucket_count_ptr + num_bucket_count,
                         (int*)bucket_prefix_sum_ptr);
  int unique_neighbor_count = 0;
  WM_CUDA_CHECK(cudaMemcpyAsync(&unique_neighbor_count,
                                bucket_prefix_sum_ptr + num_bucket_count - 1,
                                sizeof(int),
                                cudaMemcpyDeviceToHost,
                                stream));
  WM_CUDA_CHECK(cudaStreamSynchronize(stream));

  AssignValueKernel<KeyT><<<num_blocks, kAssignThreadBlockSize, 0, stream>>>(
    value_id, bucket_prefix_sum_ptr, target_count);
  wholememory_ops::output_memory_handle gen_output_unique_node_buffer_mh(
    p_env_fns, output_unique_node_memory_context);

  KeyT* output_unique_node_ptr = (KeyT*)gen_output_unique_node_buffer_mh.device_malloc(
    unique_neighbor_count + target_count, target_nodes_desc.dtype);

  WM_CUDA_CHECK(cudaMemcpyAsync(output_unique_node_ptr,
                                target_nodes_ptr,
                                target_count * sizeof(KeyT),
                                cudaMemcpyDeviceToDevice,
                                stream));

  ComputeOutputUniqueNeighborAndCountKernel<KeyT>
    <<<num_blocks, kAssignThreadBlockSize, 0, stream>>>(
      table_keys, value_id, target_count, output_unique_node_ptr);
  if (output_neighbor_raw_to_unique_mapping_ptr) {
    auh.RetrieveNeighborKeysForValueIDs(stream, (int*)output_neighbor_raw_to_unique_mapping_ptr);
  }
  WM_CUDA_CHECK(cudaStreamSynchronize(stream));
}
}  // namespace graph_ops
