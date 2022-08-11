#pragma once

#include <cuda_runtime_api.h>

namespace whole_graph {

typedef struct WholeChunkedMemory* WholeChunkedMemory_t;

#define MAX_DEVICE_COUNT (16)
struct WholeChunkedMemoryHandle {
  void* chunked_ptrs[MAX_DEVICE_COUNT];
  size_t chunk_size;
  int chunk_count;
};

/*!
 * WholeMemory Multiple Process mode malloc for WholeChunkedMemory
 * @param pwcmt : return WholeChunkedMemory object allocated
 * @param size : allocation size
 * @param min_granularity : min_granularity of each chunk, will be multiplied to 16 byte aligned
 * @param ranks : ranks participate in this allocation, nullptr means all ranks
 * @param rank_count : rank count participate in this allocation, 0 means all ranks
 */
void WcmmpMalloc(WholeChunkedMemory_t* pwcmt, size_t size, size_t min_granularity = 0, const int* ranks = nullptr, int rank_count = 0);

/*!
 * WholeMemory Multiple Process mode free for WholeChunkedMemory
 * @param wcmt : WholeChunkedMemory object to free
 */
void WcmmpFree(WholeChunkedMemory_t wcmt);

/*!
 * Get the WholeChunkedMemoryHandle of from WholeChunkedMemory
 * @param wcmt : WholeChunkedMemory object
 * @param dev_id : device id of WholeChunkedMemoryHandle to get, -1 for CPU
 * @return the WholeChunkedMemoryHandle on device dev_id
 */
WholeChunkedMemoryHandle* GetDeviceChunkedHandle(WholeChunkedMemory_t wcmt, int dev_id);

/*!
 * Copy from pinned host memory or device memory to WholeChunkedMemory
 * @param wcmh : WholeChunkedMemoryHandle object to copy to
 * @param offset_in_bytes : offset in WholeChunkedMemory to copy to
 * @param src : source memory pointer
 * @param copy_bytes : copy bytes
 */
void WcmmpMemcpyToWholeChunkedMemory(WholeChunkedMemoryHandle* wcmh, size_t offset_in_bytes, const void* src, size_t copy_bytes, cudaStream_t stream);

/*!
 * Copy from WholeChunkedMemory to pinned host memory or device memory
 * @param dst : destination memory pointer
 * @param wcmh : WholeChunkedMemoryHandle to copy from
 * @param offset_in_bytes : offset in WholeChunkedMemory to copy from
 * @param copy_bytes : copy bytes
 */
void WcmmpMemcpyFromWholeChunkedMemory(void* dst, WholeChunkedMemoryHandle* wcmh, size_t offset_in_bytes, size_t copy_bytes, cudaStream_t stream);

}