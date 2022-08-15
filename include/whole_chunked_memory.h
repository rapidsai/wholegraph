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
#pragma once

#include <cuda_runtime_api.h>

namespace whole_graph {

typedef struct WholeChunkedMemory *WholeChunkedMemory_t;
class BootstrapCommunicator;

#define MAX_DEVICE_COUNT (16)
struct WholeChunkedMemoryHandle {
  void *chunked_ptrs[MAX_DEVICE_COUNT];
  size_t chunk_size;
  int chunk_count;
};

/*!
 * WholeMemory Multiple Process mode malloc for WholeChunkedMemory
 * @param pwcmt : return WholeChunkedMemory object allocated
 * @param size : allocation size
 * @param bootstrap_communicator : bootstrap communicator
 * @param min_granularity : min_granularity of each chunk
 */
void WcmmpMalloc(WholeChunkedMemory_t *pwcmt,
                 size_t size,
                 BootstrapCommunicator *bootstrap_communicator,
                 size_t min_granularity);

/*!
 * WholeMemory Multiple Process mode free for WholeChunkedMemory
 * @param wcmt : WholeChunkedMemory object to free
 */
void WcmmpFree(WholeChunkedMemory_t wcmt);

/*!
 * Get bootstrap communicator
 * @param wcmt : WholeChunkedMemory
 * @return : bootstrap communicator
 */
BootstrapCommunicator *WcmmpGetBootstrapCommunicator(const WholeChunkedMemory_t wcmt);

/*!
 * Get the WholeChunkedMemoryHandle of from WholeChunkedMemory
 * @param wcmt : WholeChunkedMemory object
 * @param dev_id : device id of WholeChunkedMemoryHandle to get, -1 for CPU
 * @return the WholeChunkedMemoryHandle on device dev_id
 */
WholeChunkedMemoryHandle *GetDeviceChunkedHandle(WholeChunkedMemory_t wcmt, int dev_id);

/*!
 * Get local memory ptr and size.
 * @param wcmt : WholeChunkedMemory object
 * @param ptr : return pointer
 * @param size : return size
 */
void WcmmpGetLocalMemory(WholeChunkedMemory_t wcmt, void **ptr, size_t *size);

/*!
 * Copy from pinned host memory or device memory to WholeChunkedMemory
 * @param wcmh : WholeChunkedMemoryHandle object to copy to
 * @param offset_in_bytes : offset in WholeChunkedMemory to copy to
 * @param src : source memory pointer
 * @param copy_bytes : copy bytes
 */
void WcmmpMemcpyToWholeChunkedMemory(WholeChunkedMemoryHandle *wcmh,
                                     size_t offset_in_bytes,
                                     const void *src,
                                     size_t copy_bytes,
                                     cudaStream_t stream);

/*!
 * Copy from WholeChunkedMemory to pinned host memory or device memory
 * @param dst : destination memory pointer
 * @param wcmh : WholeChunkedMemoryHandle to copy from
 * @param offset_in_bytes : offset in WholeChunkedMemory to copy from
 * @param copy_bytes : copy bytes
 */
void WcmmpMemcpyFromWholeChunkedMemory(void *dst,
                                       WholeChunkedMemoryHandle *wcmh,
                                       size_t offset_in_bytes,
                                       size_t copy_bytes,
                                       cudaStream_t stream);

}// namespace whole_graph