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

typedef struct WholeNCCLMemory *WholeNCCLMemory_t;
class BootstrapCommunicator;

/*!
 * WholeMemory Multiple Process mode malloc for WholeChunkedMemory
 * @param pwnmt : return WholeNCCLMemory object allocated
 * @param size : allocation size
 * @param bootstrap_communicator : bootstrap communicator
 * @param min_granularity : min_granularity of each rank, will be multiplied to 16 byte aligned
 */
void WnmmpMalloc(WholeNCCLMemory_t *pwnmt,
                 size_t size,
                 BootstrapCommunicator *bootstrap_communicator,
                 size_t min_granularity = 0);

/*!
 * WholeMemory Multiple Process mode free for WholeChunkedMemory
 * @param wnmt : WholeNCCLMemory object to free
 */
void WnmmpFree(WholeNCCLMemory_t wnmt);

/*!
 * Get bootstrap communicator
 * @param wnmt : WholeNCCLMemory
 * @return : bootstrap communicator
 */
BootstrapCommunicator *WnmmpGetBootstrapCommunicator(WholeNCCLMemory_t wnmt);

/*!
 * Get local memory ptr and size.
 * @param wnmt : WholeNCCLMemory object
 * @param ptr : return pointer
 * @param size : return size
 */
void WnmmpGetLocalMemory(WholeNCCLMemory_t wnmt, void **ptr, size_t *size);

/*!
 * return chunk size of WholeNCCLMemory
 * @param wnmt : WholeNCCLMemory object
 * @return : chunk size
 */
size_t WnmmpGetChunkSize(WholeNCCLMemory_t wnmt);

}// namespace whole_graph