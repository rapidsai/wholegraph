/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "wholememory/wholememory.h"

#if CUDA_VERSION >= 12030
#include <nvml.h>
#endif
bool DevAttrPagebleMemoryAccess();

bool DeviceCanAccessPeer(int peer_device);

bool DevicesCanAccessP2P(const int* dev_ids, int count);

int GetCudaCompCap();

const char* GetCPUArch();

bool SupportMNNVL();

bool SupportEGM();

// bool SupportMNNVLForEGM();
#if CUDA_VERSION >= 12030
namespace wholememory {
wholememory_error_code_t GetGpuFabricInfo(int dev, nvmlGpuFabricInfo_t* gpuFabricInfo);
}

#endif
