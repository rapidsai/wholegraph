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
#include "system_info.hpp"

#include <string>

#include "cuda_macros.hpp"

#include "logger.hpp"
#include "system_info.hpp"
#include "wholememory/wholememory.h"
#if CUDA_VERSION >= 12030
#include <nvml.h>

namespace {

std::mutex lock;  // NVML has had some thread safety bugs
bool nvmlInitialized                = false;
thread_local bool threadInitialized = false;
wholememory_error_code_t initResult;
};  // namespace

#endif
bool DevAttrPagebleMemoryAccess()
{
  int current_dev_id = -1;
  WM_CUDA_CHECK_NO_THROW(cudaGetDevice(&current_dev_id));
  int value           = 0;
  cudaDeviceAttr attr = cudaDevAttrPageableMemoryAccess;
  WM_CUDA_CHECK_NO_THROW(cudaDeviceGetAttribute(&value, attr, current_dev_id));
  return value > 0;
}

bool DeviceCanAccessPeer(int peer_device)
{
  int current_dev_id = -1;
  WM_CUDA_CHECK_NO_THROW(cudaGetDevice(&current_dev_id));
  int can_access = 0;
  WM_CUDA_CHECK_NO_THROW(cudaDeviceCanAccessPeer(&can_access, current_dev_id, peer_device));
  return can_access > 0;
}

bool DevicesCanAccessP2P(const int* dev_ids, int count)
{
  if (count <= 1) return true;
  int current_dev_id = -1;
  WM_CUDA_CHECK_NO_THROW(cudaGetDevice(&current_dev_id));

  bool all_can_access = true;

  for (int i = 0; i < count && all_can_access; i++) {
    int src_dev = dev_ids[i];
    WM_CUDA_CHECK_NO_THROW(cudaSetDevice(src_dev));
    for (int j = 0; j < count; j++) {
      if (j == i) continue;
      int peer_dev   = dev_ids[j];
      int can_access = 0;
      WM_CUDA_CHECK_NO_THROW(cudaDeviceCanAccessPeer(&can_access, src_dev, peer_dev));
      if (can_access == 0) {
        all_can_access = false;
        break;
      }
    }
  }
  WM_CUDA_CHECK_NO_THROW(cudaSetDevice(current_dev_id));
  return all_can_access;
}

int GetCudaCompCap()
{
  int cuda_dev;
  WM_CUDA_CHECK_NO_THROW(cudaGetDevice(&cuda_dev));
  int cc_major, cc_minor;
  WM_CUDA_CHECK_NO_THROW(
    cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, cuda_dev));
  WM_CUDA_CHECK_NO_THROW(
    cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, cuda_dev));
  return cc_major * 10 + cc_minor;
}

const char* GetCPUArch()
{
#if defined(__PPC__)
  static const char* arch_str = "ppc64";
#elif defined(__aarch64__)
  static const char* arch_str = "arm64";
#elif defined(__x86_64__)
  static const char* arch_str = "x86_64";
#endif
  return arch_str;
}

bool SupportEGM()
{
  std::string const arch_str = GetCPUArch();
  return arch_str == "arm64" && DevAttrPagebleMemoryAccess();
}

// bool SupportMNNVLForEGM() { return SupportMNNVL() && SupportEGM(); }
#if CUDA_VERSION >= 12030

namespace wholememory {

wholememory_error_code_t NvmlEnsureInitialized()
{
  // Optimization to avoid repeatedly grabbing the lock when we only want to
  // read from the global tables.
  if (threadInitialized) return initResult;
  threadInitialized = true;

  std::lock_guard<std::mutex> locked(lock);

  if (nvmlInitialized) return initResult;
  nvmlInitialized       = true;
  nvmlReturn_t nvml_res = nvmlInit();
  if (nvml_res != NVML_SUCCESS) {
    WHOLEMEMORY_ERROR("nvmlInit() failed, the error is %s", nvmlErrorString(nvml_res));
    initResult = WHOLEMEMORY_SYSTEM_ERROR;

    return initResult;
  }
  initResult = WHOLEMEMORY_SUCCESS;

  return initResult;
}

wholememory_error_code_t GetGpuFabricInfo(int dev, nvmlGpuFabricInfo_t* gpuFabricInfo)
{
  WHOLEMEMORY_CHECK_NOTHROW(NvmlEnsureInitialized() == WHOLEMEMORY_SUCCESS);
  std::lock_guard<std::mutex> locked(lock);
  // gpuFabricInfo->version = nvmlGpuFabricInfo_v2;
  nvmlDevice_t nvml_device;
  nvmlReturn_t ret = nvmlDeviceGetHandleByIndex(dev, &nvml_device);
  WHOLEMEMORY_EXPECTS_NOTHROW(
    ret == NVML_SUCCESS, "nvmlDeviceGetHandleByIndex error:%s", nvmlErrorString(ret));
  ret = nvmlDeviceGetGpuFabricInfo(nvml_device, gpuFabricInfo);
  WHOLEMEMORY_EXPECTS_NOTHROW(
    ret == NVML_SUCCESS, "nvmlDeviceGetGpuFabricInfo error:%s", nvmlErrorString(ret));

  return WHOLEMEMORY_SUCCESS;
}

};  // namespace wholememory
#endif
