/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include <wholememory/wholememory.h>

namespace wholememory {

wholememory_error_code_t init(unsigned int flags, LogLevel log_level) noexcept;

wholememory_error_code_t finalize() noexcept;

/**
 * return cudaDeviceProp of dev_id, if dev_id is -1, use current device
 * @param dev_id : device id, -1 for current device
 * @return : cudaDeviceProp pointer
 */
cudaDeviceProp* get_device_prop(int dev_id) noexcept;

}  // namespace wholememory
