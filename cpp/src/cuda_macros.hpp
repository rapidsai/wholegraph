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

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <execinfo.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>

#include <raft/core/error.hpp>

#include "error.hpp"

namespace wholememory {

/**
 * @brief Exception thrown when a CUDA error is encountered.
 */
struct cuda_error : public raft::exception {
  explicit cuda_error(char const* const message) : raft::exception(message) {}
  explicit cuda_error(std::string const& message) : raft::exception(message) {}
};

/**
 * @brief Exception thrown when a CUDA driver error is encountered.
 */
struct cu_error : public raft::exception {
  explicit cu_error(char const* const message) : raft::exception(message) {}
  explicit cu_error(std::string const& message) : raft::exception(message) {}
};

}  // namespace wholememory

/**
 * @brief Error checking macro for CUDA runtime API functions.
 *
 * Invokes a CUDA runtime API function call, if the call does not return
 * cudaSuccess, invokes cudaGetLastError() to clear the error and throws an
 * exception detailing the CUDA error that occurred
 *
 */
#define WM_CUDA_TRY(call)                                      \
  do {                                                         \
    cudaError_t const status = call;                           \
    if (status != cudaSuccess) {                               \
      cudaGetLastError();                                      \
      std::string msg{};                                       \
      SET_WHOLEMEMORY_ERROR_MSG(msg,                           \
                                "CUDA error encountered at: ", \
                                "call='%s', Reason=%s:%s",     \
                                #call,                         \
                                cudaGetErrorName(status),      \
                                cudaGetErrorString(status));   \
      throw wholememory::cuda_error(msg);                      \
    }                                                          \
  } while (0)

#ifndef WM_CUDA_CHECK
#define WM_CUDA_CHECK(call) WM_CUDA_TRY(call)
#endif

// /**
//  * @brief check for cuda runtime API errors but log error instead of raising
//  *        exception.
//  */
#define WM_CUDA_TRY_NO_THROW(call)                                 \
  do {                                                             \
    cudaError_t const status = call;                               \
    if (cudaSuccess != status) {                                   \
      printf("CUDA call='%s' at file=%s line=%d failed with %s\n", \
             #call,                                                \
             __FILE__,                                             \
             __LINE__,                                             \
             cudaGetErrorString(status));                          \
      abort();                                                     \
    }                                                              \
  } while (0)

#ifndef WM_CUDA_CHECK_NO_THROW
#define WM_CUDA_CHECK_NO_THROW(call) WM_CUDA_TRY_NO_THROW(call)
#endif

/**
 * @brief Error checking macro for CUDA driver API functions.
 *
 * Invokes a CUDA driver API function call, if the call does not return
 * CUDA_SUCCESS, invokes cuGetErrorString() to clear the error and throws an
 * exception detailing the CU error that occurred
 *
 */
#define WM_CU_TRY(call)                                                       \
  do {                                                                        \
    CUresult const status = call;                                             \
    if (status != CUDA_SUCCESS) {                                             \
      const char* p_err_name = nullptr;                                       \
      cuGetErrorName(status, &p_err_name);                                    \
      const char* p_err_str = nullptr;                                        \
      if (cuGetErrorString(status, &p_err_str) == CUDA_ERROR_INVALID_VALUE) { \
        p_err_str = "Unrecoginzed CU error num";                              \
      }                                                                       \
      std::string msg{};                                                      \
      SET_WHOLEMEMORY_ERROR_MSG(msg,                                          \
                                "CU error encountered at: ",                  \
                                "call='%s', Reason=%s:%s",                    \
                                #call,                                        \
                                p_err_name,                                   \
                                p_err_str);                                   \
      throw wholememory::cu_error(msg);                                       \
    }                                                                         \
  } while (0)

#ifndef WM_CU_CHECK
#define WM_CU_CHECK(call) WM_CU_TRY(call)
#endif

// /**
//  * @brief check for cuda driver API errors but log error instead of raising
//  *        exception.
//  */
#define WM_CU_TRY_NO_THROW(call)                                                                   \
  do {                                                                                             \
    CUresult const status = call;                                                                  \
    if (status != CUDA_SUCCESS) {                                                                  \
      const char* p_err_str = nullptr;                                                             \
      if (cuGetErrorString(status, &p_err_str) == CUDA_ERROR_INVALID_VALUE) {                      \
        p_err_str = "Unrecoginzed CU error num";                                                   \
      }                                                                                            \
      std::string msg{};                                                                           \
      printf(                                                                                      \
        "CU call='%s' at file=%s line=%d failed with %s\n", #call, __FILE__, __LINE__, p_err_str); \
    }                                                                                              \
  } while (0)

#ifndef WM_CU_CHECK_NO_THROW
#define WM_CU_CHECK_NO_THROW(call) WM_CU_TRY_NO_THROW(call)
#endif

void set_debug_sync_mode(bool debug_sync_mode);

namespace wholememory {
void debug_synchronize(const char* filename, int line, cudaStream_t stream);
}

#define WM_CUDA_DEBUG_SYNC_STREAM(S) wholememory::debug_synchronize(__FILE__, __LINE__, (S))
