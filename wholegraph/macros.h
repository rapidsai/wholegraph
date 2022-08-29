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
#include <stdio.h>

#define DivUp(X, Y) (((X) + (Y) -1) / (Y))
#define AlignUp(X, ALIGN_SIZE) (((X) + (ALIGN_SIZE) -1) / (ALIGN_SIZE) * (ALIGN_SIZE))

#define WM_CHECK(X)                                          \
  do {                                                       \
    if (!(X)) {                                              \
      fprintf(stderr, "File %s Line %d %s, CHECK failed.\n", \
              __FILE__, __LINE__, #X);                       \
      abort();                                               \
    }                                                        \
  } while (0)

#define WM_CU_CHECK(X)                                                        \
  do {                                                                        \
    auto result = X;                                                          \
    if (result != CUDA_SUCCESS) {                                             \
      const char *p_err_str = nullptr;                                        \
      if (cuGetErrorString(result, &p_err_str) == CUDA_ERROR_INVALID_VALUE) { \
        p_err_str = "Unrecoginzed CU error num";                              \
      }                                                                       \
      fprintf(stderr, "File %s Line %d %s returned %s.\n",                    \
              __FILE__, __LINE__, #X, p_err_str);                             \
      abort();                                                                \
    }                                                                         \
  } while (0)

#define WM_CUDA_CHECK(X)                                   \
  do {                                                     \
    auto result = X;                                       \
    if (result != cudaSuccess) {                           \
      const char *p_err_str = cudaGetErrorName(result);    \
      fprintf(stderr, "File %s Line %d %s returned %s.\n", \
              __FILE__, __LINE__, #X, p_err_str);          \
      abort();                                             \
    }                                                      \
  } while (0)
