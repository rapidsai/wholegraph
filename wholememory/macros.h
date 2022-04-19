#pragma once

#include <cuda_runtime_api.h>
#include <stdio.h>

#define DivUp(X, Y) (((X) + (Y) - 1) / (Y))
#define AlignUp(X, ALIGN_SIZE) (((X) + (ALIGN_SIZE) - 1) / (ALIGN_SIZE) * (ALIGN_SIZE))

#define WM_CHECK(X)                                                                           \
do {                                                                                          \
  if (!(X)) {                                                                                 \
    fprintf(stderr, "File %s Line %d %s, CHECK failed.\n",                                    \
        __FILE__, __LINE__, #X);                                                              \
    abort();                                                                                  \
  }                                                                                           \
} while (0)

#define WM_CU_CHECK(X)                                                                        \
do {                                                                                          \
  auto result = X;                                                                            \
  if (result != CUDA_SUCCESS) {                                                               \
    const char* p_err_str = nullptr;                                                          \
    if (cuGetErrorString(result, &p_err_str) == CUDA_ERROR_INVALID_VALUE) {                   \
      p_err_str = "Unrecoginzed CU error num";                                                \
    }                                                                                         \
    fprintf(stderr, "File %s Line %d %s returned %s.\n",                                      \
        __FILE__, __LINE__, #X, p_err_str);                                                   \
    abort();                                                                                  \
  }                                                                                           \
} while (0)

#define WM_CUDA_CHECK(X)                                                                      \
do {                                                                                          \
  auto result = X;                                                                            \
  if (result != cudaSuccess) {                                                                \
    const char* p_err_str = cudaGetErrorName(result);                                         \
    fprintf(stderr, "File %s Line %d %s returned %s.\n",                                      \
        __FILE__, __LINE__, #X, p_err_str);                                                   \
    abort();                                                                                  \
  }                                                                                           \
} while (0)

