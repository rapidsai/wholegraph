/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#ifndef __NVSHMEM_TEMPLATE__
#define __NVSHMEM_TEMPLATE__

#ifdef WITH_NVSHMEM_SUPPORT
#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstdint>
#include <cuda_fp16.h>

namespace wholememory_ops {

template <typename DataType>
__inline__ __device__ __host__ DataType nvshmem_get(const DataType* source, int pe)
{
  return DataType{};
};

template <>
__inline__ __device__ __host__ int8_t nvshmem_get<int8_t>(const int8_t* source, int pe)
{
  char ret = nvshmem_char_g(reinterpret_cast<const char*>(source), pe);
  return reinterpret_cast<int8_t&>(ret);
}

template <>
__inline__ __device__ __host__ char nvshmem_get<char>(const char* source, int pe)
{
  return nvshmem_char_g(source, pe);
}

template <>
__inline__ __device__ __host__ int16_t nvshmem_get<int16_t>(const int16_t* source, int pe)
{
  return nvshmem_int16_g(source, pe);
}

template <>
__inline__ __device__ __host__ int32_t nvshmem_get<int32_t>(const int32_t* source, int pe)
{
  return nvshmem_int32_g(source, pe);
}

template <>
__inline__ __device__ __host__ int64_t nvshmem_get<int64_t>(const int64_t* source, int pe)
{
  return nvshmem_int64_g(source, pe);
}

template <>
__inline__ __device__ __host__ long long nvshmem_get<long long>(const long long* source, int pe)
{
  return nvshmem_longlong_g(source, pe);
}

template <>
__inline__ __device__ __host__ uint8_t nvshmem_get<uint8_t>(const uint8_t* source, int pe)
{
  return nvshmem_uint8_g(source, pe);
}

template <>
__inline__ __device__ __host__ uint16_t nvshmem_get<uint16_t>(const uint16_t* source, int pe)
{
  return nvshmem_uint16_g(source, pe);
}

template <>
__inline__ __device__ __host__ uint32_t nvshmem_get<uint32_t>(const uint32_t* source, int pe)
{
  return nvshmem_uint32_g(source, pe);
}

template <>
__inline__ __device__ __host__ uint64_t nvshmem_get<uint64_t>(const uint64_t* source, int pe)
{
  return nvshmem_uint64_g(source, pe);
}

template <>
__inline__ __device__ __host__ unsigned long long nvshmem_get<unsigned long long>(
  const unsigned long long* source, int pe)
{
  return nvshmem_ulonglong_g(source, pe);
}

template <>
__inline__ __device__ __host__ int2 nvshmem_get<int2>(const int2* source, int pe)
{
  int2 ret;
  ret.x = nvshmem_get<int>(reinterpret_cast<const int*>(source), pe);
  ret.y = nvshmem_get<int>(reinterpret_cast<const int*>(source) + 1, pe);
  return ret;
}

template <>
__inline__ __device__ __host__ int4 nvshmem_get<int4>(const int4* source, int pe)
{
  int4 ret;
  ret.x = nvshmem_get<int>(reinterpret_cast<const int*>(source), pe);
  ret.y = nvshmem_get<int>(reinterpret_cast<const int*>(source) + 1, pe);
  ret.z = nvshmem_get<int>(reinterpret_cast<const int*>(source) + 2, pe);
  ret.w = nvshmem_get<int>(reinterpret_cast<const int*>(source) + 3, pe);

  return ret;
}

template <>
__inline__ __device__ __host__ half nvshmem_get<half>(const half* source, int pe)
{
  int16_t ret = nvshmem_int16_g(reinterpret_cast<const int16_t*>(source), pe);
  return reinterpret_cast<half&>(ret);
}

template <>
__inline__ __device__ __host__ __nv_bfloat16 nvshmem_get<__nv_bfloat16>(const __nv_bfloat16* source,
                                                                        int pe)
{
  int16_t ret = nvshmem_int16_g(reinterpret_cast<const int16_t*>(source), pe);
  return reinterpret_cast<__nv_bfloat16&>(ret);
}

template <>
__inline__ __device__ __host__ float nvshmem_get<float>(const float* source, int pe)
{
  return nvshmem_float_g(source, pe);
}

template <>
__inline__ __device__ __host__ double nvshmem_get<double>(const double* source, int pe)
{
  return nvshmem_double_g(source, pe);
}

template <>
__inline__ __device__ __host__ float2 nvshmem_get<float2>(const float2* source, int pe)
{
  float2 ret;
  ret.x = nvshmem_get<float>(reinterpret_cast<const float*>(source), pe);
  ret.y = nvshmem_get<float>(reinterpret_cast<const float*>(source) + 1, pe);
  return ret;
}

template <>
__inline__ __device__ __host__ float4 nvshmem_get<float4>(const float4* source, int pe)
{
  float4 ret;
  ret.x = nvshmem_get<float>(reinterpret_cast<const float*>(source), pe);
  ret.y = nvshmem_get<float>(reinterpret_cast<const float*>(source) + 1, pe);
  ret.z = nvshmem_get<float>(reinterpret_cast<const float*>(source) + 2, pe);
  ret.w = nvshmem_get<float>(reinterpret_cast<const float*>(source) + 3, pe);
  return ret;
};

template <typename DataType>
__inline__ __device__ __host__ void nvshmem_put(DataType* dest, const DataType val, int pe)
{
  return;
};

template <>
__inline__ __device__ __host__ void nvshmem_put<int8_t>(int8_t* dest, const int8_t val, int pe)
{
  const char& val_char = reinterpret_cast<const char&>(val);
  return nvshmem_char_p(reinterpret_cast<char*>(dest), val_char, pe);
}

template <>
__inline__ __device__ __host__ void nvshmem_put<char>(char* dest, const char val, int pe)
{
  return nvshmem_char_p(dest, val, pe);
}

template <>
__inline__ __device__ __host__ void nvshmem_put<int16_t>(int16_t* dest, const int16_t val, int pe)
{
  return nvshmem_int16_p(dest, val, pe);
}

template <>
__inline__ __device__ __host__ void nvshmem_put<int32_t>(int32_t* dest, const int32_t val, int pe)
{
  return nvshmem_int32_p(dest, val, pe);
}

template <>
__inline__ __device__ __host__ void nvshmem_put<int64_t>(int64_t* dest, const int64_t val, int pe)
{
  return nvshmem_int64_p(dest, val, pe);
}

template <>
__inline__ __device__ __host__ void nvshmem_put<long long>(long long* dest,
                                                           const long long val,
                                                           int pe)
{
  return nvshmem_longlong_p(dest, val, pe);
}

template <>
__inline__ __device__ __host__ void nvshmem_put<uint8_t>(uint8_t* dest, const uint8_t val, int pe)
{
  return nvshmem_uint8_p(dest, val, pe);
}
template <>
__inline__ __device__ __host__ void nvshmem_put<uint16_t>(uint16_t* dest,
                                                          const uint16_t val,
                                                          int pe)
{
  return nvshmem_uint16_p(dest, val, pe);
}

template <>
__inline__ __device__ __host__ void nvshmem_put<uint32_t>(uint32_t* dest,
                                                          const uint32_t val,
                                                          int pe)
{
  return nvshmem_uint32_p(dest, val, pe);
}

template <>
__inline__ __device__ __host__ void nvshmem_put<uint64_t>(uint64_t* dest,
                                                          const uint64_t val,
                                                          int pe)
{
  return nvshmem_uint64_p(dest, val, pe);
}

template <>
__inline__ __device__ __host__ void nvshmem_put<unsigned long long>(unsigned long long* dest,
                                                                    const unsigned long long val,
                                                                    int pe)
{
  return nvshmem_ulonglong_p(dest, val, pe);
}

template <>
__inline__ __device__ __host__ void nvshmem_put<int2>(int2* dest, const int2 val, int pe)
{
  int* dest_int = reinterpret_cast<int*>(dest);
  nvshmem_int_p(dest_int, val.x, pe);
  nvshmem_int_p(dest_int + 1, val.y, pe);
}

template <>
__inline__ __device__ __host__ void nvshmem_put<int4>(int4* dest, const int4 val, int pe)
{
  int* dest_int = reinterpret_cast<int*>(dest);
  nvshmem_int_p(dest_int, val.x, pe);
  nvshmem_int_p(dest_int + 1, val.y, pe);
  nvshmem_int_p(dest_int + 2, val.z, pe);
  nvshmem_int_p(dest_int + 3, val.w, pe);
}

template <>
__inline__ __device__ __host__ void nvshmem_put<half>(half* dest, const half val, int pe)
{
  const int16_t& val_int = reinterpret_cast<const int16_t&>(val);
  return nvshmem_int16_p(reinterpret_cast<int16_t*>(dest), val_int, pe);
}

template <>
__inline__ __device__ __host__ void nvshmem_put<__nv_bfloat16>(__nv_bfloat16* dest,
                                                               const __nv_bfloat16 val,
                                                               int pe)
{
  const int16_t& val_int = reinterpret_cast<const int16_t&>(val);
  return nvshmem_int16_p(reinterpret_cast<int16_t*>(dest), val_int, pe);
}

template <>
__inline__ __device__ __host__ void nvshmem_put<float>(float* dest, const float val, int pe)
{
  return nvshmem_float_p(dest, val, pe);
}

template <>
__inline__ __device__ __host__ void nvshmem_put<double>(double* dest, const double val, int pe)
{
  return nvshmem_double_p(dest, val, pe);
}

template <>
__inline__ __device__ __host__ void nvshmem_put<float2>(float2* dest, const float2 val, int pe)
{
  float* dest_float = reinterpret_cast<float*>(dest);
  nvshmem_float_p(dest_float, val.x, pe);
  nvshmem_float_p(dest_float + 1, val.y, pe);
}

template <>
__inline__ __device__ __host__ void nvshmem_put<float4>(float4* dest, const float4 val, int pe)
{
  float* dest_float = reinterpret_cast<float*>(dest);
  nvshmem_float_p(dest_float, val.x, pe);
  nvshmem_float_p(dest_float + 1, val.y, pe);
  nvshmem_float_p(dest_float + 2, val.z, pe);
  nvshmem_float_p(dest_float + 3, val.w, pe);
}

}  // namespace wholememory_ops

#endif  // WITH_NVSHMEM_SUPPORT

#endif  // __NVSHMEM_TEMPLATE__
