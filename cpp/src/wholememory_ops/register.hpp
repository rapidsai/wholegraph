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

#include <unordered_map>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <wholememory/tensor_description.h>

#include "error.hpp"

namespace wholememory_ops {

struct one_wmt_hash : public std::unary_function<wholememory_dtype_t, std::size_t> {
  inline std::size_t operator()(const wholememory_dtype_t& k) const
  {
    return static_cast<size_t>(k);
  }
};

struct two_wmt_hash
  : public std::unary_function<std::tuple<wholememory_dtype_t, wholememory_dtype_t>, std::size_t> {
  inline std::size_t operator()(const std::tuple<wholememory_dtype_t, wholememory_dtype_t>& k) const
  {
    return static_cast<size_t>(std::get<1>(k)) * (static_cast<size_t>(WHOLEMEMORY_DT_COUNT)) +
           static_cast<size_t>(std::get<0>(k));
  }
};

struct three_wmt_hash : public std::unary_function<
                          std::tuple<wholememory_dtype_t, wholememory_dtype_t, wholememory_dtype_t>,
                          std::size_t> {
  inline std::size_t operator()(
    const std::tuple<wholememory_dtype_t, wholememory_dtype_t, wholememory_dtype_t>& k) const
  {
    return static_cast<size_t>(std::get<2>(k)) * (static_cast<size_t>(WHOLEMEMORY_DT_COUNT)) *
             (static_cast<size_t>(WHOLEMEMORY_DT_COUNT)) +
           static_cast<size_t>(std::get<1>(k)) * (static_cast<size_t>(WHOLEMEMORY_DT_COUNT)) +
           static_cast<size_t>(std::get<0>(k));
  }
};

}  // namespace wholememory_ops

template <typename DataTypeT>
inline wholememory_dtype_t get_wholememory_dtype()
{
  WHOLEMEMORY_FAIL_NOTHROW("get_wholememory_dtype type not supported.");
  return WHOLEMEMORY_DT_UNKNOWN;
}

template <>
inline wholememory_dtype_t get_wholememory_dtype<int8_t>()
{
  return WHOLEMEMORY_DT_INT8;
}
template <>
inline wholememory_dtype_t get_wholememory_dtype<int16_t>()
{
  return WHOLEMEMORY_DT_INT16;
}
template <>
inline wholememory_dtype_t get_wholememory_dtype<int32_t>()
{
  return WHOLEMEMORY_DT_INT;
}
template <>
inline wholememory_dtype_t get_wholememory_dtype<int64_t>()
{
  return WHOLEMEMORY_DT_INT64;
}
template <>
inline wholememory_dtype_t get_wholememory_dtype<__half>()
{
  return WHOLEMEMORY_DT_HALF;
}
template <>
inline wholememory_dtype_t get_wholememory_dtype<__nv_bfloat16>()
{
  return WHOLEMEMORY_DT_BF16;
}
template <>
inline wholememory_dtype_t get_wholememory_dtype<float>()
{
  return WHOLEMEMORY_DT_FLOAT;
}
template <>
inline wholememory_dtype_t get_wholememory_dtype<double>()
{
  return WHOLEMEMORY_DT_DOUBLE;
}

#define VEC_SINT3264 std::vector<wholememory_dtype_t>({WHOLEMEMORY_DT_INT, WHOLEMEMORY_DT_INT64})
#define VEC_ALLSINT                 \
  std::vector<wholememory_dtype_t>( \
    {WHOLEMEMORY_DT_INT8, WHOLEMEMORY_DT_INT16, WHOLEMEMORY_DT_INT, WHOLEMEMORY_DT_INT64})

#define VEC_FLOAT_DOUBLE \
  std::vector<wholememory_dtype_t>({WHOLEMEMORY_DT_FLOAT, WHOLEMEMORY_DT_DOUBLE})
#define VEC_HALF_FLOAT std::vector<wholememory_dtype_t>({WHOLEMEMORY_DT_HALF, WHOLEMEMORY_DT_FLOAT})
#define VEC_BF16_HALF_FLOAT \
  std::vector<wholememory_dtype_t>({WHOLEMEMORY_DT_BF16, WHOLEMEMORY_DT_HALF, WHOLEMEMORY_DT_FLOAT})
#define VEC_HALF_FLOAT_DOUBLE       \
  std::vector<wholememory_dtype_t>( \
    {WHOLEMEMORY_DT_HALF, WHOLEMEMORY_DT_FLOAT, WHOLEMEMORY_DT_DOUBLE})
#define VEC_ALLFLOAT                \
  std::vector<wholememory_dtype_t>( \
    {WHOLEMEMORY_DT_BF16, WHOLEMEMORY_DT_HALF, WHOLEMEMORY_DT_FLOAT, WHOLEMEMORY_DT_DOUBLE})
#define VEC_ALLSINT_ALLFLOAT                              \
  std::vector<wholememory_dtype_t>({WHOLEMEMORY_DT_INT8,  \
                                    WHOLEMEMORY_DT_INT16, \
                                    WHOLEMEMORY_DT_INT,   \
                                    WHOLEMEMORY_DT_INT64, \
                                    WHOLEMEMORY_DT_BF16,  \
                                    WHOLEMEMORY_DT_HALF,  \
                                    WHOLEMEMORY_DT_FLOAT, \
                                    WHOLEMEMORY_DT_DOUBLE})

#define CASES_SINT3264(TEMPFUNC_NAME, ...)   \
  case WHOLEMEMORY_DT_INT: {                 \
    TEMPFUNC_NAME<int32_t, ##__VA_ARGS__>(); \
    break;                                   \
  }                                          \
  case WHOLEMEMORY_DT_INT64: {               \
    TEMPFUNC_NAME<int64_t, ##__VA_ARGS__>(); \
    break;                                   \
  }

#define CASES_ALLSINT(TEMPFUNC_NAME, ...)    \
  case WHOLEMEMORY_DT_INT8: {                \
    TEMPFUNC_NAME<int8_t, ##__VA_ARGS__>();  \
    break;                                   \
  }                                          \
  case WHOLEMEMORY_DT_INT16: {               \
    TEMPFUNC_NAME<int16_t, ##__VA_ARGS__>(); \
    break;                                   \
  }                                          \
    CASES_SINT3264(TEMPFUNC_NAME, ##__VA_ARGS__)

#define CASES_FLOAT_DOUBLE(TEMPFUNC_NAME, ...) \
  case WHOLEMEMORY_DT_FLOAT: {                 \
    TEMPFUNC_NAME<float, ##__VA_ARGS__>();     \
    break;                                     \
  }                                            \
  case WHOLEMEMORY_DT_DOUBLE: {                \
    TEMPFUNC_NAME<double, ##__VA_ARGS__>();    \
    break;                                     \
  }

#define CASES_HALF_FLOAT(TEMPFUNC_NAME, ...) \
  case WHOLEMEMORY_DT_HALF: {                \
    TEMPFUNC_NAME<__half, ##__VA_ARGS__>();  \
    break;                                   \
  }                                          \
  case WHOLEMEMORY_DT_FLOAT: {               \
    TEMPFUNC_NAME<float, ##__VA_ARGS__>();   \
    break;                                   \
  }

#define CASES_BF16_HALF_FLOAT(TEMPFUNC_NAME, ...)  \
  case WHOLEMEMORY_DT_BF16: {                      \
    TEMPFUNC_NAME<__nv_bfloat16, ##__VA_ARGS__>(); \
    break;                                         \
  }                                                \
    CASES_HALF_FLOAT(TEMPFUNC_NAME, ##__VA_ARGS__)

#define CASES_HALF_FLOAT_DOUBLE(TEMPFUNC_NAME, ...) \
  case WHOLEMEMORY_DT_HALF: {                       \
    TEMPFUNC_NAME<__half, ##__VA_ARGS__>();         \
    break;                                          \
  }                                                 \
    CASES_FLOAT_DOUBLE(TEMPFUNC_NAME, ##__VA_ARGS__)

#define CASES_ALLFLOAT(TEMPFUNC_NAME, ...)         \
  case WHOLEMEMORY_DT_BF16: {                      \
    TEMPFUNC_NAME<__nv_bfloat16, ##__VA_ARGS__>(); \
    break;                                         \
  }                                                \
    CASES_HALF_FLOAT_DOUBLE(TEMPFUNC_NAME, ##__VA_ARGS__)

#define CASES_ALLSINT_ALLFLOAT(TEMPFUNC_NAME, ...) \
  CASES_ALLSINT(TEMPFUNC_NAME, ##__VA_ARGS__)      \
  CASES_ALLFLOAT(TEMPFUNC_NAME, ##__VA_ARGS__)

#define REGISTER_DISPATCH_ONE_TYPE(NAME, TEMPFUNC_NAME, ARG0_SET)                           \
  static std::unordered_map<wholememory_dtype_t,                                            \
                            decltype(&TEMPFUNC_NAME<int>),                                  \
                            wholememory_ops::one_wmt_hash>* NAME##_dispatch1_map = nullptr; \
  template <typename T0>                                                                    \
  void Register##NAME##Map1FuncHelper0()                                                    \
  {                                                                                         \
    auto key = get_wholememory_dtype<T0>();                                                 \
    NAME##_dispatch1_map->emplace(key, TEMPFUNC_NAME<T0>);                                  \
  }                                                                                         \
  __attribute__((constructor)) static void Register##NAME##Map1Func()                       \
  {                                                                                         \
    NAME##_dispatch1_map = new std::unordered_map<wholememory_dtype_t,                      \
                                                  decltype(&TEMPFUNC_NAME<int>),            \
                                                  wholememory_ops::one_wmt_hash>();         \
    auto arg0_types      = VEC_##ARG0_SET;                                                  \
    for (auto arg0_type : arg0_types) {                                                     \
      switch (arg0_type) {                                                                  \
        CASES_##ARG0_SET(Register##NAME##Map1FuncHelper0) default:                          \
        {                                                                                   \
          WHOLEMEMORY_FAIL_NOTHROW("dispatch with type=%d for function %s failed.",         \
                                   static_cast<int>(arg0_type),                             \
                                   #TEMPFUNC_NAME);                                         \
          break;                                                                            \
        }                                                                                   \
      }                                                                                     \
    }                                                                                       \
  }

#define DISPATCH_ONE_TYPE(WMTypeValue0, NAME, ...)                \
  do {                                                            \
    auto key = WMTypeValue0;                                      \
    auto it  = NAME##_dispatch1_map->find(key);                   \
    WHOLEMEMORY_CHECK_NOTHROW(it != NAME##_dispatch1_map->end()); \
    it->second(__VA_ARGS__);                                      \
  } while (0)

#define REGISTER_DISPATCH_TWO_TYPES(NAME, TEMPFUNC_NAME, ARG0_SET, ARG1_SET)                \
  static std::unordered_map<std::tuple<wholememory_dtype_t, wholememory_dtype_t>,           \
                            decltype(&TEMPFUNC_NAME<int, int>),                             \
                            wholememory_ops::two_wmt_hash>* NAME##_dispatch2_map = nullptr; \
  template <typename T0, typename T1>                                                       \
  void Register##NAME##Map2FuncHelper0()                                                    \
  {                                                                                         \
    auto key = std::make_tuple(get_wholememory_dtype<T0>(), get_wholememory_dtype<T1>());   \
    NAME##_dispatch2_map->emplace(key, TEMPFUNC_NAME<T0, T1>);                              \
  }                                                                                         \
  template <typename T1>                                                                    \
  void Register##NAME##Map2FuncHelper1()                                                    \
  {                                                                                         \
    auto arg0_types = VEC_##ARG0_SET;                                                       \
    for (auto arg0_type : arg0_types) {                                                     \
      switch (arg0_type) {                                                                  \
        CASES_##ARG0_SET(Register##NAME##Map2FuncHelper0, T1) default:                      \
        {                                                                                   \
          WHOLEMEMORY_FAIL_NOTHROW("dispatch with type0=%d for function %s failed.",        \
                                   static_cast<int>(arg0_type),                             \
                                   #TEMPFUNC_NAME);                                         \
          break;                                                                            \
        }                                                                                   \
      }                                                                                     \
    }                                                                                       \
  }                                                                                         \
  __attribute__((constructor)) static void Register##NAME##Map2Func()                       \
  {                                                                                         \
    NAME##_dispatch2_map =                                                                  \
      new std::unordered_map<std::tuple<wholememory_dtype_t, wholememory_dtype_t>,          \
                             decltype(&TEMPFUNC_NAME<int, int>),                            \
                             wholememory_ops::two_wmt_hash>();                              \
    auto arg1_types = VEC_##ARG1_SET;                                                       \
    for (auto arg1_type : arg1_types) {                                                     \
      switch (arg1_type) {                                                                  \
        CASES_##ARG1_SET(Register##NAME##Map2FuncHelper1) default:                          \
        {                                                                                   \
          WHOLEMEMORY_FAIL_NOTHROW("dispatch with type1=%d for function %s failed.",        \
                                   static_cast<int>(arg1_type),                             \
                                   #TEMPFUNC_NAME);                                         \
          break;                                                                            \
        }                                                                                   \
      }                                                                                     \
    }                                                                                       \
  }

#define DISPATCH_TWO_TYPES(WMTypeValue0, WMTypeValue1, NAME, ...) \
  do {                                                            \
    auto key = std::make_tuple(WMTypeValue0, WMTypeValue1);       \
    auto it  = NAME##_dispatch2_map->find(key);                   \
    WHOLEMEMORY_CHECK_NOTHROW(it != NAME##_dispatch2_map->end()); \
    it->second(__VA_ARGS__);                                      \
  } while (0)

#define REGISTER_DISPATCH_THREE_TYPES(NAME, TEMPFUNC_NAME, ARG0_SET, ARG1_SET, ARG2_SET)      \
  static std::unordered_map<                                                                  \
    std::tuple<wholememory_dtype_t, wholememory_dtype_t, wholememory_dtype_t>,                \
    decltype(&TEMPFUNC_NAME<int, int, int>),                                                  \
    wholememory_ops::three_wmt_hash>* NAME##_dispatch3_map = nullptr;                         \
  template <typename T0, typename T1, typename T2>                                            \
  void Register##NAME##Map3FuncHelper0()                                                      \
  {                                                                                           \
    auto key = std::make_tuple(                                                               \
      get_wholememory_dtype<T0>(), get_wholememory_dtype<T1>(), get_wholememory_dtype<T2>()); \
    NAME##_dispatch3_map->emplace(key, TEMPFUNC_NAME<T0, T1, T2>);                            \
  }                                                                                           \
  template <typename T1, typename T2>                                                         \
  void Register##NAME##Map3FuncHelper1()                                                      \
  {                                                                                           \
    auto arg0_types = VEC_##ARG0_SET;                                                         \
    for (auto arg0_type : arg0_types) {                                                       \
      switch (arg0_type) {                                                                    \
        CASES_##ARG0_SET(Register##NAME##Map3FuncHelper0, T1, T2) default:                    \
        {                                                                                     \
          WHOLEMEMORY_FAIL_NOTHROW("dispatch with type0=%d for function %s failed.",          \
                                   static_cast<int>(arg0_type),                               \
                                   #TEMPFUNC_NAME);                                           \
          break;                                                                              \
        }                                                                                     \
      }                                                                                       \
    }                                                                                         \
  }                                                                                           \
  template <typename T2>                                                                      \
  void Register##NAME##Map3FuncHelper2()                                                      \
  {                                                                                           \
    auto arg1_types = VEC_##ARG1_SET;                                                         \
    for (auto arg1_type : arg1_types) {                                                       \
      switch (arg1_type) {                                                                    \
        CASES_##ARG1_SET(Register##NAME##Map3FuncHelper1, T2) default:                        \
        {                                                                                     \
          WHOLEMEMORY_FAIL_NOTHROW("dispatch with type1=%d for function %s failed.",          \
                                   static_cast<int>(arg1_type),                               \
                                   #TEMPFUNC_NAME);                                           \
          break;                                                                              \
        }                                                                                     \
      }                                                                                       \
    }                                                                                         \
  }                                                                                           \
  __attribute__((constructor)) static void Register##NAME##Map3Func()                         \
  {                                                                                           \
    NAME##_dispatch3_map = new std::unordered_map<                                            \
      std::tuple<wholememory_dtype_t, wholememory_dtype_t, wholememory_dtype_t>,              \
      decltype(&TEMPFUNC_NAME<int, int, int>),                                                \
      wholememory_ops::three_wmt_hash>();                                                     \
    auto arg2_types = VEC_##ARG2_SET;                                                         \
    for (auto arg2_type : arg2_types) {                                                       \
      switch (arg2_type) {                                                                    \
        CASES_##ARG2_SET(Register##NAME##Map3FuncHelper2) default:                            \
        {                                                                                     \
          WHOLEMEMORY_FAIL_NOTHROW("dispatch with type2=%d for function %s failed.",          \
                                   static_cast<int>(arg2_type),                               \
                                   #TEMPFUNC_NAME);                                           \
          break;                                                                              \
        }                                                                                     \
      }                                                                                       \
    }                                                                                         \
  }

#define DISPATCH_THREE_TYPES(WMTypeValue0, WMTypeValue1, WMTypeValue2, NAME, ...) \
  do {                                                                            \
    auto key = std::make_tuple(WMTypeValue0, WMTypeValue1, WMTypeValue2);         \
    auto it  = NAME##_dispatch3_map->find(key);                                   \
    WHOLEMEMORY_CHECK_NOTHROW(it != NAME##_dispatch3_map->end());                 \
    it->second(__VA_ARGS__);                                                      \
  } while (0)
