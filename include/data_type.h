#pragma once

#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <iostream>
#include <cstdint>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace whole_graph {

typedef enum : int8_t {
  WMT_Uint8 = 0,
  WMT_Int8 = 1,
  WMT_Uint16 = 2,
  WMT_Int16 = 3,
  WMT_Uint32 = 4,
  WMT_Int32 = 5,
  WMT_Uint64 = 6,
  WMT_Int64 = 7,
  WMT_Half = 8,
  WMT_Bfloat16 = 9,
  WMT_Float = 10,
  WMT_Double = 11,
  WMT_Count,
} WMType;

constexpr const char* WMTNames[] = {
    "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "half", "bfloat16", "float", "double"
};

inline const char* GetWMTName(WMType wmt) {
  if (wmt >= 0 && wmt < WMT_Count) return WMTNames[wmt];
  std::cerr << "GetWMTName type " << wmt << " not supported.\n";
  abort();
}

constexpr const size_t WMTSizes[] = {
    1, 1, 2, 2, 4, 4, 8, 8, 2, 2, 4, 8
};

inline size_t GetWMTSize(WMType wmt) {
  if (wmt >= 0 && wmt < WMT_Count) return WMTSizes[wmt];
  std::cerr << "GetWMTSize type " << wmt << " not supported.\n";
  abort();
}

template <typename T>
inline WMType GetWMType() {
  std::cerr << "GetWMType type not supported.\n";
  abort();
}

template <> inline WMType GetWMType<uint8_t>() { return WMT_Uint8; }
template <> inline WMType GetWMType<int8_t>() { return WMT_Int8; }
template <> inline WMType GetWMType<uint16_t>() { return WMT_Uint16; }
template <> inline WMType GetWMType<int16_t>() { return WMT_Int16; }
template <> inline WMType GetWMType<uint32_t>() { return WMT_Uint32; }
template <> inline WMType GetWMType<int32_t>() { return WMT_Int32; }
template <> inline WMType GetWMType<uint64_t>() { return WMT_Uint64; }
template <> inline WMType GetWMType<int64_t>() { return WMT_Int64; }
template <> inline WMType GetWMType<__half>() { return WMT_Half; }
template <> inline WMType GetWMType<__nv_bfloat16>() { return WMT_Bfloat16; }
template <> inline WMType GetWMType<float>() { return WMT_Float; }
template <> inline WMType GetWMType<double>() { return WMT_Double; }

struct OneWMTHash : public std::unary_function<WMType, std::size_t> {
  inline std::size_t operator()(const WMType& k) const {
    return (size_t)k;
  }
};

struct TwoWMTHash : public std::unary_function<std::tuple<WMType, WMType>, std::size_t> {
  inline std::size_t operator()(const std::tuple<WMType, WMType>& k) const {
    return (size_t)std::get<1>(k) * ((size_t)WMT_Count) + (size_t)std::get<0>(k);
  }
};

struct ThreeWMTHash : public std::unary_function<std::tuple<WMType, WMType, WMType>, std::size_t> {
  inline std::size_t operator()(const std::tuple<WMType, WMType, WMType> &k) const {
    return (size_t) std::get<2>(k) * ((size_t) WMT_Count) * ((size_t) WMT_Count)
        + (size_t) std::get<1>(k) * ((size_t) WMT_Count) + (size_t) std::get<0>(k);
  }
};

}

#define VEC_SINT3264 std::vector<whole_graph::WMType>({WMT_Int32, WMT_Int64})
#define VEC_SINT std::vector<whole_graph::WMType>({WMT_Int8, WMT_Int16, WMT_Int32, WMT_Int64})
#define VEC_UINT3264 std::vector<whole_graph::WMType>({WMT_Uint32, WMT_Uint64})
#define VEC_UINT std::vector<whole_graph::WMType>({WMT_Uint8, WMT_Uint16, WMT_Uint32, WMT_Uint64})
#define VEC_ALLINT3264 std::vector<whole_graph::WMType>({WMT_Int32, WMT_Int64, WMT_Uint32, WMT_Uint64})
#define VEC_ALLINT std::vector<whole_graph::WMType>({WMT_Int8, WMT_Int16, WMT_Int32, WMT_Int64, WMT_Uint8, WMT_Uint16, WMT_Uint32, WMT_Uint64})
#define VEC_FLOAT_DOUBLE std::vector<whole_graph::WMType>({WMT_Float, WMT_Double})
#define VEC_HALF_FLOAT std::vector<whole_graph::WMType>({WMT_Half, WMT_Float})
#define VEC_BF16_HALF_FLOAT std::vector<whole_graph::WMType>({WMT_Bfloat16, WMT_Half, WMT_Float})
#define VEC_HALF_FLOAT_DOUBLE std::vector<whole_graph::WMType>({WMT_Half, WMT_Half, WMT_Float})
#define VEC_ALLFLOAT std::vector<whole_graph::WMType>({WMT_Bfloat16, WMT_Half, WMT_Half, WMT_Float})

#define CASES_SINT3264(TEMPFUNC_NAME, ...) \
case WMT_Int32: {TEMPFUNC_NAME<int32_t, ##__VA_ARGS__>(); break; } \
case WMT_Int64: {TEMPFUNC_NAME<int64_t, ##__VA_ARGS__>(); break; }

#define CASES_SINT(TEMPFUNC_NAME, ...) \
case WMT_Int8: {TEMPFUNC_NAME<int8_t, ##__VA_ARGS__>(); break; } \
case WMT_Int16: {TEMPFUNC_NAME<int16_t, ##__VA_ARGS__>(); break; } \
CASES_SINT3264(TEMPFUNC_NAME, ##__VA_ARGS__)

#define CASES_UINT3264(TEMPFUNC_NAME, ...) \
case WMT_Uint32: {TEMPFUNC_NAME<uint32_t, ##__VA_ARGS__>(); break; } \
case WMT_Uint64: {TEMPFUNC_NAME<uint32_t, ##__VA_ARGS__>(); break; }

#define CASES_UINT(TEMPFUNC_NAME, ...) \
case WMT_Uint8: {TEMPFUNC_NAME<uint8_t, ##__VA_ARGS__>(); break; } \
case WMT_Uint16: {TEMPFUNC_NAME<uint16_t, ##__VA_ARGS__>(); break; } \
CASES_UINT3264(TEMPFUNC_NAME, ##__VA_ARGS__)

#define CASES_ALLINT3264(TEMPFUNC_NAME, ...) \
CASES_SINT3264(TEMPFUNC_NAME, ##__VA_ARGS__) \
CASES_UINT3264(TEMPFUNC_NAME, ##__VA_ARGS__)

#define CASES_ALLINT(TEMPFUNC_NAME, ...) \
CASES_SINT(TEMPFUNC_NAME, ##__VA_ARGS__) \
CASES_UINT(TEMPFUNC_NAME, ##__VA_ARGS__)

#define CASES_FLOAT_DOUBLE(TEMPFUNC_NAME, ...) \
case WMT_Float: {TEMPFUNC_NAME<float, ##__VA_ARGS__>(); break; } \
case WMT_Double: {TEMPFUNC_NAME<double, ##__VA_ARGS__>(); break; }

#define CASES_HALF_FLOAT(TEMPFUNC_NAME, ...) \
case WMT_Half: {TEMPFUNC_NAME<__half, ##__VA_ARGS__>(); break; } \
case WMT_Float: {TEMPFUNC_NAME<float, ##__VA_ARGS__>(); break; } \

#define CASES_BF16_HALF_FLOAT(TEMPFUNC_NAME, ...) \
case WMT_Bfloat16: {TEMPFUNC_NAME<__nv_bfloat16, ##__VA_ARGS__>(); break; } \
CASES_HALF_FLOAT(TEMPFUNC_NAME, ##__VA_ARGS__)

#define CASES_HALF_FLOAT_DOUBLE(TEMPFUNC_NAME, ...) \
case WMT_Half: {TEMPFUNC_NAME<__half, ##__VA_ARGS__>(); break; } \
CASES_FLOAT_DOUBLE(TEMPFUNC_NAME, ##__VA_ARGS__)

#define CASES_ALLFLOAT(TEMPFUNC_NAME, ...) \
case WMT_Bfloat16: {TEMPFUNC_NAME<__nv_bfloat16, ##__VA_ARGS__>(); break; } \
CASES_HALF_FLOAT_DOUBLE(TEMPFUNC_NAME, ##__VA_ARGS__)

#define REGISTER_DISPATCH_ONE_TYPE(NAME, TEMPFUNC_NAME, ARG0_SET) \
static std::unordered_map<WMType, decltype(&TEMPFUNC_NAME<int>), whole_graph::OneWMTHash>* NAME ## _dispatch1_map = nullptr; \
template <typename T0> \
void Register ## NAME ## Map1FuncHelper0() { \
    auto key = GetWMType<T0>(); \
    NAME ## _dispatch1_map->emplace(key, TEMPFUNC_NAME<T0>); \
} \
__attribute__((constructor)) static void Register ## NAME ## Map1Func() { \
  NAME ## _dispatch1_map = new std::unordered_map<WMType, decltype(&TEMPFUNC_NAME<int>), whole_graph::OneWMTHash>(); \
  auto arg0_types = VEC_ ## ARG0_SET; \
  for (auto arg0_type : arg0_types) { \
      switch (arg0_type) { \
        CASES_ ## ARG0_SET(Register ## NAME ## Map1FuncHelper0) \
        default: { abort(); break; } \
      }\
  }\
}

#define DISPATCH_ONE_TYPE(WMTypeValue0, NAME, ...) \
do { \
  auto key = WMTypeValue0; \
  auto it = NAME ## _dispatch1_map->find(key); \
  assert(it != NAME ## _dispatch1_map->end()); \
  it->second(__VA_ARGS__); \
} while (0)

#define REGISTER_DISPATCH_TWO_TYPES(NAME, TEMPFUNC_NAME, ARG0_SET, ARG1_SET) \
static std::unordered_map<std::tuple<WMType, WMType>, decltype(&TEMPFUNC_NAME<int, int>), whole_graph::TwoWMTHash>* NAME ## _dispatch2_map = nullptr; \
template <typename T0, typename T1> \
void Register ## NAME ## Map2FuncHelper0() { \
    auto key = std::make_tuple(GetWMType<T0>(), GetWMType<T1>()); \
    NAME ## _dispatch2_map->emplace(key, TEMPFUNC_NAME<T0, T1>); \
} \
template <typename T1> \
void Register ## NAME ## Map2FuncHelper1() { \
  auto arg0_types = VEC_ ## ARG0_SET; \
  for (auto arg0_type : arg0_types) { \
      switch (arg0_type) { \
        CASES_ ## ARG1_SET(Register ## NAME ## Map2FuncHelper0, T1) \
        default: { abort(); break; } \
      }\
  }\
} \
__attribute__((constructor)) static void Register ## NAME ## Map2Func() { \
  NAME ## _dispatch2_map = new std::unordered_map<std::tuple<WMType, WMType>, decltype(&TEMPFUNC_NAME<int, int>), whole_graph::TwoWMTHash>(); \
  auto arg1_types = VEC_ ## ARG1_SET; \
  for (auto arg1_type : arg1_types) { \
      switch (arg1_type) { \
        CASES_ ## ARG1_SET(Register ## NAME ## Map2FuncHelper1) \
        default: { abort(); break; } \
      }\
  }\
}

#define DISPATCH_TWO_TYPES(WMTypeValue0, WMTypeValue1, NAME, ...) \
do { \
  auto key = std::make_tuple(WMTypeValue0, WMTypeValue1); \
  auto it = NAME ## _dispatch2_map->find(key); \
  assert(it != NAME ## _dispatch2_map->end()); \
  it->second(__VA_ARGS__); \
} while (0)

#define REGISTER_DISPATCH_THREE_TYPES(NAME, TEMPFUNC_NAME, ARG0_SET, ARG1_SET, ARG2_SET) \
static std::unordered_map<std::tuple<WMType, WMType, WMType>, decltype(&TEMPFUNC_NAME<int, int, int>), whole_graph::ThreeWMTHash>* NAME ## _dispatch3_map = nullptr; \
template <typename T0, typename T1, typename T2> \
void Register ## NAME ## Map3FuncHelper0() { \
    auto key = std::make_tuple(GetWMType<T0>(), GetWMType<T1>(), GetWMType<T2>()); \
    NAME ## _dispatch3_map->emplace(key, TEMPFUNC_NAME<T0, T1, T2>); \
} \
template <typename T1, typename T2> \
void Register ## NAME ## Map3FuncHelper1() { \
  auto arg0_types = VEC_ ## ARG0_SET; \
  for (auto arg0_type : arg0_types) { \
      switch (arg0_type) { \
        CASES_ ## ARG1_SET(Register ## NAME ## Map3FuncHelper0, T1, T2) \
        default: { abort(); break; } \
      }\
  }\
} \
template <typename T2> \
void Register ## NAME ## Map3FuncHelper2() { \
  auto arg1_types = VEC_ ## ARG1_SET; \
  for (auto arg1_type : arg1_types) { \
      switch (arg1_type) { \
        CASES_ ## ARG1_SET(Register ## NAME ## Map3FuncHelper1, T2) \
        default: { abort(); break; } \
      }\
  }\
} \
__attribute__((constructor)) static void Register ## NAME ## Map3Func() { \
  NAME ## _dispatch3_map = new std::unordered_map<std::tuple<WMType, WMType, WMType>, decltype(&TEMPFUNC_NAME<int, int, int>), whole_graph::ThreeWMTHash>(); \
  auto arg2_types = VEC_ ## ARG2_SET; \
  for (auto arg2_type : arg2_types) { \
      switch (arg2_type) { \
        CASES_ ## ARG2_SET(Register ## NAME ## Map3FuncHelper2) \
        default: { abort(); break; } \
      }\
  }\
}

#define DISPATCH_THREE_TYPES(WMTypeValue0, WMTypeValue1, WMTypeValue2, NAME, ...) \
do { \
  auto key = std::make_tuple(WMTypeValue0, WMTypeValue1, WMTypeValue2); \
  auto it = NAME ## _dispatch3_map->find(key); \
  assert(it != NAME ## _dispatch3_map->end()); \
  it->second(__VA_ARGS__); \
} while (0)



