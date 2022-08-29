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

#include <cuda.h>
#include <stdint.h>

namespace whole_graph {

typedef int64_t TypedNodeID;

__device__ __host__ __forceinline__ TypedNodeID MakeTypedID(int type, int64_t id) {
  TypedNodeID typed_id = (int64_t) type << 56;
  return typed_id | id;
}

__device__ __host__ __forceinline__ int TypeOfTypedID(TypedNodeID typed_id) {
  return typed_id >> 56;
}

__device__ __host__ __forceinline__ int64_t IDOfTypedID(TypedNodeID typed_id) {
  return typed_id & ((1LL << 56LL) - 1LL);
}

}// namespace whole_graph