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
#include "thrust_allocator.hpp"

#include "error.hpp"
#include "wholememory/integer_utils.hpp"

namespace wholememory_ops {

wm_thrust_allocator::~wm_thrust_allocator() { deallocate_all(); }

wm_thrust_allocator::value_type* wm_thrust_allocator::allocate(std::ptrdiff_t mem_size)
{
  static const std::ptrdiff_t kThrustAlignSize = 256;
  mem_size = std::max<std::ptrdiff_t>(kThrustAlignSize, mem_size);
  mem_size = wholememory::div_rounding_up_unsafe(mem_size, kThrustAlignSize) * kThrustAlignSize;
  void* memory_context = nullptr;
  fns->temporary_fns.create_memory_context_fn(&memory_context, fns->temporary_fns.global_context);
  wholememory_tensor_description_t tensor_description;
  wholememory_initialize_tensor_desc(&tensor_description);
  tensor_description.dim      = 1;
  tensor_description.dtype    = WHOLEMEMORY_DT_INT64;
  tensor_description.sizes[0] = mem_size / sizeof(int64_t);
  auto* ptr                   = static_cast<value_type*>(fns->temporary_fns.malloc_fn(
    &tensor_description, WHOLEMEMORY_MA_DEVICE, memory_context, fns->temporary_fns.global_context));
  mem_ptr_to_context_map.emplace(ptr, memory_context);
  return ptr;
}

void wm_thrust_allocator::deallocate(value_type* p, size_t /*mem_size*/)
{
  auto it = mem_ptr_to_context_map.find(p);
  WHOLEMEMORY_CHECK_NOTHROW(it != mem_ptr_to_context_map.end());
  fns->temporary_fns.free_fn(it->second, fns->temporary_fns.global_context);
  fns->temporary_fns.destroy_memory_context_fn(it->second, fns->temporary_fns.global_context);
  mem_ptr_to_context_map.erase(p);
}

void wm_thrust_allocator::deallocate_all()
{
  while (!mem_ptr_to_context_map.empty()) {
    auto it = mem_ptr_to_context_map.begin();
    fns->temporary_fns.free_fn(it->second, fns->temporary_fns.global_context);
    fns->temporary_fns.destroy_memory_context_fn(it->second, fns->temporary_fns.global_context);
    mem_ptr_to_context_map.erase(it->first);
  }
}

}  // namespace wholememory_ops
