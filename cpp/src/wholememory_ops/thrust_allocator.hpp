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

#include <map>

#include <wholememory/env_func_ptrs.h>

namespace wholememory_ops {

class wm_thrust_allocator {
 public:
  using value_type = char;
  explicit wm_thrust_allocator(wholememory_env_func_t* fns) : fns(fns) {}
  wm_thrust_allocator() = delete;
  ~wm_thrust_allocator();

  value_type* allocate(std::ptrdiff_t mem_size);
  void deallocate(value_type* p, size_t mem_size);
  void deallocate_all();

  wholememory_env_func_t* fns;
  std::map<value_type*, void*> mem_ptr_to_context_map;
};

}  // namespace wholememory_ops
