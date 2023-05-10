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

#include <nanobind/nanobind.h>

namespace nb = nanobind;

void init_wholegraph_types(nb::module_&);
void init_wholegraph_functions(nb::module_&);

NB_MODULE(pylibwholegraph_ext, m)
{
  // we want to first initialize global types and the graph classes so that
  // docstrings in later bindings can use the named types rather than the name
  // of the C++ type
  init_wholegraph_types(m);
  init_wholegraph_functions(m);
}
