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
#include <nanobind/tensor.h>

#include <wholememory/wholememory.h>

namespace nb = nanobind;

namespace wholegraph::binding {

void init_binding(unsigned int flag) { wholememory_init(flag); }

void finalize_binding() { wholememory_finalize(); }

}  // namespace wholegraph::binding

void init_wholegraph_functions(nb::module_& m)
{
  nb::class_<wholememory_unique_id_t>(m, "WholeMemoryUniqueID")
    .def(nb::init<>())
    .def("numpy_array", [](wholememory_unique_id_t* unique_id) {
      size_t shape[1] = {sizeof(wholememory_unique_id_t::internal)};
      return nb::tensor<char, nb::shape<sizeof(wholememory_unique_id_t::internal)>>(
        &unique_id->internal[0], 1, shape);
    });
  m.def("wholememory_init", &wholegraph::binding::init_binding, nb::arg("flag"), nb::raw_doc(""));
  m.def("wholememory_finalize", &wholegraph::binding::finalize_binding, nb::raw_doc(""));
}
