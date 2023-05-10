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

#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

namespace nb = nanobind;

void init_wholegraph_types(nb::module_& m)
{
  nb::enum_<wholememory_error_code_t>(m, "WholeMemoryErrorCode")
    .value("Success", WHOLEMEMORY_SUCCESS)                         // success
    .value("UnknownError", WHOLEMEMORY_UNKNOW_ERROR)               // unknown error
    .value("NotImplemented", WHOLEMEMORY_NOT_IMPLEMENTED)          // method is not implemented
    .value("LogicError", WHOLEMEMORY_LOGIC_ERROR)                  // logic error
    .value("CudaError", WHOLEMEMORY_CUDA_ERROR)                    // CUDA error
    .value("CommunicationError", WHOLEMEMORY_COMMUNICATION_ERROR)  // communication error
    .value("InvalidInput", WHOLEMEMORY_INVALID_INPUT)              // invalid input, e.g. nullptr
    .value("InvalidValue", WHOLEMEMORY_INVALID_VALUE);             // input value is invalid

  nb::enum_<wholememory_memory_type_t>(m, "WholeMemoryType")
    .value("None", WHOLEMEMORY_MT_NONE)
    .value("Continuous", WHOLEMEMORY_MT_CONTINUOUS)
    .value("Chunked", WHOLEMEMORY_MT_CHUNKED)
    .value("Distributed", WHOLEMEMORY_MT_DISTRIBUTED);

  nb::enum_<wholememory_memory_location_t>(m, "WholeMemoryLocation")
    .value("None", WHOLEMEMORY_ML_NONE)
    .value("Device", WHOLEMEMORY_ML_DEVICE)
    .value("Host", WHOLEMEMORY_ML_HOST);

  nb::enum_<wholememory_dtype_t>(m, "WholeMemoryDataType")
    .value("Unknown", WHOLEMEMORY_DT_UNKNOWN)
    .value("Float", WHOLEMEMORY_DT_FLOAT)
    .value("Half", WHOLEMEMORY_DT_HALF)
    .value("Double", WHOLEMEMORY_DT_DOUBLE)
    .value("Bf16", WHOLEMEMORY_DT_BF16)
    .value("Int", WHOLEMEMORY_DT_INT)
    .value("Int64", WHOLEMEMORY_DT_INT64)
    .value("Int16", WHOLEMEMORY_DT_INT16)
    .value("Int8", WHOLEMEMORY_DT_INT8)
    .value("DTCount", WHOLEMEMORY_DT_COUNT);
}
