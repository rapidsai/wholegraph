# Copyright (c) 2019-2024, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

cimport cpython
from libc cimport stdlib
from libc.stdio cimport printf, fprintf, stdout, stderr, fflush
import functools
import cython
from libc.stdint cimport *
from libcpp.cast cimport *
from libcpp cimport bool
from cpython cimport Py_buffer
from cpython cimport array
import array
from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF
from cpython.object cimport Py_TYPE, PyObject_CallObject
from cpython.tuple cimport *
from cpython.long cimport PyLong_AsLongLong


cdef extern from "Python.h":
    void Py_INCREF(PyObject *o)
    void Py_DECREF(PyObject *o)

    const char * PyUnicode_AsUTF8(object unicode)

    PyObject * PyUnicode_FromString(const char * u)


cdef extern from "wholememory/wholememory.h":
    ctypedef enum wholememory_error_code_t:
        WHOLEMEMORY_SUCCESS                 "WHOLEMEMORY_SUCCESS"  # success
        WHOLEMEMORY_UNKNOW_ERROR            "WHOLEMEMORY_UNKNOW_ERROR"  # unknown error
        WHOLEMEMORY_NOT_IMPLEMENTED         "WHOLEMEMORY_NOT_IMPLEMENTED"  # method is not implemented
        WHOLEMEMORY_LOGIC_ERROR             "WHOLEMEMORY_LOGIC_ERROR"  # logic error
        WHOLEMEMORY_CUDA_ERROR              "WHOLEMEMORY_CUDA_ERROR"  # CUDA error
        WHOLEMEMORY_COMMUNICATION_ERROR     "WHOLEMEMORY_COMMUNICATION_ERROR"  # communication error
        WHOLEMEMORY_INVALID_INPUT           "WHOLEMEMORY_INVALID_INPUT"  # invalid input, e.g. nullptr
        WHOLEMEMORY_INVALID_VALUE           "WHOLEMEMORY_INVALID_VALUE"  # input value is invalid
        WHOLEMEMORY_OUT_OF_MEMORY           "WHOLEMEMORY_OUT_OF_MEMORY"  # out of memory
        WHOLEMEMORY_NOT_SUPPORTED           "WHOLEMEMORY_NOT_SUPPORTED"  # not supported

    ctypedef enum wholememory_memory_type_t:
        WHOLEMEMORY_MT_NONE                 "WHOLEMEMORY_MT_NONE"
        WHOLEMEMORY_MT_CONTINUOUS           "WHOLEMEMORY_MT_CONTINUOUS"
        WHOLEMEMORY_MT_CHUNKED              "WHOLEMEMORY_MT_CHUNKED"
        WHOLEMEMORY_MT_DISTRIBUTED          "WHOLEMEMORY_MT_DISTRIBUTED"
        WHOLEMEMORY_MT_HIERARCHY            "WHOLEMEMORY_MT_HIERARCHY"

    ctypedef enum wholememory_memory_location_t:
        WHOLEMEMORY_ML_NONE                 "WHOLEMEMORY_ML_NONE"
        WHOLEMEMORY_ML_DEVICE               "WHOLEMEMORY_ML_DEVICE"
        WHOLEMEMORY_ML_HOST                 "WHOLEMEMORY_ML_HOST"

    ctypedef enum wholememory_distributed_backend_t:
        WHOLEMEMORY_DB_NONE                 "WHOLEMEMORY_DB_NONE"
        WHOLEMEMORY_DB_NCCL                 "WHOLEMEMORY_DB_NCCL"
        WHOLEMEMORY_DB_NVSHMEM              "WHOLEMEMORY_DB_NVSHMEM"

    ctypedef enum LogLevel:
        LEVEL_FATAL                         "LEVEL_FATAL"
        LEVEL_ERROR                         "LEVEL_ERROR"
        LEVEL_WARN                          "LEVEL_WARN"
        LEVEL_INFO                          "LEVEL_INFO"
        LEVEL_DEBUG                         "LEVEL_DEBUG"
        LEVEL_TRACE                         "LEVEL_TRACE"

    cdef wholememory_error_code_t wholememory_init(unsigned int flags, LogLevel log_level)

    cdef wholememory_error_code_t wholememory_finalize()

    cdef struct wholememory_unique_id_t:
        char internal[128]

    cdef struct wholememory_comm_:
        pass

    ctypedef wholememory_comm_ * wholememory_comm_t

    cdef wholememory_error_code_t wholememory_create_unique_id(wholememory_unique_id_t * unique_id)

    cdef wholememory_error_code_t wholememory_create_communicator(wholememory_comm_t * comm,
                                                                  wholememory_unique_id_t unique_id,
                                                                  int rank,
                                                                  int size)

    cdef wholememory_error_code_t wholememory_destroy_communicator(wholememory_comm_t comm)

    cdef wholememory_error_code_t wholememory_communicator_support_type_location(
            wholememory_comm_t comm,
            wholememory_memory_type_t memory_type,
            wholememory_memory_location_t memory_location)

    cdef wholememory_error_code_t wholememory_communicator_get_rank(int * rank, wholememory_comm_t comm)

    cdef wholememory_error_code_t wholememory_communicator_get_size(int * size, wholememory_comm_t comm)

    cdef wholememory_error_code_t wholememory_communicator_barrier(wholememory_comm_t comm)

    cdef struct wholememory_handle_:
        pass

    ctypedef wholememory_handle_ * wholememory_handle_t

    cdef wholememory_error_code_t wholememory_malloc(wholememory_handle_t * wholememory_handle_ptr,
                                                     size_t total_size,
                                                     wholememory_comm_t comm,
                                                     wholememory_memory_type_t memory_type,
                                                     wholememory_memory_location_t memory_location,
                                                     size_t data_granularity,
                                                     size_t * rank_entry_partition)

    cdef wholememory_error_code_t wholememory_free(wholememory_handle_t wholememory_handle)

    cdef wholememory_error_code_t wholememory_get_communicator(wholememory_comm_t * comm,
                                                               wholememory_handle_t wholememory_handle)

    cdef wholememory_error_code_t wholememory_get_local_communicator(wholememory_comm_t * comm,
                                                                     wholememory_handle_t wholememory_handle)

    cdef wholememory_error_code_t wholememory_get_cross_communicator(wholememory_comm_t * comm,
                                                                     wholememory_handle_t wholememory_handle)

    cdef wholememory_memory_type_t wholememory_get_memory_type(wholememory_handle_t wholememory_handle)

    cdef wholememory_memory_location_t wholememory_get_memory_location(wholememory_handle_t wholememory_handle)

    cdef size_t wholememory_get_total_size(wholememory_handle_t wholememory_handle)

    cdef wholememory_error_code_t wholememory_get_local_memory(void** local_ptr,
                                                               size_t * local_size,
                                                               size_t * local_offset,
                                                               wholememory_handle_t wholememory_handle)

    cdef wholememory_error_code_t wholememory_get_local_size(size_t * local_size,
                                                             wholememory_handle_t wholememory_handle)

    cdef wholememory_error_code_t wholememory_get_local_offset(size_t * local_offset,
                                                               wholememory_handle_t wholememory_handle)

    cdef wholememory_error_code_t wholememory_get_rank_memory(void** rank_memory_ptr,
                                                              size_t * rank_memory_size,
                                                              size_t * rank_memory_offset,
                                                              int rank,
                                                              wholememory_handle_t wholememory_handle)

    cdef wholememory_error_code_t wholememory_equal_entry_partition_plan(size_t* entry_per_rank,
                                                                           size_t total_entry_count,
                                                                           int world_size)

    cdef wholememory_error_code_t wholememory_get_global_pointer(void** global_ptr,
                                                                 wholememory_handle_t wholememory_handle)

    cdef wholememory_error_code_t wholememory_get_rank_partition_sizes(size_t * rank_mem_sizes,
                                                        wholememory_handle_t wholememory_handle)

    cdef wholememory_error_code_t wholememory_get_rank_partition_offsets(size_t * rank_mem_offsets,
                                                          wholememory_handle_t wholememory_handle)

    cdef int fork_get_device_count()

    cdef wholememory_error_code_t wholememory_load_from_file(wholememory_handle_t wholememory_handle,
                                                             size_t memory_offset,
                                                             size_t memory_entry_size,
                                                             size_t file_entry_size,
                                                             const char** file_names,
                                                             int file_count,
                                                             int round_robin_size)

    cdef wholememory_error_code_t wholememory_store_to_file(wholememory_handle_t wholememory_handle,
                                                            size_t memory_offset,
                                                            size_t memory_entry_stride,
                                                            size_t file_entry_size,
                                                            const char *local_file_name)

    cdef bool wholememory_is_build_with_nvshmem()

    cdef wholememory_error_code_t wholememory_communicator_set_distributed_backend(wholememory_comm_t comm,
                                                                wholememory_distributed_backend_t distributed_backend)

    cdef wholememory_distributed_backend_t wholememory_communicator_get_distributed_backend(
                                                                            wholememory_comm_t comm)
    cdef bool wholememory_is_intranode_communicator(wholememory_comm_t comm)
    cdef bool wholememory_is_intra_mnnvl_communicator(wholememory_comm_t comm)


    cdef struct clique_info_t:
        int is_in_clique
        int clique_first_rank
        int clique_rank
        int clique_rank_num
        int clique_id
        int clique_num

    cdef wholememory_error_code_t wholememory_communicator_get_clique_info(clique_info_t* clique_info, wholememory_comm_t comm)


    cdef wholememory_error_code_t wholememory_split_communicator(wholememory_comm_t* new_comm,
                                                        wholememory_comm_t comm,
                                                        int color,
                                                        int key)

cpdef enum WholeMemoryErrorCode:
    Success = WHOLEMEMORY_SUCCESS
    UnknowError = WHOLEMEMORY_UNKNOW_ERROR
    NotImplemented = WHOLEMEMORY_NOT_IMPLEMENTED
    LogicError = WHOLEMEMORY_LOGIC_ERROR
    CUDAError = WHOLEMEMORY_CUDA_ERROR
    CommunicationError = WHOLEMEMORY_COMMUNICATION_ERROR
    InvalidInput = WHOLEMEMORY_INVALID_INPUT
    InvalidValue = WHOLEMEMORY_INVALID_VALUE
    OutOfMemory = WHOLEMEMORY_OUT_OF_MEMORY
    NotSupported = WHOLEMEMORY_NOT_SUPPORTED

cpdef enum WholeMemoryMemoryType:
    MtNone = WHOLEMEMORY_MT_NONE
    MtContinuous = WHOLEMEMORY_MT_CONTINUOUS
    MtChunked = WHOLEMEMORY_MT_CHUNKED
    MtDistributed = WHOLEMEMORY_MT_DISTRIBUTED
    MtHierarchy = WHOLEMEMORY_MT_HIERARCHY

cpdef enum WholeMemoryMemoryLocation:
    MlNone = WHOLEMEMORY_ML_NONE
    MlDevice = WHOLEMEMORY_ML_DEVICE
    MlHost = WHOLEMEMORY_ML_HOST

cpdef enum WholeMemoryDistributedBackend:
    DbNone = WHOLEMEMORY_DB_NONE
    DbNCCL = WHOLEMEMORY_DB_NCCL
    DbNVSHMEM = WHOLEMEMORY_DB_NVSHMEM

cpdef enum WholeMemoryLogLevel:
    LevFatal = LEVEL_FATAL
    LevError = LEVEL_ERROR
    LevWarn = LEVEL_WARN
    LevInfo = LEVEL_INFO
    LevDebug = LEVEL_DEBUG
    LevTrace = LEVEL_TRACE

cdef check_wholememory_error_code(wholememory_error_code_t err):
    cdef WholeMemoryErrorCode err_code = int(err)
    if err_code == Success:
        return
    elif err_code == UnknowError:
        raise Exception('Unknown error')
    elif err_code == NotImplemented:
        raise NotImplementedError('Not implemented')
    elif err_code == LogicError:
        raise RuntimeError('Logic error')
    elif err_code == CUDAError:
        raise RuntimeError('CUDA error')
    elif err_code == CommunicationError:
        raise RuntimeError('Communication error')
    elif err_code == InvalidInput:
        raise ValueError('Invalid input')
    elif err_code == InvalidValue:
        raise ValueError('Invalid value')
    elif err_code == OutOfMemory:
        raise MemoryError('Out of memory')
    else:
        raise NotImplementedError('Error code %d not recognized' % (int(err),))


cdef extern from "wholememory/tensor_description.h":
    ctypedef enum wholememory_dtype_t:
        WHOLEMEMORY_DT_UNKNOWN  "WHOLEMEMORY_DT_UNKNOWN"
        WHOLEMEMORY_DT_FLOAT    "WHOLEMEMORY_DT_FLOAT"
        WHOLEMEMORY_DT_HALF     "WHOLEMEMORY_DT_HALF"
        WHOLEMEMORY_DT_DOUBLE   "WHOLEMEMORY_DT_DOUBLE"
        WHOLEMEMORY_DT_BF16     "WHOLEMEMORY_DT_BF16"
        WHOLEMEMORY_DT_INT      "WHOLEMEMORY_DT_INT"
        WHOLEMEMORY_DT_INT64    "WHOLEMEMORY_DT_INT64"
        WHOLEMEMORY_DT_INT16    "WHOLEMEMORY_DT_INT16"
        WHOLEMEMORY_DT_INT8     "WHOLEMEMORY_DT_INT8"
        WHOLEMEMORY_DT_COUNT    "WHOLEMEMORY_DT_COUNT"

    cdef struct wholememory_tensor_description_t:
        int64_t sizes[8]
        int64_t strides[8]
        int64_t storage_offset
        int dim
        wholememory_dtype_t dtype

    cdef size_t wholememory_dtype_get_element_size(wholememory_dtype_t dtype)

    cdef int64_t wholememory_get_memory_element_count_from_tensor(
            wholememory_tensor_description_t * p_tensor_description)


cdef extern from "wholememory/env_func_ptrs.h":
    ctypedef enum wholememory_memory_allocation_type_t:
        WHOLEMEMORY_MA_NONE                 "WHOLEMEMORY_MA_NONE"
        WHOLEMEMORY_MA_DEVICE               "WHOLEMEMORY_MA_DEVICE"
        WHOLEMEMORY_MA_HOST                 "WHOLEMEMORY_MA_HOST"
        WHOLEMEMORY_MA_PINNED               "WHOLEMEMORY_MA_PINNED"

    ctypedef void (*wholememory_create_memory_context_func_t)(void ** memory_context,
                                                              void * global_context)

    ctypedef void (*wholememory_destroy_memory_context_func_t)(void * memory_context,
                                                               void * global_context)

    ctypedef void * (*wholememory_malloc_func_t)(wholememory_tensor_description_t * desc,
                                                 wholememory_memory_allocation_type_t memory_allocation_type,
                                                 void * memory_context,
                                                 void * global_context)

    ctypedef void (*wholememory_free_func_t)(void * memory_context, void * global_context)

    cdef struct wholememory_temp_memory_func_t:
        wholememory_create_memory_context_func_t create_memory_context_fn
        wholememory_destroy_memory_context_func_t destroy_memory_context_fn
        wholememory_malloc_func_t malloc_fn
        wholememory_free_func_t free_fn
        void * global_context

    cdef struct wholememory_output_memory_func_t:
        wholememory_malloc_func_t malloc_fn
        wholememory_free_func_t free_fn
        void * global_context

    cdef struct wholememory_env_func_t:
        wholememory_temp_memory_func_t temporary_fns
        wholememory_output_memory_func_t output_fns


cpdef enum WholeMemoryMemoryAllocType:
    MatNone = WHOLEMEMORY_MA_NONE
    MatDevice = WHOLEMEMORY_MA_DEVICE
    MatHost = WHOLEMEMORY_MA_HOST
    MatPinned = WHOLEMEMORY_MA_PINNED

cdef class PyMemoryAllocType:
    cdef wholememory_memory_allocation_type_t alloc_type

    def __cinit__(self):
        self.alloc_type = WHOLEMEMORY_MA_NONE

    def set_type(self, WholeMemoryMemoryAllocType new_type):
        self.alloc_type = <wholememory_memory_allocation_type_t> <int64_t> new_type

    def get_type(self):
        return <int64_t> self.alloc_type

    def set_ctype(self, wholememory_memory_allocation_type_t alloc_type):
        self.alloc_type = alloc_type

    def get_ctype(self):
        return self.alloc_type

cdef class GlobalContextWrapper:
    cdef PyObject * temp_create_context_fn
    cdef PyObject * temp_destroy_context_fn
    cdef PyObject * temp_malloc_fn
    cdef PyObject * temp_free_fn
    cdef PyObject * temp_global_context
    cdef PyObject * output_malloc_fn
    cdef PyObject * output_free_fn
    cdef PyObject * output_global_context
    cdef wholememory_env_func_t env_func

    def __cinit__(self):
        self.temp_create_context_fn = NULL
        self.temp_destroy_context_fn = NULL
        self.temp_malloc_fn = NULL
        self.temp_free_fn = NULL
        self.temp_global_context = NULL
        self.output_malloc_fn = NULL
        self.output_free_fn = NULL
        self.output_global_context = NULL

    def __dealloc__(self):
        Py_DECREF(self.self.temp_create_context_fn)
        Py_DECREF(self.self.temp_destroy_context_fn)
        Py_DECREF(self.self.temp_malloc_fn)
        Py_DECREF(self.self.temp_free_fn)
        if self.temp_global_context:
            Py_DECREF(self.self.temp_global_context)
        Py_DECREF(self.self.output_malloc_fn)
        Py_DECREF(self.self.output_free_fn)
        if self.output_global_context:
            Py_DECREF(self.self.output_global_context)

    cpdef create_context(self,
                         temp_create_context_fn,
                         temp_destroy_context_fn,
                         temp_malloc_fn,
                         temp_free_fn,
                         temp_global_context,
                         output_malloc_fn,
                         output_free_fn,
                         output_global_context):
        self.temp_create_context_fn = <PyObject *> temp_create_context_fn
        Py_INCREF(self.temp_create_context_fn)
        self.temp_destroy_context_fn = <PyObject *> temp_destroy_context_fn
        Py_INCREF(self.temp_destroy_context_fn)
        self.temp_malloc_fn = <PyObject *> temp_malloc_fn
        Py_INCREF(self.temp_malloc_fn)
        self.temp_free_fn = <PyObject *> temp_free_fn
        Py_INCREF(self.temp_free_fn)
        if temp_global_context:
            self.temp_global_context = <PyObject *> temp_global_context
            Py_INCREF(self.temp_global_context)
        self.output_malloc_fn = <PyObject *> output_malloc_fn
        Py_INCREF(self.output_malloc_fn)
        self.output_free_fn = <PyObject *> output_free_fn
        Py_INCREF(self.output_free_fn)
        if output_global_context:
            self.output_global_context = <PyObject *> output_global_context
            Py_INCREF(self.output_global_context)
        self.env_func.temporary_fns.create_memory_context_fn = <wholememory_create_memory_context_func_t> &python_cb_wrapper_temp_create_context
        self.env_func.temporary_fns.destroy_memory_context_fn = <wholememory_destroy_memory_context_func_t> &python_cb_wrapper_temp_destroy_context
        self.env_func.temporary_fns.malloc_fn = <wholememory_malloc_func_t> &python_cb_wrapper_temp_malloc
        self.env_func.temporary_fns.free_fn = <wholememory_free_func_t> &python_cb_wrapper_temp_free
        self.env_func.temporary_fns.global_context = <PyObject *> self
        self.env_func.output_fns.malloc_fn = <wholememory_malloc_func_t> &python_cb_wrapper_output_malloc
        self.env_func.output_fns.free_fn = <wholememory_free_func_t> &python_cb_wrapper_output_free
        self.env_func.output_fns.global_context = <PyObject *> self

    cpdef int64_t get_env_fns(self):
        return <int64_t> (&self.env_func)

cdef void python_cb_wrapper_temp_create_context(void** memory_context,
                                                void * global_context) nogil:
    cdef PyObject * ret_memory_context = NULL
    with gil:
        wrapped_global_context = <GlobalContextWrapper> <PyObject *> global_context
        python_fn = wrapped_global_context.temp_create_context_fn
        python_global_context = wrapped_global_context.temp_global_context
        args = PyTuple_New(1)
        Py_INCREF(<object> python_global_context)
        PyTuple_SetItem(args, 0, <object> python_global_context)
        py_memory_context = PyObject_CallObject(<object> python_fn, <object> args)
        ret_memory_context = <PyObject *> py_memory_context
        Py_DECREF(args)
        Py_INCREF(ret_memory_context)
        (<PyObject **> memory_context)[0] = ret_memory_context
    return

cdef void python_cb_wrapper_temp_destroy_context(void * memory_context,
                                                 void * global_context) nogil:
    with gil:
        wrapped_global_context = <GlobalContextWrapper> <PyObject *> global_context
        python_fn = wrapped_global_context.temp_destroy_context_fn
        python_global_context = wrapped_global_context.temp_global_context
        args = PyTuple_New(2)
        Py_INCREF(<object> <PyObject *> memory_context)
        PyTuple_SetItem(args, 0, <object> <PyObject *> memory_context)
        Py_INCREF(<object> python_global_context)
        PyTuple_SetItem(args, 1, <object> python_global_context)
        PyObject_CallObject(<object> python_fn, <object> args)
        Py_DECREF(args)
        Py_DECREF(<PyObject *> memory_context)
    return

cdef void * python_cb_wrapper_temp_malloc(wholememory_tensor_description_t * tensor_desc,
                                          wholememory_memory_allocation_type_t malloc_type,
                                          void * memory_context,
                                          void * global_context) nogil:
    cdef int64_t res_ptr = 0
    with gil:
        wrapped_global_context = <GlobalContextWrapper> <PyObject *> global_context
        py_tensor_desc = PyWholeMemoryTensorDescription()
        py_tensor_desc.set_by_tensor_desc(tensor_desc)
        py_malloc_type = PyMemoryAllocType()
        py_malloc_type.set_type(malloc_type)
        python_fn = wrapped_global_context.temp_malloc_fn
        python_global_context = wrapped_global_context.temp_global_context
        args = PyTuple_New(4)
        Py_INCREF(py_tensor_desc)
        PyTuple_SetItem(args, 0, <object> py_tensor_desc)
        Py_INCREF(py_malloc_type)
        PyTuple_SetItem(args, 1, <object> py_malloc_type)
        Py_INCREF(<object> <PyObject *> memory_context)
        PyTuple_SetItem(args, 2, <object> <PyObject *> memory_context)
        Py_INCREF(<object> <PyObject *> python_global_context)
        PyTuple_SetItem(args, 3, <object> <PyObject *> python_global_context)
        res_ptr = PyLong_AsLongLong(PyObject_CallObject(<object> python_fn, <object> args))
        Py_DECREF(args)
    return <void *> res_ptr

cdef void python_cb_wrapper_temp_free(void * memory_context,
                                      void * global_context) nogil:
    with gil:
        wrapped_global_context = <GlobalContextWrapper> <PyObject *> global_context
        python_fn = wrapped_global_context.temp_free_fn
        python_global_context = wrapped_global_context.temp_global_context
        args = PyTuple_New(2)
        Py_INCREF(<object> <PyObject *> memory_context)
        PyTuple_SetItem(args, 0, <object> <PyObject *> memory_context)
        Py_INCREF(<object> python_global_context)
        PyTuple_SetItem(args, 1, <object> python_global_context)
        PyObject_CallObject(<object> python_fn, <object> args)
        Py_DECREF(args)
    return

cdef void * python_cb_wrapper_output_malloc(wholememory_tensor_description_t * tensor_desc,
                                            wholememory_memory_allocation_type_t malloc_type,
                                            void * memory_context,
                                            void * global_context) nogil:
    cdef int64_t res_ptr = 0
    with gil:
        wrapped_global_context = <GlobalContextWrapper> <PyObject *> global_context
        py_tensor_desc = PyWholeMemoryTensorDescription()
        py_tensor_desc.set_by_tensor_desc(tensor_desc)
        py_malloc_type = PyMemoryAllocType()
        py_malloc_type.set_type(malloc_type)
        python_fn = wrapped_global_context.output_malloc_fn
        python_global_context = wrapped_global_context.output_global_context
        args = PyTuple_New(4)
        Py_INCREF(py_tensor_desc)
        PyTuple_SetItem(args, 0, <object> <PyObject *> py_tensor_desc)
        Py_INCREF(py_malloc_type)
        PyTuple_SetItem(args, 1, <object> <PyObject *> py_malloc_type)
        Py_INCREF(<object> <PyObject *> memory_context)
        PyTuple_SetItem(args, 2, <object> <PyObject *> memory_context)
        Py_INCREF(<object> <PyObject *> python_global_context)
        PyTuple_SetItem(args, 3, <object> <PyObject *> python_global_context)
        res_ptr = PyLong_AsLongLong(PyObject_CallObject(<object> python_fn, <object> args))
        Py_DECREF(args)
    return <void *> res_ptr

cdef void python_cb_wrapper_output_free(void * memory_context,
                                        void * global_context) nogil:
    with gil:
        wrapped_global_context = <GlobalContextWrapper> <PyObject *> global_context
        python_fn = wrapped_global_context.output_free_fn
        python_global_context = wrapped_global_context.output_global_context
        args = PyTuple_New(2)
        Py_INCREF(<object> <PyObject *> memory_context)
        PyTuple_SetItem(args, 0, <object> <PyObject *> memory_context)
        Py_INCREF(<object> python_global_context)
        PyTuple_SetItem(args, 1, <object> python_global_context)
        PyObject_CallObject(<object> python_fn, <object> args)
        Py_DECREF(args)
    return


cdef extern from "wholememory/wholememory_tensor.h":
    cdef struct wholememory_tensor_:
        pass

    ctypedef wholememory_tensor_ * wholememory_tensor_t

    cdef wholememory_error_code_t wholememory_create_tensor(wholememory_tensor_t *wholememory_tensor,
                                                            wholememory_tensor_description_t *tensor_description,
                                                            wholememory_comm_t comm,
                                                            wholememory_memory_type_t memory_type,
                                                            wholememory_memory_location_t memory_location,
                                                            size_t * tensor_entry_partition)

    cdef wholememory_error_code_t wholememory_destroy_tensor(wholememory_tensor_t wholememory_tensor)

    cdef wholememory_error_code_t wholememory_make_tensor_from_pointer(wholememory_tensor_t *wholememory_tensor,
                                                                       void *data_ptr,
                                                                       wholememory_tensor_description_t *tensor_description)

    cdef wholememory_error_code_t wholememory_make_tensor_from_handle(wholememory_tensor_t *wholememory_tensor,
                                                                      wholememory_handle_t wholememory_handle,
                                                                      wholememory_tensor_description_t *tensor_description)

    cdef bool wholememory_tensor_has_handle(wholememory_tensor_t wholememory_tensor)

    cdef wholememory_handle_t wholememory_tensor_get_memory_handle(wholememory_tensor_t wholememory_tensor)

    cdef wholememory_tensor_description_t * wholememory_tensor_get_tensor_description(
            wholememory_tensor_t wholememory_tensor)

    cdef wholememory_error_code_t wholememory_tensor_get_entry_offsets(
        size_t * entry_offsets, wholememory_tensor_t wholememory_tensor);

    cdef wholememory_error_code_t wholememory_tensor_get_entry_partition_sizes(
        size_t * entry_partition, wholememory_tensor_t wholememory_tensor);

    cdef wholememory_error_code_t wholememory_tensor_get_local_entry_count(
        size_t * local_entry_count, wholememory_tensor_t wholememory_tensor);

    cdef wholememory_error_code_t wholememory_tensor_get_local_entry_start(
        size_t * local_entry_start, wholememory_tensor_t wholememory_tensor);

    cdef wholememory_error_code_t wholememory_tensor_get_subtensor(wholememory_tensor_t wholememory_tensor,
                                                                   int64_t *starts,
                                                                   int64_t *ends,
                                                                   wholememory_tensor_t *sub_wholememory_tensor)

    int64_t get_wholememory_tensor_count()


def py_get_wholememory_tensor_count():
    return get_wholememory_tensor_count()

cpdef enum WholeMemoryDataType:
    DtUnknown = WHOLEMEMORY_DT_UNKNOWN
    DtFloat = WHOLEMEMORY_DT_FLOAT
    DtHalf = WHOLEMEMORY_DT_HALF
    DtDouble = WHOLEMEMORY_DT_DOUBLE
    DtBF16 = WHOLEMEMORY_DT_BF16
    DtInt = WHOLEMEMORY_DT_INT
    DtInt64 = WHOLEMEMORY_DT_INT64
    DtInt16 = WHOLEMEMORY_DT_INT16
    DtInt8 = WHOLEMEMORY_DT_INT8
    DtCount = WHOLEMEMORY_DT_COUNT

cdef extern from "wholememory/embedding.h":
    cdef struct wholememory_embedding_cache_policy_:
        pass

    cdef struct wholememory_embedding_optimizer_:
        pass

    cdef struct wholememory_embedding_:
        pass

    ctypedef wholememory_embedding_cache_policy_ * wholememory_embedding_cache_policy_t
    ctypedef wholememory_embedding_optimizer_ * wholememory_embedding_optimizer_t
    ctypedef wholememory_embedding_ * wholememory_embedding_t

    ctypedef enum wholememory_access_type_t:
        WHOLEMEMORY_AT_NONE                 "WHOLEMEMORY_AT_NONE"
        WHOLEMEMORY_AT_READONLY             "WHOLEMEMORY_AT_READONLY"
        WHOLEMEMORY_AT_READWRITE            "WHOLEMEMORY_AT_READWRITE"

    ctypedef enum wholememory_optimizer_type_t:
        WHOLEMEMORY_OPT_NONE                "WHOLEMEMORY_OPT_NONE"
        WHOLEMEMORY_OPT_SGD                 "WHOLEMEMORY_OPT_SGD"
        WHOLEMEMORY_OPT_LAZY_ADAM           "WHOLEMEMORY_OPT_LAZY_ADAM"
        WHOLEMEMORY_OPT_RMSPROP             "WHOLEMEMORY_OPT_RMSPROP"
        WHOLEMEMORY_OPT_ADAGRAD             "WHOLEMEMORY_OPT_ADAGRAD"

    cdef wholememory_error_code_t wholememory_create_embedding_optimizer(
            wholememory_embedding_optimizer_t * optimizer, wholememory_optimizer_type_t optimizer_type)

    cdef wholememory_error_code_t wholememory_optimizer_set_parameter(
            wholememory_embedding_optimizer_t optimizer, const char * parameter_name, void * value)

    cdef void wholememory_destroy_embedding_optimizer(wholememory_embedding_optimizer_t optimizer)

    cdef wholememory_error_code_t wholememory_create_embedding_cache_policy(
            wholememory_embedding_cache_policy_t * cache_policy,
            wholememory_comm_t cache_level_comm,
            wholememory_memory_type_t memory_type,
            wholememory_memory_location_t memory_location,
            wholememory_access_type_t access_type,
            float cache_ratio)

    cdef wholememory_error_code_t wholememory_destroy_embedding_cache_policy(
            wholememory_embedding_cache_policy_t cache_policy)

    cdef wholememory_error_code_t wholememory_create_embedding(
            wholememory_embedding_t * wholememory_embedding,
            wholememory_tensor_description_t * embedding_tensor_description,
            wholememory_comm_t comm,
            wholememory_memory_type_t memory_type,
            wholememory_memory_location_t memory_location,
            wholememory_embedding_cache_policy_t cache_policy,
            size_t * embedding_entry_partition,
            int user_defined_sms,
            int round_robin_size)

    cdef wholememory_error_code_t wholememory_destroy_embedding(
            wholememory_embedding_t wholememory_embedding)

    cdef wholememory_error_code_t wholememory_embedding_set_optimizer(
            wholememory_embedding_t  wholememory_embedding,
            wholememory_embedding_optimizer_t optimizer);

    cdef wholememory_error_code_t wholememory_embedding_gather(wholememory_embedding_t wholememory_embedding,
                                                               wholememory_tensor_t indices,
                                                               wholememory_tensor_t output,
                                                               bool adjust_cache,
                                                               wholememory_env_func_t * p_env_fns,
                                                               int64_t stream_int)

    cdef wholememory_error_code_t wholememory_embedding_gather_gradient_apply(
            wholememory_embedding_t wholememory_embedding,
            wholememory_tensor_t indices,
            wholememory_tensor_t grads,
            bool adjust_cache,
            float lr,
            wholememory_env_func_t * p_env_fns,
            int64_t stream_int)

    cdef wholememory_tensor_t wholememory_embedding_get_embedding_tensor(
            wholememory_embedding_t wholememory_embedding)

    cdef const char * const * wholememory_embedding_get_optimizer_state_names(
            wholememory_embedding_t wholememory_embedding)

    cdef wholememory_tensor_t wholememory_embedding_get_optimizer_state(
            wholememory_embedding_t wholememory_embedding, const char * name)

    cdef wholememory_error_code_t wholememory_embedding_writeback_cache(
            wholememory_embedding_t wholememory_embedding, int64_t stream_int)

    cdef wholememory_error_code_t wholememory_embedding_drop_all_cache(
            wholememory_embedding_t wholememory_embedding, int64_t stream_int)


cpdef enum WholeMemoryAccessType:
    AtNone = WHOLEMEMORY_AT_NONE
    AtReadOnly = WHOLEMEMORY_AT_READONLY
    AtReadWrite = WHOLEMEMORY_AT_READWRITE

cpdef enum WholeMemoryOptimizerType:
    OptNone = WHOLEMEMORY_OPT_NONE
    OptSgd = WHOLEMEMORY_OPT_SGD
    OptLazyAdam = WHOLEMEMORY_OPT_LAZY_ADAM
    OptAdaGrad = WHOLEMEMORY_OPT_ADAGRAD
    OptRmsProp = WHOLEMEMORY_OPT_RMSPROP

cdef class WholeMemoryOptimizer:
    cdef wholememory_embedding_optimizer_t wm_optimizer
    cdef wholememory_optimizer_type_t optimizer_type
    cdef public dict param_dict

    def __cinit__(self):
        self.wm_optimizer = NULL
        self.optimizer_type = WHOLEMEMORY_OPT_NONE

    def __init__(self):
        self.param_dict = {}

    def create_optimizer(self,
                         WholeMemoryOptimizerType optimizer_type,
                         dict param_dict):
        cdef str param_key
        cdef float param_value
        self.optimizer_type = <wholememory_optimizer_type_t> <int> optimizer_type
        self.param_dict = param_dict
        check_wholememory_error_code(wholememory_create_embedding_optimizer(&self.wm_optimizer, self.optimizer_type))
        for param_key, param_value in self.param_dict.items():
            key_bytes = param_key.encode('utf-8')
            check_wholememory_error_code(
                wholememory_optimizer_set_parameter(self.wm_optimizer, key_bytes, &param_value))

    def add_embedding(self,
                    PyWholeMemoryEmbedding embedding):
        wholememory_embedding_set_optimizer(embedding.wm_embedding, self.wm_optimizer)

    def destroy_optimizer(self):
        if self.wm_optimizer == NULL:
            return
        wholememory_destroy_embedding_optimizer(self.wm_optimizer)
        self.wm_optimizer = NULL
        self.optimizer_type = WHOLEMEMORY_OPT_NONE
        self.param_dict = None

def create_optimizer(WholeMemoryOptimizerType optimizer_type,
                     dict param_dict):
    wm_optimizer = WholeMemoryOptimizer()
    wm_optimizer.create_optimizer(optimizer_type, param_dict)
    return wm_optimizer

def create_non_optimizer():
    return WholeMemoryOptimizer()

cdef class WholeMemoryCachePolicy:
    cdef wholememory_embedding_cache_policy_t cache_policy
    cdef wholememory_memory_type_t memory_type
    cdef wholememory_memory_location_t memory_location
    cdef wholememory_access_type_t access_type
    cdef float ratio
    cdef PyWholeMemoryComm comm

    def __cinit__(self):
        self.cache_policy = NULL
        self.memory_type = WHOLEMEMORY_MT_NONE
        self.memory_location = WHOLEMEMORY_ML_NONE
        self.access_type = WHOLEMEMORY_AT_NONE
        self.ratio = 0.5
        self.comm = None

    def create_policy(self,
                      PyWholeMemoryComm comm,
                      WholeMemoryMemoryType memory_type,
                      WholeMemoryMemoryLocation memory_location,
                      WholeMemoryAccessType access_type,
                      float ratio):
        self.memory_type = <wholememory_memory_type_t> <int> memory_type
        self.memory_location = <wholememory_memory_location_t> <int> memory_location
        self.access_type = <wholememory_access_type_t> <int> access_type
        self.ratio = ratio
        check_wholememory_error_code(wholememory_create_embedding_cache_policy(&self.cache_policy,
                                                                               comm.comm_id,
                                                                               self.memory_type,
                                                                               self.memory_location,
                                                                               self.access_type,
                                                                               self.ratio))

    def destroy_policy(self):
        if self.cache_policy == NULL:
            return
        check_wholememory_error_code(wholememory_destroy_embedding_cache_policy(self.cache_policy))
        self.cache_policy = NULL
        self.memory_type = WHOLEMEMORY_MT_NONE
        self.memory_location = WHOLEMEMORY_ML_NONE
        self.access_type = WHOLEMEMORY_AT_NONE
        self.ratio = 0.5
        self.comm = None

def create_cache_policy(PyWholeMemoryComm comm,
                        WholeMemoryMemoryType memory_type,
                        WholeMemoryMemoryLocation memory_location,
                        WholeMemoryAccessType access_type,
                        float ratio):
    cache_policy = WholeMemoryCachePolicy()
    cache_policy.create_policy(comm, memory_type, memory_location, access_type, ratio)
    return cache_policy

def create_non_cache_policy():
    return WholeMemoryCachePolicy()

cdef class PyWholeMemoryEmbedding:
    cdef wholememory_embedding_t wm_embedding
    cdef wholememory_memory_type_t memory_type
    cdef wholememory_memory_location_t memory_location

    def __cinit__(self):
        self.wm_embedding = NULL
        self.memory_type = WHOLEMEMORY_MT_NONE
        self.memory_location = WHOLEMEMORY_ML_NONE

    def create_embedding(self,
                         PyWholeMemoryTensorDescription tensor_desc,
                         PyWholeMemoryComm comm,
                         WholeMemoryMemoryType memory_type,
                         WholeMemoryMemoryLocation memory_location,
                         WholeMemoryCachePolicy cache_policy,
                         cython.size_t[:] embedding_entry_partition,
                         int user_defined_sms,
                         int round_robin_size):
        self.memory_type = <wholememory_memory_type_t> <int> memory_type
        self.memory_location = <wholememory_memory_location_t> <int> memory_location
        cdef size_t* partition_ptr = NULL
        if embedding_entry_partition is not None and embedding_entry_partition.size > 0:
            partition_ptr = <size_t*>&embedding_entry_partition[0]
        check_wholememory_error_code(wholememory_create_embedding(&self.wm_embedding,
                                                                  &tensor_desc.tensor_description,
                                                                  comm.comm_id,
                                                                  self.memory_type,
                                                                  self.memory_location,
                                                                  cache_policy.cache_policy,
                                                                  partition_ptr,
                                                                  user_defined_sms,
                                                                  round_robin_size))

    def destroy_embedding(self):
        check_wholememory_error_code(wholememory_destroy_embedding(self.wm_embedding))

    def writeback_all_cache(self,
                            int64_t stream):
        check_wholememory_error_code(wholememory_embedding_writeback_cache(self.wm_embedding, stream))

    def drop_all_cache(self,
                       int64_t stream):
        check_wholememory_error_code(wholememory_embedding_drop_all_cache(self.wm_embedding, stream))

    def get_embedding_tensor(self):
        cdef wholememory_tensor_t wm_tensor
        wm_tensor = wholememory_embedding_get_embedding_tensor(self.wm_embedding)
        py_wm_tensor = PyWholeMemoryTensor()
        py_wm_tensor.from_c_handle(wm_tensor)
        return py_wm_tensor

    def get_optimizer_state_names(self):
        cdef int i = 0
        result = []
        cdef const char * const * state_names
        state_names = wholememory_embedding_get_optimizer_state_names(self.wm_embedding)
        if state_names != NULL:
            while state_names[i] != NULL:
                result.append(<object> PyUnicode_FromString(state_names[i]))
                i += 1
        return result

    def get_optimizer_state(self,
                            state_name):
        cdef wholememory_tensor_t state_tensor
        state_tensor = wholememory_embedding_get_optimizer_state(
            self.wm_embedding,
            PyUnicode_AsUTF8(state_name))
        py_state_tensor = PyWholeMemoryTensor()
        py_state_tensor.from_c_handle(state_tensor)
        return py_state_tensor

def create_embedding(PyWholeMemoryTensorDescription tensor_desc,
                     PyWholeMemoryComm comm,
                     WholeMemoryMemoryType memory_type,
                     WholeMemoryMemoryLocation memory_location,
                     WholeMemoryCachePolicy cache_policy,
                     cython.size_t[:] embedding_entry_partition,
                     int user_defined_sms,
                     int round_robin_size):
    wm_embedding = PyWholeMemoryEmbedding()
    wm_embedding.create_embedding(tensor_desc,
                                  comm,
                                  memory_type,
                                  memory_location,
                                  cache_policy,
                                  embedding_entry_partition,
                                  user_defined_sms,
                                  round_robin_size)
    return wm_embedding

cpdef void EmbeddingGatherForward(PyWholeMemoryEmbedding wm_embedding,
                                  WrappedLocalTensor indice,
                                  WrappedLocalTensor output,
                                  bool adjust_cache,
                                  int64_t p_env_fns_int,
                                  int64_t stream_int):
    check_wholememory_error_code(wholememory_embedding_gather(wm_embedding.wm_embedding,
                                                              <wholememory_tensor_t> <int64_t> indice.get_c_handle(),
                                                              <wholememory_tensor_t> <int64_t> output.get_c_handle(),
                                                              adjust_cache,
                                                              <wholememory_env_func_t *> <void *> p_env_fns_int,
                                                              stream_int))

cpdef void EmbeddingGatherGradientApply(PyWholeMemoryEmbedding wm_embedding,
                                        WrappedLocalTensor indice,
                                        WrappedLocalTensor grads,
                                        bool adjust_cache,
                                        float lr,
                                        int64_t p_env_fns_int,
                                        int64_t stream_int):
    check_wholememory_error_code(wholememory_embedding_gather_gradient_apply(
        wm_embedding.wm_embedding,
        <wholememory_tensor_t> <int64_t> indice.get_c_handle(),
        <wholememory_tensor_t> <int64_t> grads.get_c_handle(),
        adjust_cache,
        lr,
        <wholememory_env_func_t *> <void *> p_env_fns_int,
        stream_int))

######################################################################
# dlpack
# https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h
# https://github.com/cupy/cupy/blob/master/cupy/_core/dlpack.pyx

cpdef enum DLDeviceType:
    kDLCPU = 1
    kDLCUDA = 2
    kDLCUDAHost = 3

ctypedef struct DLDevice:
    DLDeviceType device_type
    int device_id

cdef enum DLDataTypeCode:
    kDLInt = 0
    kDLUInt = 1
    kDLFloat = 2
    kDLBfloat = 4

ctypedef struct DLDataType:
    uint8_t code
    uint8_t bits
    uint16_t lanes

ctypedef struct DLTensor:
    void * data
    DLDevice device
    int ndim
    DLDataType dtype
    int64_t * shape
    int64_t * strides
    uint64_t byte_offset

ctypedef struct DLManagedTensor:
    DLTensor dl_tensor
    void * manager_ctx
    void (*deleter)(DLManagedTensor *)

cdef void pycapsule_deleter(object dltensor):
    cdef DLManagedTensor * dlm_tensor
    # Do not invoke the deleter on a used capsule
    if cpython.PyCapsule_IsValid(dltensor, 'dltensor'):
        dlm_tensor = <DLManagedTensor *> cpython.PyCapsule_GetPointer(
            dltensor, 'dltensor')
        dlm_tensor.deleter(dlm_tensor)

cdef void deleter(DLManagedTensor * tensor) with gil:
    if tensor.manager_ctx is NULL:
        return
    cpython.Py_DECREF(<PyWholeMemoryFlattenDlpack> tensor.manager_ctx)
    tensor.manager_ctx = NULL
    stdlib.free(tensor)

# end dlpack
######################################################################

cdef class PyWholeMemoryUniqueID:
    cdef wholememory_unique_id_t wholememory_unique_id
    cdef Py_ssize_t shape[1]
    cdef Py_ssize_t strides[1]
    cdef int64_t shape_int64_t[1]
    cdef int64_t strides_int64_t[1]

    def __cinit__(self):
        self.shape[0] = sizeof(self.wholememory_unique_id.internal)
        self.strides[0] = 1
        self.shape_int64_t[0] = self.shape[0]
        self.strides_int64_t[0] = self.strides[0]

    def __len__(self):
        return self.shape[0]

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        buffer.buf = &self.wholememory_unique_id.internal[0]
        buffer.format = 'c'
        buffer.internal = NULL
        buffer.itemsize = 1
        buffer.len = self.shape[0]
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        buffer.buf = NULL
        buffer.format = 'c'
        buffer.len = 0
        buffer.ndim = 0
        buffer.obj = None
        buffer.shape = NULL
        buffer.strides = NULL

    def __dlpack__(self, stream=None):
        cdef DLManagedTensor * dlm_tensor = \
            <DLManagedTensor *> stdlib.malloc(sizeof(DLManagedTensor))
        cdef DLTensor * dl_tensor = &dlm_tensor.dl_tensor
        dl_tensor.data = &self.wholememory_unique_id.internal[0]
        dl_tensor.ndim = 1
        dl_tensor.shape = &self.shape_int64_t[0]
        dl_tensor.strides = &self.strides_int64_t[0]
        dl_tensor.byte_offset = 0
        dl_tensor.device.device_type, dl_tensor.device.device_id = self.__dlpack_device__()
        cdef DLDataType * dtype = &dl_tensor.dtype
        dtype.code = <uint8_t> kDLInt
        dtype.lanes = <uint16_t> 1
        dtype.bits = <uint8_t> 8

        dlm_tensor.manager_ctx = <void *> self
        cpython.Py_INCREF(self)
        dlm_tensor.deleter = deleter
        return cpython.PyCapsule_New(dlm_tensor, 'dltensor', <cpython.PyCapsule_Destructor> &pycapsule_deleter)

    def __dlpack_device__(self):
        return (kDLCPU, 0)

def init(unsigned int flags, LogLevel log_level = LEVEL_INFO):
    check_wholememory_error_code(wholememory_init(flags, log_level))

def finalize():
    check_wholememory_error_code(wholememory_finalize())

def create_unique_id():
    py_uid = PyWholeMemoryUniqueID()
    check_wholememory_error_code(wholememory_create_unique_id(&py_uid.wholememory_unique_id))
    return py_uid

cpdef enum WholeMemoryViewType:
    VtNone = 0
    VtLocal = 1
    VtGlobal = 2
    VtRemote = 3

def get_type_string(WholeMemoryDataType data_type):
    # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.interface.html#__array_interface__
    if data_type == DtFloat:
        return '<f4'
    elif data_type == DtHalf:
        return '<f2'
    elif data_type == DtDouble:
        return '<f8'
    elif data_type == DtBF16:
        return '<f2'
    elif data_type == DtInt:
        return '<i4'
    elif data_type == DtInt64:
        return '<i8'
    elif data_type == DtInt16:
        return '<i2'
    elif data_type == DtInt8:
        return '|i1'
    else:
        raise ValueError('data type %d not valid' % (int(data_type),))

cdef class PyWholeMemoryFlattenDlpack:
    cdef void * c_ptr
    cdef WholeMemoryDataType data_type
    cdef Py_ssize_t itemsize
    cdef public object typestr
    cdef Py_ssize_t shape[1]
    cdef Py_ssize_t strides[1]
    cdef int64_t shape_int64_t[1]
    cdef int64_t strides_int64_t[1]
    cdef WholeMemoryMemoryLocation device_type
    cdef int device_id

    def __cinit__(self):
        self.c_ptr = NULL
        self.shape[0] = 0
        self.strides[0] = 1
        self.shape_int64_t[0] = 0
        self.strides_int64_t[0] = 1
        self.itemsize = 0
        self.typestr = ''
        self.data_type = DtUnknown
        self.device_type = MlHost
        self.device_id = 0

    def set_view_device(self, WholeMemoryMemoryLocation device_type, int device_id):
        self.device_type = device_type
        self.device_id = device_id

    def get_view(self,
                 PyWholeMemoryHandle handle,
                 WholeMemoryDataType data_type,
                 WholeMemoryViewType view_type,
                 int target_rank):
        """Get view of a WholeMemoryHandle

        Parameters
        ----------
        handle : PyWholeMemoryHandle
            handler to the WholeMemory
        data_type: WholeMemoryDataType
            data type of the WholeMemory
        view_type : WholeMemoryViewType
            view type
        target_rank: int
            if view_type is VtRemote, target_rank is the rank of remote rank's memory, otherwise target_rank is ignored
        """
        self.data_type = data_type
        elt_size = wholememory_dtype_get_element_size(int(data_type))
        self.itemsize = elt_size
        if elt_size <= 0 or elt_size > 8:
            raise ValueError('data_type not supported')
        self.typestr = get_type_string(data_type)
        cdef WholeMemoryMemoryType mem_type
        cdef WholeMemoryMemoryLocation mem_location
        mem_type = int(wholememory_get_memory_type(handle.wholememory_handle))
        mem_location = int(wholememory_get_memory_location(handle.wholememory_handle))
        if self.device_type == MlHost and mem_location == MlDevice:
            raise ValueError('Device WholeMemory cannot get view from host.')
        if mem_type == MtDistributed and (view_type == VtGlobal or view_type == VtRemote):
            raise ValueError('Distributed WholeMemory have no view of Global or Remote')
        cdef size_t map_size
        cdef size_t map_offset
        cdef size_t global_size
        cdef wholememory_comm_t comm
        cdef int world_rank
        cdef int world_size
        if self.device_type == MlHost and mem_type == MtContinuous:
            check_wholememory_error_code(wholememory_get_communicator(&comm, handle.wholememory_handle))
            if wholememory_is_intranode_communicator(comm) == False :
                raise ValueError('Multi-node continuous type wholememory does not support host_view. Only supports host_view=false regardless of whether location is host or not.')
        global_size = wholememory_get_total_size(handle.wholememory_handle)
        if global_size % elt_size != 0:
            raise ValueError('global_size=%d not multiple of elt_size=%d' % (global_size, elt_size))
        global_elt_count = global_size // elt_size
        if view_type == VtLocal:
            check_wholememory_error_code(
                wholememory_get_local_memory(&self.c_ptr, &map_size, &map_offset, handle.wholememory_handle))
            if map_size % elt_size != 0 or map_offset % elt_size != 0:
                raise ValueError('map_size=%d, map_offset=%d not multiple of elt_size=%d'
                                 % (map_size, map_offset, elt_size))
            local_elt_count = map_size // elt_size
            local_start = map_offset // elt_size
            self.shape[0] = map_size // elt_size
            self.shape_int64_t[0] = map_size // elt_size
            return local_elt_count, local_start
        elif view_type == VtGlobal:
            check_wholememory_error_code(wholememory_get_global_pointer(&self.c_ptr, handle.wholememory_handle))
            self.shape[0] = global_size // elt_size
            self.shape_int64_t[0] = global_size // elt_size
            global_elt_count
            return global_elt_count, 0
        elif view_type == VtRemote:
            check_wholememory_error_code(wholememory_get_communicator(&comm, handle.wholememory_handle))
            check_wholememory_error_code(wholememory_communicator_get_rank(&world_rank, comm))
            check_wholememory_error_code(wholememory_communicator_get_size(&world_size, comm))
            if target_rank < 0 or target_rank >= world_size:
                raise IndexError('target_rank=%d but world_size=%d' % (target_rank, int(world_size)))
            check_wholememory_error_code(wholememory_get_rank_memory(
                &self.c_ptr, &map_size, &map_offset, target_rank, handle.wholememory_handle))
            if map_size % elt_size != 0 or map_offset % elt_size != 0:
                raise ValueError('target_rank=%d map_size=%d, map_offset=%d not multiple of elt_size=%d'
                                 % (target_rank, map_size, map_offset, elt_size))
            target_elt_count = map_size // elt_size
            target_start = map_offset // elt_size
            self.shape[0] = map_size // elt_size
            self.shape_int64_t[0] = map_size // elt_size
            return target_elt_count, target_start
        else:
            raise ValueError('view type should be VtLocal or VtGlobal or VtRemote')

    def __len__(self):
        return self.shape[0]

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        buffer.buf = self.c_ptr
        buffer.format = 'c'
        buffer.internal = NULL
        buffer.itemsize = self.itemsize
        buffer.len = self.shape[0]
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        buffer.buf = NULL
        buffer.format = 'c'
        buffer.len = 0
        buffer.ndim = 0
        buffer.obj = None
        buffer.shape = NULL
        buffer.strides = NULL

    @property
    def ptr(self):
        return int(<uintptr_t> self.c_ptr)

    @property
    def __cuda_array_interface__(self):
        """See
        https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.interface.html#__array_interface__
        and
        https://numba.pydata.org/numba-doc/dev/cuda/cuda_array_interface.html
        """
        cdef dict intf = {
            "data": (self.ptr, False),
            "shape": (self.shape[0],),
            "strides": None,
            "typestr": self.typestr,
            "version": 2
        }
        return intf

    def __dlpack__(self, stream=None):
        cdef DLManagedTensor * dlm_tensor = \
            <DLManagedTensor *> stdlib.malloc(sizeof(DLManagedTensor))
        cdef DLTensor * dl_tensor = &dlm_tensor.dl_tensor
        dl_tensor.data = self.c_ptr
        dl_tensor.ndim = 1
        dl_tensor.shape = &self.shape_int64_t[0]
        dl_tensor.strides = &self.strides_int64_t[0]
        dl_tensor.byte_offset = 0
        dl_tensor.device.device_type, dl_tensor.device.device_id = self.__dlpack_device__()
        cdef DLDataType * dtype = &dl_tensor.dtype
        if self.data_type == DtInt or self.data_type == DtInt64 \
                or self.data_type == DtInt16 or self.data_type == DtInt8:
            dtype.code = <uint8_t> kDLInt
        elif self.data_type == DtFloat or self.data_type == DtDouble \
                or self.data_type == DtHalf:
            dtype.code = <uint8_t> kDLFloat
        elif self.data_type == DtHalf:
            dtype.code = <uint8_t> kDLBfloat
        else:
            raise ValueError('Invalid data_type')
        dtype.lanes = <uint16_t> 1
        dtype.bits = <uint8_t> (self.itemsize * 8)

        dlm_tensor.manager_ctx = <void *> self
        cpython.Py_INCREF(self)
        dlm_tensor.deleter = deleter
        return cpython.PyCapsule_New(dlm_tensor, 'dltensor', <cpython.PyCapsule_Destructor> &pycapsule_deleter)

    def __dlpack_device__(self):
        if self.device_type == MlHost:
            return (kDLCPU, 0)
        elif self.device_type == MlDevice:
            return (kDLCUDA, self.device_id)
        else:
            raise ValueError('self.device_type=%d' % (int(self.device_type),))

cdef class PyWholeMemoryComm:
    cdef wholememory_comm_t comm_id

    def __cinit__(self):
        self.comm_id = NULL

    def get_c_handle(self):
        return <int64_t> self.comm_id

    def support_type_location(self,
                              WholeMemoryMemoryType memory_type,
                              WholeMemoryMemoryLocation memory_location):
        cdef WholeMemoryErrorCode err_code = int(
            wholememory_communicator_support_type_location(self.comm_id, int(memory_type), int(memory_location)))
        return err_code == Success

    def get_rank(self):
        cdef int world_rank = -1
        check_wholememory_error_code(wholememory_communicator_get_rank(&world_rank, self.comm_id))
        return world_rank
    def get_size(self):
        cdef int world_size = -1
        check_wholememory_error_code(wholememory_communicator_get_size(&world_size, self.comm_id))
        return world_size
    def get_clique_info(self):
        cdef clique_info_t clique_info
        check_wholememory_error_code(wholememory_communicator_get_clique_info(&clique_info,self.comm_id))

        cdef bint is_in_clique = clique_info.is_in_clique > 0

        return is_in_clique,clique_info.clique_first_rank,clique_info.clique_rank,clique_info.clique_rank_num,clique_info.clique_id,clique_info.clique_num

    def barrier(self):
        check_wholememory_error_code(wholememory_communicator_barrier(self.comm_id))

    def get_distributed_backend(self):
        return WholeMemoryDistributedBackend(wholememory_communicator_get_distributed_backend(self.comm_id))

    def set_distributed_backend(self,WholeMemoryDistributedBackend distributed_backend):
        check_wholememory_error_code(wholememory_communicator_set_distributed_backend(self.comm_id,int(distributed_backend)))

cdef class PyWholeMemoryHandle:
    cdef wholememory_handle_t wholememory_handle

    def __cinit__(self):
        self.wholememory_handle = NULL

    def get_c_handle(self):
        return <int64_t> self.wholememory_handle

    def get_communicator(self):
        py_comm = PyWholeMemoryComm()
        check_wholememory_error_code(wholememory_get_communicator(&py_comm.comm_id, self.wholememory_handle))
        return py_comm

    def get_local_communicator(self):
        py_comm = PyWholeMemoryComm()
        check_wholememory_error_code(wholememory_get_local_communicator(&py_comm.comm_id, self.wholememory_handle))
        return py_comm

    def get_cross_communicator(self):
        py_comm = PyWholeMemoryComm()
        check_wholememory_error_code(wholememory_get_cross_communicator(&py_comm.comm_id, self.wholememory_handle))
        return py_comm

    def get_memory_type(self):
        return WholeMemoryMemoryType(wholememory_get_memory_type(self.wholememory_handle))

    def get_memory_location(self):
        return WholeMemoryMemoryLocation(wholememory_get_memory_location(self.wholememory_handle))

    def get_global_flatten_tensor(self,
                                  object import_dlpack_fn,
                                  WholeMemoryDataType data_type,
                                  WholeMemoryMemoryLocation view_from_device,
                                  int view_from_device_id):
        tb = PyWholeMemoryFlattenDlpack()
        tb.set_view_device(view_from_device, view_from_device_id)
        tsize, toffset = tb.get_view(self, data_type, VtGlobal, 0)
        assert toffset == 0
        return import_dlpack_fn(tb), toffset

    def get_local_flatten_tensor(self,
                                 object import_dlpack_fn,
                                 WholeMemoryDataType data_type,
                                 WholeMemoryMemoryLocation view_from_device,
                                 int view_from_device_id):
        tb = PyWholeMemoryFlattenDlpack()
        tb.set_view_device(view_from_device, view_from_device_id)
        tsize, toffset = tb.get_view(self, data_type, VtLocal, 0)
        return import_dlpack_fn(tb), toffset

    def get_all_chunked_flatten_tensor(self,
                                       object import_dlpack_fn,
                                       WholeMemoryDataType data_type,
                                       WholeMemoryMemoryLocation view_from_device,
                                       int view_from_device_id):
        cdef Whole
        cdef int world_rank
        cdef int world_size
        cdef wholememory_comm_t comm
        check_wholememory_error_code(wholememory_get_communicator(&comm, self.wholememory_handle))
        check_wholememory_error_code(wholememory_communicator_get_rank(&world_rank, comm))
        check_wholememory_error_code(wholememory_communicator_get_size(&world_size, comm))
        chunked_tensors = []
        toffsets = []
        for r in range(world_size):
            tb = PyWholeMemoryFlattenDlpack()
            tb.set_view_device(view_from_device, view_from_device_id)
            tsize, toffset = tb.get_view(self, data_type, VtRemote, r)
            chunked_tensors.append(import_dlpack_fn(tb))
            toffsets.append(toffset)
        return chunked_tensors, toffsets

    def from_filelist(self,
                      int64_t memory_offset,
                      int64_t memory_entry_size,
                      int64_t file_entry_size,
                      int round_robin_size,
                      file_list):
        load_wholememory_handle_from_filelist(<int64_t> self.wholememory_handle,
                                              memory_offset,
                                              memory_entry_size,
                                              file_entry_size,
                                              round_robin_size,
                                              file_list)

    def to_file(self,
                int64_t memory_offset,
                int64_t memory_entry_size,
                int64_t file_entry_size,
                file_name):
        store_wholememory_handle_to_file(<int64_t> self.wholememory_handle,
                                         memory_offset,
                                         memory_entry_size,
                                         file_entry_size,
                                         file_name)

cdef class PyWholeMemoryTensorDescription:
    cdef wholememory_tensor_description_t tensor_description

    def __cinit__(self):
        self.tensor_description.dim = 0
        self.tensor_description.dtype = int(0)
        self.tensor_description.storage_offset = 0

    cdef set_by_tensor_desc(self, wholememory_tensor_description_t * td):
        self.tensor_description = td[0]

    def set_dtype(self, WholeMemoryDataType dtype):
        self.tensor_description.dtype = int(dtype)

    def set_shape(self, shape):
        assert 0 < len(shape) < 8
        dim = len(shape)
        self.tensor_description.dim = dim
        for i in range(dim):
            self.tensor_description.sizes[i] = shape[i]

    def set_stride(self, strides):
        assert len(strides) == self.tensor_description.dim
        for i in range(self.tensor_description.dim):
            self.tensor_description.strides[i] = strides[i]

    def set_storage_offset(self, storage_offset):
        self.tensor_description.storage_offset = storage_offset

    @property
    def dtype(self):
        return WholeMemoryDataType(self.tensor_description.dtype)

    def dim(self):
        return self.tensor_description.dim

    @property
    def shape(self):
        ret_shape = tuple([self.tensor_description.sizes[i] for i in range(self.tensor_description.dim)])
        return ret_shape

    def stride(self):
        return tuple([self.tensor_description.strides[i] for i in range(self.dim())])

    def storage_offset(self):
        return self.tensor_description.storage_offset

cdef class WrappedLocalTensor:
    cdef wholememory_tensor_t wm_tensor

    def __cinit__(self):
        self.wm_tensor = NULL

    def __dealloc__(self):
        if self.wm_tensor:
            check_wholememory_error_code(wholememory_destroy_tensor(self.wm_tensor))
            self.wm_tensor = NULL

    def wrap_tensor(self,
                    PyWholeMemoryTensorDescription py_desc,
                    int64_t data_ptr):
        check_wholememory_error_code(wholememory_make_tensor_from_pointer(&self.wm_tensor,
                                                                          <void *> data_ptr,
                                                                          &py_desc.tensor_description))

        return self

    def get_c_handle(self) -> int:
        if self.wm_tensor:
            return <int64_t> self.wm_tensor
        else:
            return 0

cdef class PyWholeMemoryTensor:
    cdef wholememory_tensor_t wholememory_tensor
    cdef wholememory_tensor_description_t tensor_description

    def __cinit__(self):
        self.wholememory_tensor = NULL

    cdef from_c_handle(self,
                       wholememory_tensor_t wm_tensor):
        self.wholememory_tensor = wm_tensor
        self.tensor_description = wholememory_tensor_get_tensor_description(wm_tensor)[0]

    def get_c_handle(self):
        return <int64_t> self.wholememory_tensor

    def get_wholememory_handle(self):
        handle = PyWholeMemoryHandle()
        handle.wholememory_handle = wholememory_tensor_get_memory_handle(self.wholememory_tensor)
        return handle

    @property
    def dtype(self):
        return WholeMemoryDataType(self.tensor_description.dtype)

    def dim(self):
        return self.tensor_description.dim

    @property
    def shape(self):
        if self.dim() == 1:
            return (self.tensor_description.sizes[0],)
        elif self.dim() == 2:
            return (self.tensor_description.sizes[0], self.tensor_description.sizes[1])
        else:
            raise ValueError('self.dim()=%d' % (self.dim(),))

    def stride(self):
        if self.dim() == 1:
            return (self.tensor_description.strides[0],)
        elif self.dim() == 2:
            return (self.tensor_description.strides[0], self.tensor_description.strides[1])
        else:
            raise ValueError('self.dim()=%d' % (self.dim(),))

    def storage_offset(self):
        return self.tensor_description.storage_offset

    def get_local_entry_count(self):
        cdef size_t local_entry_count = 0
        check_wholememory_error_code(wholememory_tensor_get_local_entry_count(&local_entry_count, self.wholememory_tensor))
        return local_entry_count

    def get_local_entry_start(self):
        cdef size_t local_entry_start = 0
        check_wholememory_error_code(wholememory_tensor_get_local_entry_start(&local_entry_start, self.wholememory_tensor))
        return local_entry_start

    def get_sub_tensor(self, starts, ends):
        cdef int64_t start_array[2]
        cdef int64_t end_array[2]
        start_array[0] = starts[0]
        end_array[0] = ends[0]
        if self.dim() == 1:
            pass
        elif self.dim() == 2:
            start_array[1] = starts[1]
            end_array[1] = ends[1]
        else:
            raise ValueError('self.dim()=%d' % (self.dim(),))
        sub_tensor = PyWholeMemoryTensor()
        check_wholememory_error_code(
            wholememory_tensor_get_subtensor(self.wholememory_tensor, start_array, end_array,
                                             &sub_tensor.wholememory_tensor))
        sub_tensor.from_c_handle(sub_tensor.wholememory_tensor)
        return sub_tensor

    def get_tensor_in_window(self,
                             flatten_tensor,
                             int64_t storage_window_offset):
        if self.tensor_description.dim == 1:
            start_indice = max(0, self.tensor_description.storage_offset - storage_window_offset)
            end_indice = min(flatten_tensor.shape[0],
                             self.tensor_description.storage_offset + self.tensor_description.sizes[
                                 0] - storage_window_offset)
            return flatten_tensor[start_indice: end_indice], max(0,
                                                                 storage_window_offset - self.tensor_description.storage_offset)
        elif self.tensor_description.dim == 2:
            embedding_stride = self.tensor_description.strides[0]
            storage_offset0 = self.tensor_description.storage_offset // embedding_stride
            storage_offset1 = self.tensor_description.storage_offset % embedding_stride
            mat_tensor = flatten_tensor.reshape(-1, embedding_stride)
            assert storage_window_offset % self.tensor_description.strides[0] == 0
            vector_start_offset = storage_window_offset // self.tensor_description.strides[0]
            start_indice0 = max(0, storage_offset0 - vector_start_offset)
            end_indice0 = min(mat_tensor.shape[0],
                              storage_offset0 + self.tensor_description.sizes[0] - vector_start_offset)
            start_indice_1 = storage_offset1
            assert mat_tensor.shape[1] >= storage_offset1 + self.tensor_description.sizes[1]
            end_indice_1 = storage_offset1 + self.tensor_description.sizes[1]
            return mat_tensor[start_indice0:end_indice0, start_indice_1:end_indice_1], max(0,
                                                                                           vector_start_offset - storage_offset0)
        else:
            raise ValueError('tensor dim should be 1 or 2')

    def get_local_tensor(self,
                         object import_dlpack_fn,
                         WholeMemoryMemoryLocation view_from_device,
                         int view_from_device_id):
        flatten_tensor, element_offset = self.get_wholememory_handle().get_local_flatten_tensor(import_dlpack_fn,
                                                                                                self.tensor_description.dtype,
                                                                                                view_from_device,
                                                                                                view_from_device_id)
        return self.get_tensor_in_window(flatten_tensor, element_offset)

    def get_global_tensor(self,
                          object import_dlpack_fn,
                          WholeMemoryMemoryLocation view_from_device,
                          int view_from_device_id):
        global_flatten_tensor, _ = self.get_wholememory_handle().get_global_flatten_tensor(import_dlpack_fn,
                                                                                           self.tensor_description.dtype,
                                                                                           view_from_device,
                                                                                           view_from_device_id)
        return self.get_tensor_in_window(global_flatten_tensor, 0)[0]

    def get_all_chunked_tensor(self,
                               object import_dlpack_fn,
                               WholeMemoryMemoryLocation view_from_device,
                               int view_from_device_id):
        chunked_flatten_tensors, element_offsets = self.get_wholememory_handle().get_all_chunked_flatten_tensor(
            import_dlpack_fn,
            self.tensor_description.dtype,
            view_from_device,
            view_from_device_id)
        chunked_tensors = []
        for i in range(len(chunked_flatten_tensors)):
            chunked_tensors.append(self.get_tensor_in_window(chunked_flatten_tensors[i], element_offsets[i])[0])
        return chunked_tensors

    def from_filelist(self, filelist, round_robin_size:int = 0):
        handle = self.get_wholememory_handle()
        strides = self.stride()
        shape = self.shape
        cdef size_t elt_size = wholememory_dtype_get_element_size(self.tensor_description.dtype)

        cdef size_t memory_offset
        cdef size_t memory_entry_size
        cdef size_t file_entry_size
        memory_offset = self.storage_offset() * elt_size
        memory_entry_size = elt_size * strides[0]
        if self.dim() == 1:
            file_entry_size = elt_size
        elif self.dim() == 2:
            file_entry_size = elt_size * shape[1]
        else:
            raise ValueError('tensor dim should be 1 or 2')
        handle.from_filelist(memory_offset, memory_entry_size, file_entry_size, round_robin_size, filelist)

    def to_file(self, filename):
        handle = self.get_wholememory_handle()
        strides = self.stride()
        shape = self.shape
        cdef size_t elt_size = wholememory_dtype_get_element_size(self.tensor_description.dtype)

        cdef size_t memory_offset
        cdef size_t memory_entry_size
        cdef size_t file_entry_size
        memory_offset = self.storage_offset() * elt_size
        memory_entry_size = elt_size * strides[0]
        if self.dim() == 1:
            file_entry_size = elt_size
        elif self.dim() == 2:
            file_entry_size = elt_size * shape[1]
        else:
            raise ValueError('tensor dim should be 1 or 2')
        handle.to_file(memory_offset, memory_entry_size, file_entry_size, filename)

###############################################################################


def create_communicator(PyWholeMemoryUniqueID py_uid, int world_rank, int world_size):
    py_comm = PyWholeMemoryComm()
    check_wholememory_error_code(wholememory_create_communicator(&py_comm.comm_id,
                                                                 py_uid.wholememory_unique_id,
                                                                 world_rank,
                                                                 world_size))
    return py_comm

def destroy_communicator(PyWholeMemoryComm py_comm):
    check_wholememory_error_code(wholememory_destroy_communicator(py_comm.comm_id))

def split_communicator(PyWholeMemoryComm comm,int color,int key):
    py_comm = PyWholeMemoryComm()
    check_wholememory_error_code(wholememory_split_communicator(&py_comm.comm_id,comm.comm_id,color,key))
    return py_comm

def communicator_set_distributed_backend(PyWholeMemoryComm py_comm,WholeMemoryDistributedBackend distributed_backend):
    check_wholememory_error_code(wholememory_communicator_set_distributed_backend(py_comm.comm_id,int(distributed_backend)))

def equal_partition_plan(int64_t entry_count,
                             int world_size):
    cdef size_t per_rank_count
    check_wholememory_error_code(wholememory_equal_entry_partition_plan(&per_rank_count,
                                                                            entry_count,
                                                                            world_size))
    return per_rank_count

def malloc(cython.size_t total_size,
           PyWholeMemoryComm py_comm,
           WholeMemoryMemoryType memory_type,
           WholeMemoryMemoryLocation memory_location,
           cython.size_t data_granularity,
           cython.size_t[:] rank_entry_partition=None):
    handle = PyWholeMemoryHandle()
    cdef size_t* partition_ptr = NULL
    if rank_entry_partition is not None and rank_entry_partition.size > 0:
        partition_ptr = <size_t*>&rank_entry_partition[0]
    check_wholememory_error_code(wholememory_malloc(&handle.wholememory_handle, total_size, py_comm.comm_id,
                                                    int(memory_type), int(memory_location),
                                                    data_granularity, partition_ptr))
    return handle

def free(PyWholeMemoryHandle handle):
    check_wholememory_error_code(wholememory_free(handle.wholememory_handle))

def create_wholememory_array(WholeMemoryDataType dtype,
                             int64_t size,
                             PyWholeMemoryComm comm,
                             WholeMemoryMemoryType mem_type,
                             WholeMemoryMemoryLocation mem_location,
                             cython.size_t[:]  tensor_entry_partition=None):
    wholememory_tensor = PyWholeMemoryTensor()
    wholememory_tensor.tensor_description.dtype = int(dtype)
    wholememory_tensor.tensor_description.storage_offset = 0
    wholememory_tensor.tensor_description.dim = 1
    wholememory_tensor.tensor_description.strides[0] = 1
    wholememory_tensor.tensor_description.sizes[0] = size
    cdef size_t* partition_ptr = NULL
    if tensor_entry_partition is not None and tensor_entry_partition.size > 0:
        partition_ptr = <size_t*>&tensor_entry_partition[0]
    check_wholememory_error_code(wholememory_create_tensor(&wholememory_tensor.wholememory_tensor,
                                                           &wholememory_tensor.tensor_description,
                                                           comm.comm_id,
                                                           int(mem_type),
                                                           int(mem_location),
                                                           partition_ptr))
    return wholememory_tensor

def create_wholememory_matrix(WholeMemoryDataType dtype,
                              int64_t row,
                              int64_t column,
                              int64_t stride,
                              PyWholeMemoryComm comm,
                              WholeMemoryMemoryType mem_type,
                              WholeMemoryMemoryLocation mem_location,
                              cython.size_t[:] tensor_entry_partition=None):
    wholememory_tensor = PyWholeMemoryTensor()
    wholememory_tensor.tensor_description.dtype = int(dtype)
    wholememory_tensor.tensor_description.storage_offset = 0
    wholememory_tensor.tensor_description.dim = 2
    if stride == -1:
        stride = column
    wholememory_tensor.tensor_description.strides[0] = stride
    wholememory_tensor.tensor_description.strides[1] = 1
    wholememory_tensor.tensor_description.sizes[0] = row
    wholememory_tensor.tensor_description.sizes[1] = column
    cdef size_t* partition_ptr = NULL
    if tensor_entry_partition is not None and tensor_entry_partition.size > 0:
        partition_ptr = <size_t*>&tensor_entry_partition[0]
    check_wholememory_error_code(wholememory_create_tensor(&wholememory_tensor.wholememory_tensor,
                                                           &wholememory_tensor.tensor_description,
                                                           comm.comm_id,
                                                           int(mem_type),
                                                           int(mem_location),
                                                           partition_ptr))
    return wholememory_tensor

def create_wholememory_tensor(PyWholeMemoryTensorDescription tensor_description,
                              PyWholeMemoryComm comm,
                              WholeMemoryMemoryType mem_type,
                              WholeMemoryMemoryLocation mem_location,
                              cython.size_t[:] tensor_entry_partition=None):
    if tensor_description.dim() != 1 and tensor_description.dim() != 2:
        raise NotImplementedError('WholeMemory currently only support 1D or 2D tensor')
    if tensor_description.stride()[tensor_description.dim() - 1] != 1:
        raise ValueError('last stride should be 1')
    if tensor_description.storage_offset() != 0:
        raise ValueError('storage_offset be 0 when created')
    wholememory_tensor = PyWholeMemoryTensor()
    wholememory_tensor.tensor_description = tensor_description.tensor_description
    cdef size_t* partition_ptr = NULL
    if tensor_entry_partition is not None and tensor_entry_partition.size > 0:
        partition_ptr = <size_t*>&tensor_entry_partition[0]
    check_wholememory_error_code(wholememory_create_tensor(&wholememory_tensor.wholememory_tensor,
                                                           &wholememory_tensor.tensor_description,
                                                           comm.comm_id,
                                                           int(mem_type),
                                                           int(mem_location),
                                                           partition_ptr))
    return wholememory_tensor

def make_tensor_as_wholememory(PyWholeMemoryTensorDescription tensor_description,
                               int64_t data_ptr):
    if tensor_description.stride()[tensor_description.dim() - 1] != 1:
        raise ValueError('last stride should be 1')
    wholememory_tensor = PyWholeMemoryTensor()
    check_wholememory_error_code(wholememory_make_tensor_from_pointer(&wholememory_tensor.wholememory_tensor,
                                                                      <void *> data_ptr,
                                                                      &tensor_description.tensor_description))
    wholememory_tensor.from_c_handle(wholememory_tensor.wholememory_tensor)
    return wholememory_tensor

def make_handle_as_wholememory(PyWholeMemoryTensorDescription tensor_description,
                               PyWholeMemoryHandle handle):
    if tensor_description.stride()[tensor_description.dim() - 1] != 1:
        raise ValueError('last stride should be 1')
    wholememory_tensor = PyWholeMemoryTensor()
    check_wholememory_error_code(wholememory_make_tensor_from_handle(&wholememory_tensor.wholememory_tensor,
                                                                     handle.wholememory_handle,
                                                                     &tensor_description.tensor_description))
    wholememory_tensor.from_c_handle(wholememory_tensor.wholememory_tensor)
    return wholememory_tensor

def destroy_wholememory_tensor(PyWholeMemoryTensor wholememory_tensor):
    check_wholememory_error_code(wholememory_destroy_tensor(wholememory_tensor.wholememory_tensor))

def fork_get_gpu_count():
    return fork_get_device_count()

cpdef load_wholememory_handle_from_filelist(int64_t wholememory_handle_int_ptr,
                                            int64_t memory_offset,
                                            int64_t memory_entry_size,
                                            int64_t file_entry_size,
                                            int round_robin_size,
                                            file_list):
    cdef const char ** filenames
    cdef int num_files = len(file_list)
    cdef int i

    filenames = <const char**> stdlib.malloc(num_files * sizeof(char *))

    try:
        for i in range(num_files):
            filenames[i] = PyUnicode_AsUTF8(file_list[i])

        check_wholememory_error_code(wholememory_load_from_file(
            <wholememory_handle_t> <int64_t> wholememory_handle_int_ptr,
            memory_offset,
            memory_entry_size,
            file_entry_size,
            filenames,
            num_files,
            round_robin_size))
    finally:
        stdlib.free(filenames)

cpdef store_wholememory_handle_to_file(int64_t wholememory_handle_int_ptr,
                                       int64_t memory_offset,
                                       int64_t memory_entry_size,
                                       int64_t file_entry_size,
                                       file_name):
    check_wholememory_error_code(wholememory_store_to_file(
        <wholememory_handle_t> <int64_t> wholememory_handle_int_ptr,
        memory_offset,
        memory_entry_size,
        file_entry_size,
        PyUnicode_AsUTF8(file_name)))

cdef extern from "wholememory/wholememory_op.h":
    cdef wholememory_error_code_t wholememory_gather(wholememory_tensor_t wholememory_tensor,
                                                     wholememory_tensor_t indices_tensor,
                                                     wholememory_tensor_t output_tensor,
                                                     wholememory_env_func_t * p_env_fns,
                                                     void * stream)

    cdef wholememory_error_code_t wholememory_scatter(wholememory_tensor_t input_tensor,
                                                      wholememory_tensor_t indices_tensor,
                                                      wholememory_tensor_t wholememory_tensor,
                                                      wholememory_env_func_t * p_env_fns,
                                                      void * stream)
    cdef wholememory_error_code_t wholememory_env_test_op(wholememory_tensor_t input_tensor,
                                                          wholememory_tensor_t output_fixed_tensor,
                                                          void *output_variable_device_tensor_handle,
                                                          void *output_variable_pinned_tensor_handle,
                                                          void *output_variable_host_tensor_handle,
                                                          int64_t output_variable_entry_count,
                                                          wholememory_env_func_t *p_env_fns,
                                                          void *stream)


cpdef void wholememory_gather_op(PyWholeMemoryTensor wholememory_tensor,
                                 WrappedLocalTensor indices_tensor,
                                 WrappedLocalTensor output_tensor,
                                 int64_t p_env_fns_int,
                                 int64_t stream_int):
    check_wholememory_error_code(wholememory_gather(<wholememory_tensor_t> <int64_t> wholememory_tensor.get_c_handle(),
                                                    <wholememory_tensor_t> <int64_t> indices_tensor.get_c_handle(),
                                                    <wholememory_tensor_t> <int64_t> output_tensor.get_c_handle(),
                                                    <wholememory_env_func_t *> p_env_fns_int,
                                                    <void *> stream_int))

cpdef void wholememory_scatter_op(WrappedLocalTensor input_tensor,
                                  WrappedLocalTensor indices_tensor,
                                  PyWholeMemoryTensor wholememory_tensor,
                                  int64_t p_env_fns_int,
                                  int64_t stream_int):
    check_wholememory_error_code(wholememory_scatter(<wholememory_tensor_t> <int64_t> input_tensor.get_c_handle(),
                                                     <wholememory_tensor_t> <int64_t> indices_tensor.get_c_handle(),
                                                     <wholememory_tensor_t> <int64_t> wholememory_tensor.get_c_handle(),
                                                     <wholememory_env_func_t *> p_env_fns_int,
                                                     <void *> stream_int))

cpdef void wholememory_env_test_cython_op(WrappedLocalTensor input,
                                          WrappedLocalTensor output,
                                          int64_t output_variable_device_tensor_handle,
                                          int64_t output_variable_pinned_tensor_handle,
                                          int64_t output_variable_host_tensor_handle,
                                          int64_t output_variable_entry_count,
                                          int64_t p_env_fns_int,
                                          int64_t stream_int):
    check_wholememory_error_code(wholememory_env_test_op(<wholememory_tensor_t> <int64_t> input.get_c_handle(),
                                                         <wholememory_tensor_t> <int64_t> output.get_c_handle(),
                                                         <void *> output_variable_device_tensor_handle,
                                                         <void *> output_variable_pinned_tensor_handle,
                                                         <void *> output_variable_host_tensor_handle,
                                                         output_variable_entry_count,
                                                         <wholememory_env_func_t *> p_env_fns_int,
                                                         <void *> stream_int))
    return

cdef extern from "wholememory/wholegraph_op.h":
    cdef wholememory_error_code_t wholegraph_csr_unweighted_sample_without_replacement(
            wholememory_tensor_t wm_csr_row_ptr_tensor,
            wholememory_tensor_t wm_csr_col_ptr_tensor,
            wholememory_tensor_t center_nodes_tensor,
            int max_sample_count,
            wholememory_tensor_t output_sample_offset_tensor,
            void * output_dest_memory_context,
            void * output_center_localid_memory_context,
            void * output_edge_gid_memory_context,
            unsigned long long random_seed,
            wholememory_env_func_t * p_env_fns,
            void * stream)

    cdef wholememory_error_code_t wholegraph_csr_weighted_sample_without_replacement(
            wholememory_tensor_t wm_csr_row_ptr_tensor,
            wholememory_tensor_t wm_csr_col_ptr_tensor,
            wholememory_tensor_t wm_csr_weight_ptr_tensor,
            wholememory_tensor_t center_nodes_tensor,
            int max_sample_count,
            wholememory_tensor_t output_sample_offset_tensor,
            void * output_dest_memory_context,
            void * output_center_localid_memory_context,
            void * output_edge_gid_memory_context,
            unsigned long long random_seed,
            wholememory_env_func_t * p_env_fns,
            void * stream)

    cdef wholememory_error_code_t generate_random_positive_int_cpu(
            int64_t random_seed,
            int64_t subsequence,
            wholememory_tensor_t output)

    cdef wholememory_error_code_t generate_exponential_distribution_negative_float_cpu(
            int64_t random_seed,
            int64_t subsequence,
            wholememory_tensor_t output)

cpdef void csr_unweighted_sample_without_replacement(
        PyWholeMemoryTensor wm_csr_row_ptr_tensor,
        PyWholeMemoryTensor wm_csr_col_ptr_tensor,
        WrappedLocalTensor center_nodes_tensor,
        int max_sample_count,
        WrappedLocalTensor output_sample_offset_tensor,
        int64_t output_dest_memory_handle,
        int64_t output_center_localid_memory_handle,
        int64_t output_edge_gid_memory_handle,
        unsigned long long random_seed,
        int64_t p_env_fns_int,
        int64_t stream_int
):
    check_wholememory_error_code(wholegraph_csr_unweighted_sample_without_replacement(
        <wholememory_tensor_t> <int64_t> wm_csr_row_ptr_tensor.get_c_handle(),
        <wholememory_tensor_t> <int64_t> wm_csr_col_ptr_tensor.get_c_handle(),
        <wholememory_tensor_t> <int64_t> center_nodes_tensor.get_c_handle(),
        max_sample_count,
        <wholememory_tensor_t> <int64_t> output_sample_offset_tensor.get_c_handle(),
        <void *> output_dest_memory_handle,
        <void *> output_center_localid_memory_handle,
        <void *> output_edge_gid_memory_handle,
        random_seed,
        <wholememory_env_func_t *> p_env_fns_int,
        <void *> stream_int))

cpdef void csr_weighted_sample_without_replacement(
        PyWholeMemoryTensor wm_csr_row_ptr_tensor,
        PyWholeMemoryTensor wm_csr_col_ptr_tensor,
        PyWholeMemoryTensor wm_csr_weight_ptr_tensor,
        WrappedLocalTensor center_nodes_tensor,
        int max_sample_count,
        WrappedLocalTensor output_sample_offset_tensor,
        int64_t output_dest_memory_handle,
        int64_t output_center_localid_memory_handle,
        int64_t output_edge_gid_memory_handle,
        unsigned long long random_seed,
        int64_t p_env_fns_int,
        int64_t stream_int
):
    check_wholememory_error_code(wholegraph_csr_weighted_sample_without_replacement(
        <wholememory_tensor_t> <int64_t> wm_csr_row_ptr_tensor.get_c_handle(),
        <wholememory_tensor_t> <int64_t> wm_csr_col_ptr_tensor.get_c_handle(),
        <wholememory_tensor_t> <int64_t> wm_csr_weight_ptr_tensor.get_c_handle(),
        <wholememory_tensor_t> <int64_t> center_nodes_tensor.get_c_handle(),
        max_sample_count,
        <wholememory_tensor_t> <int64_t> output_sample_offset_tensor.get_c_handle(),
        <void *> output_dest_memory_handle,
        <void *> output_center_localid_memory_handle,
        <void *> output_edge_gid_memory_handle,
        random_seed,
        <wholememory_env_func_t *> p_env_fns_int,
        <void *> stream_int))

cpdef void host_generate_random_positive_int(
        int64_t random_seed,
        int64_t subsequence,
        WrappedLocalTensor output
):
    check_wholememory_error_code(generate_random_positive_int_cpu(
        random_seed,
        subsequence,
        <wholememory_tensor_t> <int64_t> output.get_c_handle()
    ))

cpdef void host_generate_exponential_distribution_negative_float(
        int64_t random_seed,
        int64_t subsequence,
        WrappedLocalTensor output
):
    check_wholememory_error_code(generate_exponential_distribution_negative_float_cpu(
        random_seed,
        subsequence,
        <wholememory_tensor_t> <int64_t> output.get_c_handle()
    ))


cdef extern from "wholememory/graph_op.h":
    cdef wholememory_error_code_t graph_append_unique(wholememory_tensor_t target_nodes_tensor,
                                                      wholememory_tensor_t neighbor_nodes_tensor,
                                                      void * output_unique_node_memory_context,
                                                      wholememory_tensor_t output_neighbor_raw_to_unique_mapping_tensor,
                                                      wholememory_env_func_t * p_env_fns,
                                                      void * stream)

    cdef wholememory_error_code_t csr_add_self_loop(wholememory_tensor_t csr_row_ptr_tensor,
                                                    wholememory_tensor_t csr_col_ptr_tensor,
                                                    wholememory_tensor_t output_csr_row_ptr_tensor,
                                                    wholememory_tensor_t output_csr_col_ptr_tensor,
                                                    void * stream)


cpdef void append_unique(
        WrappedLocalTensor target_node_tensor,
        WrappedLocalTensor neighbor_node_tensor,
        int64_t output_unique_node_memory_handle,
        WrappedLocalTensor output_neighbor_raw_to_unique_mapping_tensor,
        int64_t p_env_fns_int,
        int64_t stream_int):
    check_wholememory_error_code(graph_append_unique(
        <wholememory_tensor_t> <int64_t> target_node_tensor.get_c_handle(),
        <wholememory_tensor_t> <int64_t> neighbor_node_tensor.get_c_handle(),
        <void *> output_unique_node_memory_handle,
        <wholememory_tensor_t> <int64_t> output_neighbor_raw_to_unique_mapping_tensor.get_c_handle(),
        <wholememory_env_func_t *> p_env_fns_int,
        <void *> stream_int
    ))

cpdef void add_csr_self_loop(
        WrappedLocalTensor csr_row_ptr_tensor,
        WrappedLocalTensor csr_col_ptr_tensor,
        WrappedLocalTensor csr_row_ptr_self_tensor,
        WrappedLocalTensor csr_col_ptr_self_tensor,
        int64_t stream_int):
    check_wholememory_error_code(csr_add_self_loop(
        <wholememory_tensor_t> <int64_t> csr_row_ptr_tensor.get_c_handle(),
        <wholememory_tensor_t> <int64_t> csr_col_ptr_tensor.get_c_handle(),
        <wholememory_tensor_t> <int64_t> csr_row_ptr_self_tensor.get_c_handle(),
        <wholememory_tensor_t> <int64_t> csr_col_ptr_self_tensor.get_c_handle(),
        <void *> stream_int))
