/*===---- __openmp_cuda_device_wrapper.h - CUDA device support for OpenMP ---===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __OPENMP_CUDA_DEVICE_WRAPPER_H__
#define __OPENMP_CUDA_DEVICE_WRAPPER_H__

#ifdef __cplusplus
extern "C" {
#endif

#define _NVVM_GETTER(DIM) \
  __device__ uint32_t __kmpc_get_hardware_thread_id_in_block_##DIM(); \
  __device__ __attribute__((always_inline,flatten)) inline \
  int __nvvm_read_ptx_sreg_tid_##DIM() { \
    return __kmpc_get_hardware_thread_id_in_block_##DIM(); \
  } \
  __device__ uint32_t __kmpc_get_hardware_num_threads_in_block_##DIM(); \
  __device__ __attribute__((always_inline,flatten)) inline \
  int __nvvm_read_ptx_sreg_ntid_##DIM() { \
    return __kmpc_get_hardware_num_threads_in_block_##DIM(); \
  } \
  __device__ uint32_t __kmpc_get_hardware_block_id_in_grid_##DIM(); \
  __device__ __attribute__((always_inline,flatten)) inline \
  int __nvvm_read_ptx_sreg_ctaid_##DIM() { \
    return __kmpc_get_hardware_block_id_in_grid_##DIM(); \
  } \
  __device__ uint32_t __kmpc_get_hardware_num_blocks_in_grid_##DIM(); \
  __device__ __attribute__((always_inline,flatten)) inline \
  int __nvvm_read_ptx_sreg_nctaid_##DIM() { \
    return __kmpc_get_hardware_num_blocks_in_grid_##DIM(); \
  }

_NVVM_GETTER(x)
_NVVM_GETTER(y)
_NVVM_GETTER(z)

//extern "C" __attribute__((used,retain,device)) const int __omp_rtl_debug_kind = 0;
//extern "C" __attribute__((used,retain,device)) const int __omp_rtl_assume_teams_oversubscription = 0;
//extern "C" __attribute__((used,retain,device)) const int __omp_rtl_assume_threads_oversubscription = 0;
//extern "C" __attribute__((used,retain,device)) const int __omp_rtl_assume_no_thread_state = 0;

#ifdef __cplusplus
}
#endif

#endif
