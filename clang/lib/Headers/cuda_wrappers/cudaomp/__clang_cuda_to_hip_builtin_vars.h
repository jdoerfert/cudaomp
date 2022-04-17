/*===---- cuda_to_hip_builtin_vars.h - CUDA built-in variables -------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __CUDA_TO_HIP_BUILTIN_VARS_H
#define __CUDA_TO_HIP_BUILTIN_VARS_H

extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_local_id(uint);
extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_group_id(uint);
extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_local_size(uint);
extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_num_groups(uint);

// Forward declares from vector_types.h.
struct uint3;
struct dim3;

// The file implements built-in CUDA variables using __declspec(property).
// https://msdn.microsoft.com/en-us/library/yhfk0thd.aspx
// All read accesses of built-in variable fields get converted into calls to a
// getter function which in turn calls the appropriate builtin to fetch the
// value.
//
// Example:
//    int x = threadIdx.x;
// IR output:
//  %0 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3
// PTX output:
//  mov.u32     %r2, %tid.x;

#define __CUDA_DEVICE_BUILTIN(FIELD, INTRINSIC)                                \
  __declspec(property(get = __fetch_builtin_##FIELD)) unsigned int FIELD;      \
  static inline __attribute__((always_inline))                                 \
      __attribute__((device)) unsigned int __fetch_builtin_##FIELD(void) {     \
    return INTRINSIC;                                                          \
  }

#if __cplusplus >= 201103L
#define __DELETE =delete
#else
#define __DELETE
#endif

// Make sure nobody can create instances of the special variable types.  nvcc
// also disallows taking address of special variables, so we disable address-of
// operator as well.
#define __CUDA_DISALLOW_BUILTINVAR_ACCESS(TypeName)                            \
  __attribute__((device)) TypeName() __DELETE;                                 \
  __attribute__((device)) TypeName(const TypeName &) __DELETE;                 \
  __attribute__((device)) void operator=(const TypeName &) const __DELETE;     \
  __attribute__((device)) TypeName *operator&() const __DELETE

struct __cuda_builtin_threadIdx_t {
  __CUDA_DEVICE_BUILTIN(x,__ockl_get_local_id(0));
  __CUDA_DEVICE_BUILTIN(y,__ockl_get_local_id(1));
  __CUDA_DEVICE_BUILTIN(z,__ockl_get_local_id(2));
  // threadIdx should be convertible to uint3 (in fact in nvcc, it *is* a
  // uint3).  This function is defined after we pull in vector_types.h.
  __attribute__((device)) operator dim3() const;
  __attribute__((device)) operator uint3() const;

private:
  __CUDA_DISALLOW_BUILTINVAR_ACCESS(__cuda_builtin_threadIdx_t);
};

struct __cuda_builtin_blockIdx_t {
  __CUDA_DEVICE_BUILTIN(x,__ockl_get_group_id(0));
  __CUDA_DEVICE_BUILTIN(y,__ockl_get_group_id(1));
  __CUDA_DEVICE_BUILTIN(z,__ockl_get_group_id(2));
  // blockIdx should be convertible to uint3 (in fact in nvcc, it *is* a
  // uint3).  This function is defined after we pull in vector_types.h.
  __attribute__((device)) operator dim3() const;
  __attribute__((device)) operator uint3() const;

private:
  __CUDA_DISALLOW_BUILTINVAR_ACCESS(__cuda_builtin_blockIdx_t);
};

struct __cuda_builtin_blockDim_t {
  __CUDA_DEVICE_BUILTIN(x,__ockl_get_local_id(0));
  __CUDA_DEVICE_BUILTIN(y,__ockl_get_local_id(1));
  __CUDA_DEVICE_BUILTIN(z,__ockl_get_local_id(2));
  // blockDim should be convertible to dim3 (in fact in nvcc, it *is* a
  // dim3).  This function is defined after we pull in vector_types.h.
  __attribute__((device)) operator dim3() const;
  __attribute__((device)) operator uint3() const;

private:
  __CUDA_DISALLOW_BUILTINVAR_ACCESS(__cuda_builtin_blockDim_t);
};

struct __cuda_builtin_gridDim_t {
  __CUDA_DEVICE_BUILTIN(x,__ockl_get_num_groups(0));
  __CUDA_DEVICE_BUILTIN(y,__ockl_get_num_groups(1));
  __CUDA_DEVICE_BUILTIN(z,__ockl_get_num_groups(2));
  // gridDim should be convertible to dim3 (in fact in nvcc, it *is* a
  // dim3).  This function is defined after we pull in vector_types.h.
  __attribute__((device)) operator dim3() const;
  __attribute__((device)) operator uint3() const;

private:
  __CUDA_DISALLOW_BUILTINVAR_ACCESS(__cuda_builtin_gridDim_t);
};

#define __CUDA_BUILTIN_VAR                                                     \
  extern const __attribute__((device)) __attribute__((weak))
__CUDA_BUILTIN_VAR __cuda_builtin_threadIdx_t threadIdx;
__CUDA_BUILTIN_VAR __cuda_builtin_blockIdx_t blockIdx;
__CUDA_BUILTIN_VAR __cuda_builtin_blockDim_t blockDim;
__CUDA_BUILTIN_VAR __cuda_builtin_gridDim_t gridDim;

// warpSize should translate to read of %WARP_SZ but there's currently no
// builtin to do so. According to PTX v4.2 docs 'to date, all target
// architectures have a WARP_SZ value of 32'.
__attribute__((device)) const int warpSize = 32;

#undef __CUDA_DEVICE_BUILTIN
#undef __CUDA_BUILTIN_VAR
#undef __CUDA_DISALLOW_BUILTINVAR_ACCESS
#undef __DELETE

#endif /* __CUDA_TO_HIP_BUILTIN_VARS_H */
