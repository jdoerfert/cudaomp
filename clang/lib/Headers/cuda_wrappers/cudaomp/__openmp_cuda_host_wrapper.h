/*===---- __openmp_cuda_host_wrapper.h - CUDA host support for OpenMP ------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __OPENMP_CUDA_HOST_WRAPPER_H__
#define __OPENMP_CUDA_HOST_WRAPPER_H__

#include "fatbinary_cuda.h"
#include "internal_types.h"

#include <cstdint>
#include <cstdio>
#include <omp.h>

extern "C" {
int __tgt_kernel(int64_t device_id, const void *host_ptr, void **args,
                 int32_t grid_dim_x, int32_t grid_dim_y, int32_t grid_dim_z,
                 int32_t block_dim_x, int32_t block_dim_y, int32_t block_dim_z,
                 size_t shared_mem, void *stream);
int __tgt_kernel_synchronize(int64_t device_id, void *Stream);

struct __omp_kernel_t {
  dim3 __grid_size;
  dim3 __block_size;
  size_t __shared_memory;

  void* __stream;
};

static __omp_kernel_t __current_kernel;
#pragma omp threadprivate(__current_kernel);

struct __tgt_offload_entry {
  void *addr;   // Pointer to the offload entry info (function or global)
  char *name;   // Name of the function or global
  size_t size;  // Size of the entry info (0 if it is a function)
  int32_t flags; // Flags associated with the entry, e.g. 'link'.
  int32_t reserved; // Reserved, to be used by the runtime library.
};

/// This struct is a record of the device image information
struct __tgt_device_image {
  void *ImageStart;                  // Pointer to the target code start
  void *ImageEnd;                    // Pointer to the target code end
  __tgt_offload_entry *EntriesBegin; // Begin of table with all target entries
  __tgt_offload_entry *EntriesEnd;   // End of table (non inclusive)
};

/// This struct is a record of all the host code that may be offloaded to a
/// target.
struct __tgt_bin_desc {
  int32_t NumDeviceImages;           // Number of device types supported
  __tgt_device_image *DeviceImages;  // Array of device images (1 per dev. type)
  __tgt_offload_entry *HostEntriesBegin; // Begin of table with all host entries
  __tgt_offload_entry *HostEntriesEnd;   // End of table (non inclusive)
};

void __tgt_register_lib(__tgt_bin_desc *desc);
void __tgt_unregister_lib(__tgt_bin_desc *desc);

// This is CUDA specific, fatbin wrapper.
struct __cuda_fatbin_wrapper_t {
  int32_t magic;
  int32_t version;
  void *gpu_binary;
  void *data_ptr;
};

enum OMPTgtExecModeFlags : int8_t {
  OMP_TGT_EXEC_MODE_GENERIC = 1 << 0,
  OMP_TGT_EXEC_MODE_SPMD = 1 << 1,
  OMP_TGT_EXEC_MODE_GENERIC_SPMD =
      OMP_TGT_EXEC_MODE_GENERIC | OMP_TGT_EXEC_MODE_SPMD,
  OMP_TGT_EXEC_MODE_CUDA = 1 << 2,
};

#define MAX_OFFLOAD_ENTRIES 256

// TODO: do we need support for multiple device images?
static struct __tgt_device_image __device_image;
static struct __tgt_bin_desc __bin_desc;
static struct __tgt_offload_entry __offload_entries[MAX_OFFLOAD_ENTRIES];
static unsigned __offload_entries_counter = 0;

inline unsigned __cudaPushCallConfiguration(dim3 __grid_size, dim3 __block_size,
                                            size_t __shared_memory = 0,
                                            cudaStream_t __stream = 0) {
  DEBUGP("===> __cudaPushCallConfiguration\n");
  __omp_kernel_t &__kernel = __current_kernel;
  __kernel.__grid_size = __grid_size;
  __kernel.__block_size = __block_size;
  __kernel.__shared_memory = __shared_memory;
  __kernel.__stream = __stream;
  return 0;
}

inline unsigned __cudaPopCallConfiguration(dim3 *__grid_size,
                                           dim3 *__block_size,
                                           size_t *__shared_memory,
                                           void *__stream) {
  DEBUGP("===> __cudaPopCallConfiguration\n");
  __omp_kernel_t &__kernel = __current_kernel;
  *__grid_size = __kernel.__grid_size;
  *__block_size = __kernel.__block_size;
  *__shared_memory = __kernel.__shared_memory;
  *((void**)__stream) = __kernel.__stream;
  return 0;
}

inline cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim,
                                    dim3 blockDim, void **args,
                                    size_t sharedMem = 0, cudaStream_t stream = 0) {
  __omp_kernel_t &__kernel = __current_kernel;

  DEBUGP("===> overloaded cudaLaunchKernel grid [%d,%d,%d] blocks, block [%d,%d,%d] "
         "threads\n",
         gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
  int rv = __tgt_kernel(omp_get_default_device(), func, args, gridDim.x, gridDim.y,
               gridDim.z, blockDim.x, blockDim.y, blockDim.z, sharedMem,
               stream);
  return cudaError_t(rv);
}


void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                            char *deviceFun, const char *deviceName,
                            int thread_limit, uint3 *tid, uint3 *bid,
                            dim3 *bDim, dim3 *gDim, int *wSize) {
  DEBUGP("===> cudaRegisterFunction cnt %d\n", __offload_entries_counter);
  DEBUGP("===> deviceName %s\n", deviceFun);
  // Assign is fine, global symbols.
  __offload_entries[__offload_entries_counter].addr = (void *)hostFun;
  __offload_entries[__offload_entries_counter].name = deviceFun;
  __offload_entries[__offload_entries_counter].size = 0;
  // TODO: using flags to communicate CUDA exec mode. Is there a better way?
  __offload_entries[__offload_entries_counter].flags = 0;
  __offload_entries[__offload_entries_counter].reserved = OMP_TGT_EXEC_MODE_CUDA;
  ++__offload_entries_counter;
}

// TODO: return a pointer to an internal handle.
void **__cudaRegisterFatBinary(void *fatCubin) {
  struct __cuda_fatbin_wrapper_t *header = (struct __cuda_fatbin_wrapper_t *)fatCubin;
  DEBUGP("===> magic %x\n", header->magic);
  struct fatBinaryHeader *header2 = (struct fatBinaryHeader *)header->gpu_binary;
  DEBUGP("===> magic2 %x\n", header2->magic);
  DEBUGP("===> headersize %d\n", header2->headerSize);
  DEBUGP("===> fatSize %llu\n", header2->fatSize);
  char *ImageStart = (char *)header->gpu_binary;
  //int offset;
  //for (offset = 0; offset < 128; offset++) {
  //  DEBUGP("===> ImageStart[%d] %x\n", offset, ImageStart[offset]);
  //  if (ImageStart[offset] == 0x7F && ImageStart[offset+1] == 'E' &&
  //  ImageStart[offset+2] == 'L' && ImageStart[offset+3] == 'F')
  //    break;
  //}
  // +80 bytes is reverse engineered from the fatbin produced file.
  ImageStart += 80;
  __device_image.ImageStart = ImageStart;
  __device_image.ImageEnd = ImageStart + header2->fatSize;

  return nullptr;
}

void __cudaRegisterFatBinaryEnd(void **fatCubinHandle){
  DEBUGP("===> __cudaRegisterFatBinaryEnd\n");
  // The descriptor and call to tgt_register_lib should be done in this
  // end function, after cuda_register_globals registers functions, vars.
  __device_image.EntriesBegin = &__offload_entries[0];
  __device_image.EntriesEnd = &__offload_entries[__offload_entries_counter];
  __bin_desc = {1, &__device_image, &__offload_entries[0],
                            &__offload_entries[__offload_entries_counter]};
  __tgt_register_lib(&__bin_desc);
}

void __cudaUnregisterFatBinary(void *handle){
  DEBUGP("===> __cudaUnregisterFatBinary\n");
  __tgt_unregister_lib(&__bin_desc);
}

void __cudaRegisterVar(
        void **fatCubinHandle,
        char  *hostVar,
        char  *deviceVar,
  const char  *nameVar,
        int    ext,
        int    size,
        int    constant,
        int    global
) {
  DEBUGP("===> __cudaRegisterVar\n");
/*   DEBUGP("           fatCubinHandle is %p \n", fatCubinHandle);
  DEBUGP("           hostVar is %p \n", hostVar);
  DEBUGP("           deviceVar is %s \n", deviceVar);
  DEBUGP("           nameVar is %s \n", nameVar);
  DEBUGP("           ext is %d \n", ext);
  DEBUGP("           size is %d \n", size);
  DEBUGP("           constant is %d \n", constant);
  DEBUGP("           global is %d \n", global); */
  __offload_entries[__offload_entries_counter].addr = (void *)hostVar;
  __offload_entries[__offload_entries_counter].name = deviceVar;
  __offload_entries[__offload_entries_counter].size = size;
  __offload_entries[__offload_entries_counter].flags = 0;
  __offload_entries[__offload_entries_counter].reserved = OMP_TGT_EXEC_MODE_CUDA;
  ++__offload_entries_counter;
}

}

#endif
