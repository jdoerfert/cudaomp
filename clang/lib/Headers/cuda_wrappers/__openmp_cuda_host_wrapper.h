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

#include "cuda_types.h"
// This is CUDA specific, needed for types.
/*
 * NVIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2010-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * NVIDIA_COPYRIGHT_END
 */

#ifndef fatbinary_INCLUDED
#define fatbinary_INCLUDED

/* 
 * This is the fat binary header structure. 
 * Because all layout information is contained in all the structures, 
 * it is both forward and backward compatible. 
 * A new driver can interpret an old binary 
 * as it will not address fields that are present in the current version. 
 * An old driver can, for minor version differences, 
 * still interpret a new binary, 
 * as the new features in the binary will be ignored by the driver.
 *
 * This is the top level type for the binary format. 
 * It points to a fatBinaryHeader structure. 
 * It is followed by a number of code binaries.
 * The structures must be 8-byte aligned, 
 * and are the same on both 32bit and 64bit platforms.
 *
 * The details of the format for the binaries that follow the header
 * are in a separate internal header.
 */

typedef struct fatBinaryHeader * computeFatBinaryFormat_t;
typedef const struct fatBinaryHeader * computeFatBinaryFormat_ct;

/* ensure 8-byte alignment */
#if defined(__GNUC__)
#define fatbinary_ALIGN_(n) __attribute__((aligned(n)))
#elif defined(_WIN32)
#define fatbinary_ALIGN_(n) __declspec(align(n))
#else
#error !! UNSUPPORTED COMPILER !!
#endif

/* Magic numbers */
#define FATBIN_MAGIC 0xBA55ED50U
#define OLD_STYLE_FATBIN_MAGIC 0x1EE55A01U

#define FATBIN_VERSION 0x0001U

/*
 * This is the fat binary header structure. 
 * The 'magic' field holds the magic number. 
 * A magic of OLD_STYLE_FATBIN_MAGIC indicates an old style fat binary. 
 * Because old style binaries are in little endian, we can just read 
 * the magic in a 32 bit container for both 32 and 64 bit platforms. 
 * The 'version' fields holds the fatbin version.
 * It should be the goal to never bump this version. 
 * The headerSize holds the size of the header (must be multiple of 8).
 * The 'fatSize' fields holds the size of the entire fat binary, 
 * excluding this header. It must be a multiple of 8.
 */
struct fatbinary_ALIGN_(8) fatBinaryHeader
{
  unsigned int           magic;
  unsigned short         version;
  unsigned short         headerSize;
  unsigned long long int fatSize;
};

/* Code kinds supported by the driver */
typedef enum {
  FATBIN_KIND_PTX      = 0x0001,
  FATBIN_KIND_ELF      = 0x0002,
  FATBIN_KIND_OLDCUBIN = 0x0004, /* old format no longer generated */
  FATBIN_KIND_IR       = 0x0008, /* NVVM IR */
} fatBinaryCodeKind;

#endif /* fatbinary_INCLUDED */

#include <cstdint>
#include <cstdio>
#include <omp.h>

#define DEBUGP(...) printf(__VA_ARGS__)

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
                                            size_t __shared_memory,
                                            void* __stream_ptr) {
  DEBUGP("===> __cudaPushCallConfiguration\n");
  __omp_kernel_t &__kernel = __current_kernel;
  __kernel.__stream = __stream_ptr;
  __kernel.__grid_size = __grid_size;
  __kernel.__block_size = __block_size;
  __kernel.__shared_memory = __shared_memory;
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

// TODO
inline cudaError_t cudaDeviceSynchronize() {
  DEBUGP("===> TODO cudaDeviceSynchronize\n");
  return cudaError_t(0);
}

// TODO, fix tgt_kernel_synchronize
inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  DEBUGP("===> TODO FIX cudaStreamSynchronize\n");
  __tgt_kernel_synchronize(omp_get_default_device(), stream);
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
