#ifndef __CUDA_H__
#define __CUDA_H__

#include <assert.h>
#include <atomic>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "cuda_types.h"
#include "__openmp_cuda_host_wrapper.h"

#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __managed__ __attribute__((managed))

//typedef cudaStream_t unsigned = 0;

static int *kernels = nullptr;
static std::atomic<unsigned long> num_kernels = {0};
static std::atomic<unsigned long> synced_kernels = {0};

//#pragma omp begin declare target
//__constant__ dim3 grid;
//__constant__ dim3 block;
//#pragma omp end declare target

// Keeps track of the last error produced
cudaError_t lastError = cudaSuccess;

cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {

  if (device >= omp_get_num_devices()) {

    lastError = cudaErrorInvalidDevice;
    return cudaErrorInvalidDevice;
  }
  if (device != omp_get_default_device()) {
    omp_set_default_device(device);
  }

  // set some of the properties
  strncpy(prop->name, "OpenMP device", 256);

  lastError = cudaSuccess;
  return cudaSuccess;
}

// Returns the last error that has been produced and resets it to cudaSuccess
cudaError_t cudaGetLastError() {
  cudaError_t tempError = lastError;
  lastError = cudaSuccess;
  return tempError;
}

// Returns the laste error from a runtime call
cudaError_t cudaPeekAtLastError() { return lastError; }

const char *cudaGetErrorString(cudaError_t error) {
  switch (error) {
  case cudaSuccess:
    return "Success!";
  case cudaErrorInvalidValue:
    return "One or more of the parameters passed to the API call is not within "
           "an acceptable range of values";
  case cudaErrorMemoryAllocation:
    return "The API call failed because it was unable to allocate enough "
           "memory to perform the requested operation";
  case cudaErrorNoDevice:
    return "No CUDA-capable devices were detected by the installed CUDA driver";
  case cudaErrorInvalidDevice:
    return "The device ordinal supplied by the user does not correspond to a "
           "valid CUDA device or the action requested is invalid for the "
           "specified device";
  default:
    return "Unrecognized Error Code!";
  }
  return nullptr;
}

/// used in cudaMemcpy to specify the copy direction
enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4
};

cudaError_t cudaMalloc(void **devPtr, size_t size) {
  int num_devices = omp_get_num_devices();
  if (size < 0 || num_devices < 1) {
    return cudaErrorInvalidValue;
  }
  *devPtr = omp_target_alloc(size, omp_get_default_device());
  if (*devPtr == NULL) {
    return cudaErrorMemoryAllocation;
  }
  return cudaSuccess;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       cudaMemcpyKind kind) {
  // First , make sure we have at least one nonhost device
  int num_devices = omp_get_num_devices();

  if (count < 0 || num_devices < 1) {
    lastError = cudaErrorInvalidValue;
    return cudaErrorInvalidValue;
  }

  // get the host device number (which is the inital device)
  int host_device_num = omp_get_initial_device();

  // use the default device for gpu
  int gpu_device_num = omp_get_default_device();

  // default to copy from host to device
  int dst_device_num = gpu_device_num;
  int src_device_num = host_device_num;

  if (kind == cudaMemcpyDeviceToHost) {
    // copy from device to host
    dst_device_num = host_device_num;
    src_device_num = gpu_device_num;
  }

  // omp_target_memcpy returns 0 on success and non-zero on failure
  if (omp_target_memcpy(dst, src, count, 0, 0, dst_device_num,
                        src_device_num)) {
    lastError = cudaErrorMemoryAllocation;
    return cudaErrorMemoryAllocation;
  }

  lastError = cudaErrorMemoryAllocation;
  return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream = 0) {
  // First , make sure we have at least one nonhost device
  int num_devices = omp_get_num_devices();

  if (count < 0 || num_devices < 1) {
    lastError = cudaErrorInvalidValue;
    return cudaErrorInvalidValue;
  }

  // get the host device number (which is the inital device)
  int host_device_num = omp_get_initial_device();

  // use the default device for gpu
  int gpu_device_num = omp_get_default_device();

  // default to copy from host to device
  int dst_device_num = gpu_device_num;
  int src_device_num = host_device_num;

  if (kind == cudaMemcpyDeviceToHost) {
    // copy from device to host
    dst_device_num = host_device_num;
    src_device_num = gpu_device_num;
  }

  // omp_target_memcpy returns 0 on success and non-zero on failure
  if (omp_target_memcpy(dst, src, count, 0, 0, dst_device_num,
                        src_device_num)) {
    lastError = cudaErrorMemoryAllocation;
    return cudaErrorMemoryAllocation;
  }

  lastError = cudaSuccess;
  return cudaSuccess;
}

cudaError_t cudaThreadSynchronize() {

  unsigned long kernel_first = synced_kernels;
  unsigned long kernel_last = num_kernels;
  if (kernel_first < kernel_last) {
    for (unsigned long i = kernel_first; i < kernel_last; ++i) {
#pragma omp parallel
#pragma omp single
#pragma omp task depend(in : kernels[i])
      {}
    }
    synced_kernels.compare_exchange_strong(kernel_first, kernel_last);
  }

  lastError = cudaSuccess;
  return cudaSuccess;
}

cudaError_t cudaFree(void *devPtr) {
  if (omp_get_num_devices() < 1) {
    lastError = cudaErrorNoDevice;
    return cudaErrorNoDevice;
  }

  cudaDeviceSynchronize();

  omp_target_free(devPtr, omp_get_default_device());

  lastError = cudaSuccess;
  return cudaSuccess;
}

cudaError_t cudaGetDevice(int *device) {
  *device = omp_get_default_device();
  return cudaSuccess;
}

cudaError_t cudaGetDeviceCount(int *count) {

  *count = omp_get_num_devices();
  lastError = cudaSuccess;
  return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {

  omp_set_default_device(device);
  lastError = cudaSuccess;
  return cudaSuccess;
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
  unsigned char *ptr = (unsigned char*) devPtr;
#pragma omp target teams distribute parallel for is_device_ptr(ptr)
  for (int i = 0; i < count; ++i) {
    ptr[i] = value;
  }

  lastError = cudaSuccess;
  return cudaSuccess;
}
#endif
