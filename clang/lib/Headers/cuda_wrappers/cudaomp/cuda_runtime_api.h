#ifndef __CUDA_RUNTIME_API__
#define __CUDA_RUNTIME_API__

#include <atomic>
#include <stdint.h>
#include <string.h>
#include <omp.h>

#include "internal_types.h"

#if !defined(__default_val)
#if defined(__cplusplus)
#define __default_val(val) = val
#else
#define __default_val(val)
#endif
#endif

#if 1
typedef struct cudaDeviceProp {
  char name[256];
  cudaUUID_t uuid;
  size_t totalGlobalMem;
  size_t sharedMemPerBlock;
  int regsPerBlock;
  int warpSize;
  size_t memPitch;
  int maxThreadsPerBlock;
  int maxThreadsDim[3];
  int maxGridSize[3];
  int clockRate;
  size_t totalConstMem;
  int major;
  int minor;
  size_t textureAlignment;
  size_t texturePitchAlignment;
  int deviceOverlap;
  int multiProcessorCount;
  int kernelExecTimeoutEnabled;
  int integrated;
  int canMapHostMemory;
  int computeMode;
  int maxTexture1D;
  int maxTexture1DMipmap;
  int maxTexture1DLinear;
  int maxTexture2D[2];
  int maxTexture2DMipmap[2];
  int maxTexture2DLinear[3];
  int maxTexture2DGather[2];
  int maxTexture3D[3];
  int maxTexture3DAlt[3];
  int maxTextureCubemap;
  int maxTexture1DLayered[2];
  int maxTexture2DLayered[3];
  int maxTextureCubemapLayered[2];
  int maxSurface1D;
  int maxSurface2D[2];
  int maxSurface3D[3];
  int maxSurface1DLayered[2];
  int maxSurface2DLayered[3];
  int maxSurfaceCubemap;
  int maxSurfaceCubemapLayered[2];
  size_t surfaceAlignment;
  int concurrentKernels;
  int ECCEnabled;
  int pciBusID;
  int pciDeviceID;
  int pciDomainID;
  int tccDriver;
  int asyncEngineCount;
  int unifiedAddressing;
  int memoryClockRate;
  int memoryBusWidth;
  int l2CacheSize;
  int persistingL2CacheMaxSize;
  int maxThreadsPerMultiProcessor;
  int streamPrioritiesSupported;
  int globalL1CacheSupported;
  int localL1CacheSupported;
  size_t sharedMemPerMultiprocessor;
  int regsPerMultiprocessor;
  int managedMemory;
  int isMultiGpuBoard;
  int multiGpuBoardGroupID;
  int singleToDoublePrecisionPerfRatio;
  int pageableMemoryAccess;
  int concurrentManagedAccess;
  int computePreemptionSupported;
  int canUseHostPointerForRegisteredMem;
  int cooperativeLaunch;
  int cooperativeMultiDeviceLaunch;
  int pageableMemoryAccessUsesHostPageTables;
  int directManagedMemAccessFromHost;
  int accessPolicyMaxWindowSize;
} cudaDeviceProp;
#endif

//Declarations of global variables
static cudaError_t __cudaomp_last_error = cudaSuccess;
static int *kernels = nullptr;
static std::atomic<unsigned long> num_kernels = {0};
static std::atomic<unsigned long> synced_kernels = {0};

#if 0
#pragma omp begin declare variant match(implementation={compiler(llvm)})
#endif

inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {

  if (device >= omp_get_num_devices()) {

    __cudaomp_last_error = cudaErrorInvalidDevice;
    return cudaErrorInvalidDevice;
  }
  if (device != omp_get_default_device()) {
    omp_set_default_device(device);
  }

  // set some of the properties
  strncpy(prop->name, "OpenMP device", 256);

  __cudaomp_last_error = cudaSuccess;
  return cudaSuccess;
}

// Returns the last error that has been produced and resets it to cudaSuccess
inline cudaError_t cudaGetLastError() {
  cudaError_t tempError = __cudaomp_last_error;
  __cudaomp_last_error = cudaSuccess;
  return tempError;
}

inline cudaError_t cudaPeekAtLastError() { return __cudaomp_last_error; }

inline const char *cudaGetErrorString(cudaError_t error) {
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
  case cudaErrorOTHER:
    return "NOT IMPLEMENTED";
  default:
    return "Unrecognized Error Code!";
  }
  return nullptr;
}
inline const char *cudaGetErrorName(cudaError_t error) {
  return cudaGetErrorString(error);
}

// TODO, fix tgt_kernel_synchronize
inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  //DEBUGP("===> TODO FIX cudaStreamSynchronize\n");
  if (__tgt_kernel_synchronize(omp_get_default_device(), stream))
    return __cudaomp_last_error = cudaErrorOTHER;
  return __cudaomp_last_error = cudaSuccess;
}

inline cudaError_t cudaStreamDestroy (cudaStream_t stream) {
  if (__tgt_destroy_stream(omp_get_default_device(), (void *) stream))
    return __cudaomp_last_error = cudaErrorOTHER;
  return __cudaomp_last_error = cudaSuccess;
}


// TODO
inline cudaError_t cudaDeviceSynchronize() {
  //return cudaStreamSynchronize(nullptr);
  if (__tgt_device_synchronize(omp_get_default_device()))
    return __cudaomp_last_error = cudaErrorOTHER;
  return __cudaomp_last_error = cudaSuccess;
}

inline cudaError_t __cudaMemset(void *devPtr, int value, size_t count) {
  unsigned char *ptr = (unsigned char*) devPtr;
#pragma omp target teams distribute parallel for is_device_ptr(ptr)
  for (int i = 0; i < count; ++i) {
    ptr[i] = value;
  }

  return __cudaomp_last_error = cudaSuccess;
}

template<class T>
inline cudaError_t cudaMemset(T *devPtr, int value, size_t count){
  return __cudaMemset((void *) devPtr, value, count);
}

inline cudaError_t __cudaFree(void *devPtr) {
  if (omp_get_num_devices() < 1) {
    return __cudaomp_last_error = cudaErrorNoDevice;
  }

  cudaDeviceSynchronize();

  omp_target_free(devPtr, omp_get_default_device());

  return __cudaomp_last_error = cudaSuccess;
}

template<class T>
inline cudaError_t cudaFree(T* ptr){
  return __cudaFree((void*) ptr);
}

inline cudaError_t __cudaMalloc(void **devPtr, size_t size) {
  int num_devices = omp_get_num_devices();
  if (size < 0 || num_devices < 1) {
    return __cudaomp_last_error = cudaErrorInvalidValue;
  }
  *devPtr = omp_target_alloc(size, omp_get_default_device());
  if (*devPtr == NULL) {
    return __cudaomp_last_error = cudaErrorMemoryAllocation;
  }
  return __cudaomp_last_error = cudaSuccess;
}

template<class T>
cudaError_t cudaMalloc(T **devPtr, size_t size) {
  return __cudaMalloc((void **)devPtr, size);
}

inline cudaError_t __cudaMemcpy(void *dst, const void *src, size_t count,
                       cudaMemcpyKind kind) {
  // First , make sure we have at least one nonhost device
  int num_devices = omp_get_num_devices();

  if (count < 0 || num_devices < 1) {
    return __cudaomp_last_error = cudaErrorInvalidValue;
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
  if (omp_target_memcpy_stream(dst, src, count, 0, 0, dst_device_num,
                        src_device_num, nullptr)) {
    return __cudaomp_last_error = cudaErrorMemoryAllocation;
  }

  return __cudaomp_last_error = cudaSuccess;
}

template<class T>
inline cudaError_t cudaMemcpy(T *dst, const T *src, size_t count,
                       cudaMemcpyKind kind){
  return __cudaMemcpy((void*) dst, (void *) src, count, kind);
}

inline cudaError_t __cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream = 0) {
  // First , make sure we have at least one nonhost device
  int num_devices = omp_get_num_devices();

  if (count < 0 || num_devices < 1) {
    return __cudaomp_last_error = cudaErrorInvalidValue;
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
  if (omp_target_memcpy_stream(dst, src, count, 0, 0, dst_device_num,
                        src_device_num, stream)) {
    return __cudaomp_last_error = cudaErrorMemoryAllocation;
  }

  return __cudaomp_last_error = cudaSuccess;
}

template<class T>
inline cudaError_t cudaMemcpyAsync(T *dst, const T *src, size_t count,
  cudaMemcpyKind kind, cudaStream_t stream = 0){
    return __cudaMemcpyAsync((void*) dst, (void *) src, count, kind, stream);
}

inline cudaError_t cudaThreadSynchronize() {
  return cudaDeviceSynchronize();
}

inline cudaError_t cudaGetDevice(int *device) {

  *device = omp_get_default_device();
  return __cudaomp_last_error = cudaSuccess;
}

inline cudaError_t cudaGetDeviceCount(int *count) {

  *count = omp_get_num_devices();
  return __cudaomp_last_error = cudaSuccess;
}

inline cudaError_t cudaSetDevice(int device) {

  omp_set_default_device(device);
  return __cudaomp_last_error = cudaSuccess;
}

inline cudaError_t __cudaMallocHost(void **ptr, size_t size, unsigned int flags){
  *ptr = llvm_omp_target_alloc_host(size, omp_get_default_device());
  return __cudaomp_last_error = cudaSuccess;
}

template<class T>
inline cudaError_t cudaMallocHost(T **ptr, size_t size, unsigned int flags = 0){
  return __cudaMallocHost((void **)ptr, size, flags);
}

inline cudaError_t __cudaFreeHost(void *ptr){
  omp_target_free(ptr, omp_get_default_device());
  return __cudaomp_last_error = cudaSuccess;
}

template<class T>
inline cudaError_t cudaFreeHost(T* ptr){
  return __cudaFreeHost((void *) ptr);
}

inline cudaError_t cudaStreamCreate (cudaStream_t *pStream){
  int num_devices = omp_get_num_devices();
  int64_t gpu_device_num = omp_get_default_device();

  if (num_devices < 1) {
    return __cudaomp_last_error = cudaErrorInvalidValue;
  }

  __tgt_create_stream(gpu_device_num, (void **) pStream);
  return __cudaomp_last_error = cudaSuccess;
}

inline cudaError_t cudaDeviceReset(void){
  DEBUGP("===> TODO FIX cudaDeviceReset\n");
  return __cudaomp_last_error = cudaSuccess;
}

inline cudaError_t __cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream=0){
  return __cudaomp_last_error = cudaSuccess;
}

template<class T>
inline cudaError_t cudaMemsetAsync(T *devPtr, int value, size_t count, cudaStream_t stream=0){
  return __cudaMemsetAsync((void *) devPtr, value, count, stream);
}

void *llvm_omp_target_alloc_shared(size_t size, int device_num);
inline cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags __default_val(0)){
  *devPtr = llvm_omp_target_alloc_shared(size, omp_get_default_device());
  return __cudaomp_last_error = cudaSuccess;
}

#if 0
#pragma omp end declare variant
#endif

#endif
