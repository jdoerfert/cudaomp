#ifndef __CUDA_TYPES__
#define __CUDA_TYPES__

#include <stddef.h>

#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __managed__ __attribute__((managed))

#define __forceinline__ __inline__ __attribute__((always_inline))

extern "C" {
__device__ int printf(const char *, ...);
}

#define DEBUGP(...) printf(__VA_ARGS__)

#include "vector_types.h"
#include "vector_functions.h"

#if 0
#include "driver_types.h"
#else

struct CUuuid_st {
  char bytes[16];
};

typedef struct CUuuid_st cudaUUID_t;

typedef struct CUstream_st *cudaStream_t;

// TODO: There are many fields missing in this enumeration.
typedef enum cudaError {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorNoDevice = 100,
  cudaErrorInvalidDevice = 101
} cudaError_t;

/// used in cudaMemcpy to specify the copy direction
enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4
};
#endif

#endif
