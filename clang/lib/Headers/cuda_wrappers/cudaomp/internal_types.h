#ifndef __CUDA_TYPES__
#define __CUDA_TYPES__

#include <stddef.h>

#define DEBUGP(...) printf(__VA_ARGS__)

typedef struct uint3 {
  unsigned int x;
  unsigned int y;
  unsigned int z;
} uint3;

typedef struct dim3 {
  unsigned int x;
  unsigned int y;
  unsigned int z;
  dim3(unsigned int _x = 1, unsigned int _y = 1, unsigned int _z = 1)
      : x(_x), y(_y), z(_z) {}
} dim3;

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