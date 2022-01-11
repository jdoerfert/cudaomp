#ifndef __CUDA_H__
#define __CUDA_H__

#include <assert.h>
#include <atomic>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "internal_types.h"

#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __managed__ __attribute__((managed))

//#pragma omp begin declare target
//__constant__ dim3 grid;
//__constant__ dim3 block;
//#pragma omp end declare target

#endif
