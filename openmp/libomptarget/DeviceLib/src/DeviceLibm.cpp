//===-------- OpenMPMath.cpp - Implementation of OpenMP math fns -----------===
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===
#if defined(__CUDA__)
#define __OPENMP_NVPTX__

// Include declarations for libdevice functions.
#include <Headers/__clang_cuda_libdevice_declares.h>
// Include the device functions.
#include <Headers/__clang_cuda_device_functions.h>

extern "C" {
// Call libdevice functions from the standard math names.
#include <Headers/__clang_cuda_math.h>
}

#undef __OPENMP_NVPTX__
#elif defined(__AMDGPU__)
#define __OPENMP_AMDGCN__

// Include declarations for libdevice functions.
#include <Headers/__clang_hip_libdevice_declares.h>

// Call libdevice functions from the standard math names.
#include <Headers/__clang_hip_math.h>

#undef __OPENMP_AMDGCN__
#endif
