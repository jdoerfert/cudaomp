//===---- ThreadEnvironment.h - Virtual GPU thread environment ----- C++ --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_VGPU_SRC_THREADENVIRONMENT_H
#define OPENMP_LIBOMPTARGET_PLUGINS_VGPU_SRC_THREADENVIRONMENT_H

using LaneMaskTy = uint64_t;

// Forward declaration
class WarpEnvironmentTy;
class ThreadBlockEnvironmentTy;
class CTAEnvironmentTy;
namespace VGPUImpl {
class ThreadEnvironmentTy;
void initLock(uint32_t *Lock);
void destroyLock(uint32_t *Lock);
void setLock(uint32_t *Lock);
void unsetLock(uint32_t *Lock);
bool testLock(uint32_t *Lock);
uint32_t atomicInc(uint32_t *Address, uint32_t Val, int Ordering);
} // namespace VGPUImpl

class ThreadEnvironmentTy {
  VGPUImpl::ThreadEnvironmentTy *Impl;

public:
  ThreadEnvironmentTy(WarpEnvironmentTy *WE, CTAEnvironmentTy *CTAE);

  ~ThreadEnvironmentTy();

  unsigned getThreadIdInWarp() const;

  unsigned getThreadIdInBlock() const;

  unsigned getGlobalThreadId() const;

  unsigned getBlockSize() const;

  unsigned getKernelSize() const;

  unsigned getBlockId() const;

  unsigned getNumberOfBlocks() const;

  LaneMaskTy getActiveMask() const;

  unsigned getWarpSize() const;

  int32_t shuffle(uint64_t Mask, int32_t Var, uint64_t SrcLane);

  int32_t shuffleDown(uint64_t Mask, int32_t Var, uint32_t Delta);

  void fenceKernel(int32_t MemoryOrder);

  void fenceTeam(int MemoryOrder);

  void syncWarp(int Mask);

  void namedBarrier(bool Generic);

  void setBlockEnv(ThreadBlockEnvironmentTy *TBE);

  void resetBlockEnv();
};

ThreadEnvironmentTy *getThreadEnvironment(void);

#endif // OPENMP_LIBOMPTARGET_PLUGINS_VGPU_SRC_THREADENVIRONMENT_H
