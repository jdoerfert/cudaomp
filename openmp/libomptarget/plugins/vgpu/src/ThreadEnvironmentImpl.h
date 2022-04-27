//===---- ThreadEnvironmentImpl.h - Virtual GPU thread environment - C++ --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_VGPU_SRC_THREADENVIRONMENTIMPL_H
#define OPENMP_LIBOMPTARGET_PLUGINS_VGPU_SRC_THREADENVIRONMENTIMPL_H

#include "ThreadEnvironment.h"
#include <barrier>
#include <cstdio>
#include <functional>
#include <map>
#include <thread>
#include <vector>

using BarrierTy = std::barrier<std::function<void(void)>>;

class WarpEnvironmentTy {
  static unsigned Idx;

  const unsigned ID;

  std::vector<int32_t> ShuffleBuffer;

  BarrierTy Barrier;
  BarrierTy ShuffleBarrier;
  BarrierTy ShuffleDownBarrier;

public:
  static void configure(unsigned NumThreadsInWarp);

  static unsigned ThreadsPerWarp;

  WarpEnvironmentTy();

  unsigned getWarpId() const;
  int getNumThreads() const;

  void sync(int Ordering);
  void writeShuffleBuffer(int32_t Var, unsigned LaneId);

  int32_t getShuffleBuffer(unsigned LaneId);

  void waitShuffleBarrier();
  void waitShuffleDownBarrier();
};

class CTAEnvironmentTy {
  static unsigned Idx;

public:
  unsigned ID;
  static unsigned NumThreads;
  static unsigned NumCTAs;

  BarrierTy Barrier;
  BarrierTy SyncThreads;
  BarrierTy NamedBarrier;

  static void configure(unsigned TotalNumThreads, unsigned NumBlocksInCTA);

  CTAEnvironmentTy();

  unsigned getId() const;
  unsigned getNumThreads() const;

  unsigned getNumBlocks() const;

  void fence(int Ordering);
  void syncThreads();
  void namedBarrier();
};

class ThreadBlockEnvironmentTy {
  unsigned ID;
  unsigned NumBlocks;

public:
  ThreadBlockEnvironmentTy(unsigned ID, unsigned NumBlocks);

  unsigned getId() const;
  unsigned getNumBlocks() const;
};

namespace VGPUImpl {
class ThreadEnvironmentTy {
  static unsigned Idx;
  unsigned ThreadIdInWarp;
  unsigned ThreadIdInBlock;
  unsigned GlobalThreadIdx;

  WarpEnvironmentTy *WarpEnvironment;
  ThreadBlockEnvironmentTy *ThreadBlockEnvironment;
  CTAEnvironmentTy *CTAEnvironment;

public:
  ThreadEnvironmentTy(WarpEnvironmentTy *WE, CTAEnvironmentTy *CTAE);

  void setBlockEnv(ThreadBlockEnvironmentTy *TBE);

  void resetBlockEnv();

  unsigned getThreadIdInWarp() const;
  unsigned getThreadIdInBlock() const;
  unsigned getGlobalThreadId() const;

  unsigned getBlockSize() const;

  unsigned getBlockId() const;

  unsigned getNumberOfBlocks() const;
  unsigned getKernelSize() const;

  // FIXME: This is wrong
  LaneMaskTy getActiveMask() const;

  void fenceTeam(int Ordering);
  void syncWarp(int Ordering);

  int32_t shuffle(uint64_t Mask, int32_t Var, uint64_t SrcLane);

  int32_t shuffleDown(uint64_t Mask, int32_t Var, uint32_t Delta);

  void namedBarrier(bool Generic);

  void fenceKernel(int32_t MemoryOrder);

  unsigned getWarpSize() const;
};

} // namespace VGPUImpl

#endif // OPENMP_LIBOMPTARGET_PLUGINS_VGPU_SRC_THREADENVIRONMENTIMPL_H
