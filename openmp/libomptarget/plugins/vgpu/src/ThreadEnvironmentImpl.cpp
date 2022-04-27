//===---- ThreadEnvironmentImpl.h - Virtual GPU thread environment - C++ --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdint>

#include "ThreadEnvironmentImpl.h"
#include <barrier>
#include <cstdio>
#include <functional>
#include <map>
#include <thread>
#include <vector>

void WarpEnvironmentTy::configure(unsigned NumThreads) {
  ThreadsPerWarp = NumThreads;
}

WarpEnvironmentTy::WarpEnvironmentTy()
    : ID(Idx++), ShuffleBuffer(ThreadsPerWarp),
      Barrier(ThreadsPerWarp, []() {}), ShuffleBarrier(ThreadsPerWarp, []() {}),
      ShuffleDownBarrier(ThreadsPerWarp, []() {}) {}

unsigned WarpEnvironmentTy::getWarpId() const { return ID; }

int WarpEnvironmentTy::getNumThreads() const { return ThreadsPerWarp; }

void WarpEnvironmentTy::sync(int Ordering) { Barrier.arrive_and_wait(); }

void WarpEnvironmentTy::writeShuffleBuffer(int32_t Var, unsigned LaneId) {
  ShuffleBuffer[LaneId] = Var;
}

int32_t WarpEnvironmentTy::getShuffleBuffer(unsigned LaneId) {
  return ShuffleBuffer[LaneId];
}

void WarpEnvironmentTy::waitShuffleBarrier() {
  ShuffleBarrier.arrive_and_wait();
}

void WarpEnvironmentTy::waitShuffleDownBarrier() {
  ShuffleBarrier.arrive_and_wait();
}

unsigned WarpEnvironmentTy::Idx = 0;
unsigned WarpEnvironmentTy::ThreadsPerWarp = 0;

void CTAEnvironmentTy::configure(unsigned TotalNumThreads, unsigned NumBlocks) {
  NumThreads = TotalNumThreads / NumBlocks;
  NumCTAs = NumBlocks;
}

CTAEnvironmentTy::CTAEnvironmentTy()
    : ID(Idx++), Barrier(NumThreads, []() {}), SyncThreads(NumThreads, []() {}),
      NamedBarrier(NumThreads, []() {}) {}

unsigned CTAEnvironmentTy::getId() const { return ID; }
unsigned CTAEnvironmentTy::getNumThreads() const { return NumThreads; }

unsigned CTAEnvironmentTy::getNumBlocks() const { return NumCTAs; }

void CTAEnvironmentTy::fence(int Ordering) { Barrier.arrive_and_wait(); }
void CTAEnvironmentTy::syncThreads() { SyncThreads.arrive_and_wait(); }
void CTAEnvironmentTy::namedBarrier() { NamedBarrier.arrive_and_wait(); }

unsigned CTAEnvironmentTy::Idx = 0;
unsigned CTAEnvironmentTy::NumThreads = 0;
unsigned CTAEnvironmentTy::NumCTAs = 0;

ThreadBlockEnvironmentTy::ThreadBlockEnvironmentTy(unsigned ID,
                                                   unsigned NumBlocks)
    : ID(ID), NumBlocks(NumBlocks) {}

unsigned ThreadBlockEnvironmentTy::getId() const { return ID; }
unsigned ThreadBlockEnvironmentTy::getNumBlocks() const { return NumBlocks; }

namespace VGPUImpl {
ThreadEnvironmentTy::ThreadEnvironmentTy(WarpEnvironmentTy *WE,
                                         CTAEnvironmentTy *CTAE)
    : ThreadIdInWarp(Idx++ % WE->getNumThreads()),
      ThreadIdInBlock(WE->getWarpId() * WE->getNumThreads() + ThreadIdInWarp),
      GlobalThreadIdx(CTAE->getId() * CTAE->getNumThreads() + ThreadIdInBlock),
      WarpEnvironment(WE), CTAEnvironment(CTAE) {}

void ThreadEnvironmentTy::setBlockEnv(ThreadBlockEnvironmentTy *TBE) {
  ThreadBlockEnvironment = TBE;
}

void ThreadEnvironmentTy::resetBlockEnv() {
  delete ThreadBlockEnvironment;
  ThreadBlockEnvironment = nullptr;
}

unsigned ThreadEnvironmentTy::getThreadIdInWarp() const {
  return ThreadIdInWarp;
}
unsigned ThreadEnvironmentTy::getThreadIdInBlock() const {
  return ThreadIdInBlock;
}
unsigned ThreadEnvironmentTy::getGlobalThreadId() const {
  return GlobalThreadIdx;
}

unsigned ThreadEnvironmentTy::getBlockSize() const {
  return CTAEnvironment->getNumThreads();
}

unsigned ThreadEnvironmentTy::getBlockId() const {
  return ThreadBlockEnvironment->getId();
}

unsigned ThreadEnvironmentTy::getNumberOfBlocks() const {
  return ThreadBlockEnvironment->getNumBlocks();
}
unsigned ThreadEnvironmentTy::getKernelSize() const {
  return getBlockSize() * getNumberOfBlocks();
}

// FIXME: This is wrong
LaneMaskTy ThreadEnvironmentTy::getActiveMask() const { return ~0U; }

void ThreadEnvironmentTy::fenceTeam(int Ordering) {
  CTAEnvironment->fence(Ordering);
}
void ThreadEnvironmentTy::syncWarp(int Ordering) {
  WarpEnvironment->sync(Ordering);
}

int32_t ThreadEnvironmentTy::shuffle(uint64_t Mask, int32_t Var,
                                     uint64_t SrcLane) {
  WarpEnvironment->waitShuffleBarrier();
  WarpEnvironment->writeShuffleBuffer(Var, ThreadIdInWarp);
  WarpEnvironment->waitShuffleBarrier();
  Var = WarpEnvironment->getShuffleBuffer(ThreadIdInWarp);
  return Var;
}

int32_t ThreadEnvironmentTy::shuffleDown(uint64_t Mask, int32_t Var,
                                         uint32_t Delta) {
  WarpEnvironment->waitShuffleDownBarrier();
  WarpEnvironment->writeShuffleBuffer(Var, ThreadIdInWarp);
  WarpEnvironment->waitShuffleDownBarrier();
  Var = WarpEnvironment->getShuffleBuffer((ThreadIdInWarp + Delta) %
                                          getWarpSize());
  return Var;
}

void ThreadEnvironmentTy::namedBarrier(bool Generic) {
  if (Generic) {
    CTAEnvironment->namedBarrier();
  } else {
    CTAEnvironment->syncThreads();
  }
}

void ThreadEnvironmentTy::fenceKernel(int32_t MemoryOrder) {
  std::atomic_thread_fence(static_cast<std::memory_order>(MemoryOrder));
}

unsigned ThreadEnvironmentTy::getWarpSize() const {
  return WarpEnvironment->getNumThreads();
}

unsigned ThreadEnvironmentTy::Idx = 0;

} // namespace VGPUImpl
