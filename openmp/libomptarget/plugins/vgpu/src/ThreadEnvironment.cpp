//===---- DeviceEnvironment.cpp - Virtual GPU Device Environment -- C++ ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of VGPU environment classes.
//
//===----------------------------------------------------------------------===//
//
#include <cstdint>

#include "ThreadEnvironment.h"
#include "ThreadEnvironmentImpl.h"
#include <barrier>
#include <mutex>

std::mutex AtomicIncLock;

uint32_t VGPUImpl::atomicInc(uint32_t *Address, uint32_t Val, int Ordering) {
  std::lock_guard G(AtomicIncLock);
  uint32_t V = *Address;
  if (V >= Val)
    *Address = 0;
  else
    *Address += 1;
  return V;
}

void VGPUImpl::initLock(uint32_t *Lock) { Lock = (uint32_t *)new std::mutex; }

void VGPUImpl::destroyLock(uint32_t *Lock) {
  std::mutex *Mtx = (std::mutex *)Lock;
  delete Mtx;
}

void VGPUImpl::setLock(uint32_t *Lock) { ((std::mutex *)Lock)->lock(); }

void VGPUImpl::unsetLock(uint32_t *Lock) { ((std::mutex *)Lock)->unlock(); }

bool VGPUImpl::testLock(uint32_t *Lock) {
  return ((std::mutex *)Lock)->try_lock();
}

extern thread_local ThreadEnvironmentTy *ThreadEnvironment;

ThreadEnvironmentTy *getThreadEnvironment() { return ThreadEnvironment; }

ThreadEnvironmentTy::ThreadEnvironmentTy(WarpEnvironmentTy *WE,
                                         CTAEnvironmentTy *CTAE)
    : Impl(new VGPUImpl::ThreadEnvironmentTy(WE, CTAE)) {}

ThreadEnvironmentTy::~ThreadEnvironmentTy() { delete Impl; }

void ThreadEnvironmentTy::fenceTeam(int Ordering) { Impl->fenceTeam(Ordering); }

void ThreadEnvironmentTy::syncWarp(int Ordering) { Impl->syncWarp(Ordering); }

unsigned ThreadEnvironmentTy::getThreadIdInWarp() const {
  return Impl->getThreadIdInWarp();
}

unsigned ThreadEnvironmentTy::getThreadIdInBlock() const {
  return Impl->getThreadIdInBlock();
}

unsigned ThreadEnvironmentTy::getGlobalThreadId() const {
  return Impl->getGlobalThreadId();
}

unsigned ThreadEnvironmentTy::getBlockSize() const {
  return Impl->getBlockSize();
}

unsigned ThreadEnvironmentTy::getKernelSize() const {
  return Impl->getKernelSize();
}

unsigned ThreadEnvironmentTy::getBlockId() const { return Impl->getBlockId(); }

unsigned ThreadEnvironmentTy::getNumberOfBlocks() const {
  return Impl->getNumberOfBlocks();
}

LaneMaskTy ThreadEnvironmentTy::getActiveMask() const {
  return Impl->getActiveMask();
}

int32_t ThreadEnvironmentTy::shuffle(uint64_t Mask, int32_t Var,
                                     uint64_t SrcLane) {
  return Impl->shuffle(Mask, Var, SrcLane);
}

int32_t ThreadEnvironmentTy::shuffleDown(uint64_t Mask, int32_t Var,
                                         uint32_t Delta) {
  return Impl->shuffleDown(Mask, Var, Delta);
}

void ThreadEnvironmentTy::fenceKernel(int32_t MemoryOrder) {
  return Impl->fenceKernel(MemoryOrder);
}

void ThreadEnvironmentTy::namedBarrier(bool Generic) {
  Impl->namedBarrier(Generic);
}

void ThreadEnvironmentTy::setBlockEnv(ThreadBlockEnvironmentTy *TBE) {
  Impl->setBlockEnv(TBE);
}

void ThreadEnvironmentTy::resetBlockEnv() { Impl->resetBlockEnv(); }

unsigned ThreadEnvironmentTy::getWarpSize() const {
  return Impl->getWarpSize();
}
