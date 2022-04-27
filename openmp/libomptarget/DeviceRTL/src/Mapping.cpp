//===------- Mapping.cpp - OpenMP device runtime mapping helpers -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "Mapping.h"
#include "Interface.h"
#include "State.h"
#include "Types.h"
#include "Utils.h"

#pragma omp declare target

#include "ThreadEnvironment.h"
#include "llvm/Frontend/OpenMP/OMPGridValues.h"

using namespace _OMP;

/// Virtual GPU Implementation
///
///{
#pragma omp begin declare variant match(device = {kind(cpu)})

namespace _OMP {
namespace impl {

constexpr const llvm::omp::GV &getGridValue() {
  return llvm::omp::VirtualGpuGridValues;
}

LaneMaskTy activemask() {
  uint64_t B = 0;
  uint32_t N = mapping::getWarpSize();
  while (N)
    B |= (1 << (--N));
  return B;
}

LaneMaskTy lanemaskLT() {
  const uint32_t Lane = mapping::getThreadIdInWarp();
  LaneMaskTy Ballot = mapping::activemask();
  LaneMaskTy Mask = ((LaneMaskTy)1 << Lane) - (LaneMaskTy)1;
  return Mask & Ballot;
}

LaneMaskTy lanemaskGT() {
  const uint32_t Lane = mapping::getThreadIdInWarp();
  if (Lane == (mapping::getWarpSize() - 1))
    return 0;
  LaneMaskTy Ballot = mapping::activemask();
  LaneMaskTy Mask = (~((LaneMaskTy)0)) << (Lane + 1);
  return Mask & Ballot;
}

uint32_t getThreadIdInWarp() {
  return mapping::getThreadIdInBlock() & (mapping::getWarpSize() - 1);
}

uint32_t getThreadIdInBlock(int Dim = 0) {
  return getThreadEnvironment()->getThreadIdInBlock();
}

uint32_t getNumHardwareThreadsInBlock(int Dim = 0) {
  return getThreadEnvironment()->getBlockSize();
}

uint32_t getKernelSize() { return getThreadEnvironment()->getKernelSize(); }

uint32_t getBlockId(int Dim = 0) { return getThreadEnvironment()->getBlockId(); }

uint32_t getNumberOfBlocks(int Dim = 0) {
  return getThreadEnvironment()->getNumberOfBlocks();
}

uint32_t getNumberOfProcessorElements() { return mapping::getBlockSize(); }

uint32_t getWarpId() {
  return mapping::getThreadIdInBlock() / mapping::getWarpSize();
}

uint32_t getWarpSize() { return getThreadEnvironment()->getWarpSize(); }

uint32_t getNumberOfWarpsInBlock() {
  return (mapping::getBlockSize() + mapping::getWarpSize() - 1) /
         mapping::getWarpSize();
}

} // namespace impl
} // namespace _OMP

#pragma omp end declare variant

namespace _OMP {
namespace impl {

/// AMDGCN Implementation
///
///{
#pragma omp begin declare variant match(device = {arch(amdgcn)})

static const llvm::omp::GV &getGridValue() {
  return llvm::omp::getAMDGPUGridValues<__AMDGCN_WAVEFRONT_SIZE>();
}

uint32_t getGridDim(uint32_t n, uint16_t d) {
  uint32_t q = n / d;
  return q + (n > q * d);
}

uint32_t getWorkgroupDim(uint32_t group_id, uint32_t grid_size,
                         uint16_t group_size) {
  uint32_t r = grid_size - group_id * group_size;
  return (r < group_size) ? r : group_size;
}

uint32_t getThreadIdInBlock(int Dim = 0) {
  switch(Dim) {
    case 0:
  return __builtin_amdgcn_workitem_id_x();
    case 1:
  return __builtin_amdgcn_workitem_id_y();
    case 2:
  return __builtin_amdgcn_workitem_id_z();
    default:break;
  }
  __builtin_unreachable();
}

uint32_t getKernelSize(int Dim = 0) {
  switch(Dim) {
    case 0:
  return __builtin_amdgcn_grid_size_x();
    case 1:
  return __builtin_amdgcn_grid_size_y();
    case 2:
  return __builtin_amdgcn_grid_size_z();
    default:break;
  }
  __builtin_unreachable();
}

uint32_t getBlockId(int Dim = 0) {
  switch(Dim) {
    case 0:
  return __builtin_amdgcn_workgroup_id_x();
    case 1:
  return __builtin_amdgcn_workgroup_id_y();
    case 2:
  return __builtin_amdgcn_workgroup_id_z();
    default:break;
  }
  __builtin_unreachable();
}

uint32_t getBlockSize(int Dim = 0) {
  switch(Dim) {
    case 0:
  return __builtin_amdgcn_workgroup_size_x();
    case 1:
  return __builtin_amdgcn_workgroup_size_y();
    case 2:
  return __builtin_amdgcn_workgroup_size_z();
    default:break;
  }
  __builtin_unreachable();
}

uint32_t getNumHardwareThreadsInBlock(int Dim = 0) {
  return getWorkgroupDim(getBlockId(Dim),
                         getKernelSize(Dim),
                         getBlockSize(Dim));
}

LaneMaskTy activemask() { return __builtin_amdgcn_read_exec(); }

LaneMaskTy lanemaskLT() {
  uint32_t Lane = mapping::getThreadIdInWarp();
  int64_t Ballot = mapping::activemask();
  uint64_t Mask = ((uint64_t)1 << Lane) - (uint64_t)1;
  return Mask & Ballot;
}

LaneMaskTy lanemaskGT() {
  uint32_t Lane = mapping::getThreadIdInWarp();
  if (Lane == (mapping::getWarpSize() - 1))
    return 0;
  int64_t Ballot = mapping::activemask();
  uint64_t Mask = (~((uint64_t)0)) << (Lane + 1);
  return Mask & Ballot;
}

uint32_t getThreadIdInWarp() {
  return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
}


uint32_t getNumberOfBlocks(int Dim = 0) {
  return getGridDim(getKernelSize(Dim),
                    getBlockSize(Dim));
}

uint32_t getWarpId() {
  return impl::getThreadIdInBlock() / mapping::getWarpSize();
}

uint32_t getNumberOfWarpsInBlock() {
  return mapping::getBlockSize() / mapping::getWarpSize();
}

#pragma omp end declare variant
///}

/// NVPTX Implementation
///
///{
#pragma omp begin declare variant match(                                       \
    device = {arch(nvptx, nvptx64)}, implementation = {extension(match_any)})

uint32_t getNumHardwareThreadsInBlock(int Dim = 0) {
  switch(Dim) {
    case 0:
  return __nvvm_read_ptx_sreg_ntid_x();
    case 1:
  return __nvvm_read_ptx_sreg_ntid_y();
    case 2:
  return __nvvm_read_ptx_sreg_ntid_z();
    default:break;
  }
  __builtin_unreachable();
}

static const llvm::omp::GV &getGridValue() {
  return llvm::omp::NVPTXGridValues;
}

LaneMaskTy activemask() {
  unsigned int Mask;
  asm("activemask.b32 %0;" : "=r"(Mask));
  return Mask;
}

LaneMaskTy lanemaskLT() {
  __kmpc_impl_lanemask_t Res;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(Res));
  return Res;
}

LaneMaskTy lanemaskGT() {
  __kmpc_impl_lanemask_t Res;
  asm("mov.u32 %0, %%lanemask_gt;" : "=r"(Res));
  return Res;
}

uint32_t getThreadIdInBlock(int Dim = 0) {
  switch(Dim) {
    case 0:
  return __nvvm_read_ptx_sreg_tid_x();
    case 1:
  return __nvvm_read_ptx_sreg_tid_y();
    case 2:
  return __nvvm_read_ptx_sreg_tid_z();
    default:break;
  }
  __builtin_unreachable();
}

uint32_t getThreadIdInWarp() {
  return impl::getThreadIdInBlock() & (mapping::getWarpSize() - 1);
}

uint32_t getKernelSize(int Dim = 0) {
  return __nvvm_read_ptx_sreg_nctaid_x() *
         mapping::getNumberOfProcessorElements();
}

uint32_t getBlockId(int Dim = 0) {
  switch(Dim) {
    case 0:
  return __nvvm_read_ptx_sreg_ctaid_x();
    case 1:
  return __nvvm_read_ptx_sreg_ctaid_y();
    case 2:
  return __nvvm_read_ptx_sreg_ctaid_z();
    default:break;
  }
  __builtin_unreachable();
}

uint32_t getNumberOfBlocks(int Dim = 0) {
  switch(Dim) {
    case 0:
  return __nvvm_read_ptx_sreg_nctaid_x();
    case 1:
  return __nvvm_read_ptx_sreg_nctaid_y();
    case 2:
  return __nvvm_read_ptx_sreg_nctaid_z();
    default:break;
  }
  __builtin_unreachable();
}

uint32_t getWarpId(int Dim = 0) {
  return impl::getThreadIdInBlock(Dim) / mapping::getWarpSize();
}

uint32_t getNumberOfWarpsInBlock() {
  return (mapping::getBlockSize() + mapping::getWarpSize() - 1) /
         mapping::getWarpSize();
}

#pragma omp end declare variant
///}

uint32_t getWarpSize() { return getGridValue().GV_Warp_Size; }

} // namespace impl
} // namespace _OMP

/// We have to be deliberate about the distinction of `mapping::` and `impl::`
/// below to avoid repeating assumptions or including irrelevant ones.
///{

static bool isInLastWarp() {
  uint32_t MainTId = (mapping::getNumberOfProcessorElements() - 1) &
                     ~(mapping::getWarpSize() - 1);
  return mapping::getThreadIdInBlock() == MainTId;
}

bool mapping::isMainThreadInGenericMode(bool IsSPMD) {
  if (IsSPMD || icv::Level)
    return false;

  // Check if this is the last warp in the block.
  return isInLastWarp();
}

bool mapping::isMainThreadInGenericMode() {
  return mapping::isMainThreadInGenericMode(mapping::isSPMDMode());
}

bool mapping::isInitialThreadInLevel0(bool IsSPMD) {
  if (IsSPMD)
    return mapping::getThreadIdInBlock() == 0;
  return isInLastWarp();
}

bool mapping::isLeaderInWarp() {
  __kmpc_impl_lanemask_t Active = mapping::activemask();
  __kmpc_impl_lanemask_t LaneMaskLT = mapping::lanemaskLT();
  return utils::popc(Active & LaneMaskLT) == 0;
}

LaneMaskTy mapping::activemask() { return impl::activemask(); }

LaneMaskTy mapping::lanemaskLT() { return impl::lanemaskLT(); }

LaneMaskTy mapping::lanemaskGT() { return impl::lanemaskGT(); }

uint32_t mapping::getThreadIdInWarp() {
  uint32_t ThreadIdInWarp = impl::getThreadIdInWarp();
  ASSERT(ThreadIdInWarp < impl::getWarpSize());
  return ThreadIdInWarp;
}

uint32_t mapping::getThreadIdInBlock(int Dim) {
  uint32_t ThreadIdInBlock = impl::getThreadIdInBlock(Dim);
  ASSERT(ThreadIdInBlock < impl::getNumHardwareThreadsInBlock(Dim));
  return ThreadIdInBlock;
}

uint32_t mapping::getWarpSize() { return impl::getWarpSize(); }

uint32_t mapping::getBlockSize(bool IsSPMD) {
  uint32_t BlockSize = mapping::getNumberOfProcessorElements() -
                       (!IsSPMD * impl::getWarpSize());
  return BlockSize;
}
uint32_t mapping::getBlockSize() {
  return mapping::getBlockSize(mapping::isSPMDMode());
}

uint32_t mapping::getKernelSize() { return impl::getKernelSize(); }

uint32_t mapping::getWarpId() {
  uint32_t WarpID = impl::getWarpId();
  ASSERT(WarpID < impl::getNumberOfWarpsInBlock());
  return WarpID;
}

uint32_t mapping::getBlockId(int Dim) {
  uint32_t BlockId = impl::getBlockId(Dim);
  ASSERT(BlockId < impl::getNumberOfBlocks(Dim));
  return BlockId;
}

uint32_t mapping::getNumberOfWarpsInBlock() {
  uint32_t NumberOfWarpsInBlocks = impl::getNumberOfWarpsInBlock();
  ASSERT(impl::getWarpId() < NumberOfWarpsInBlocks);
  return NumberOfWarpsInBlocks;
}

uint32_t mapping::getNumberOfBlocks(int Dim) {
  uint32_t NumberOfBlocks = impl::getNumberOfBlocks(Dim);
  ASSERT(impl::getBlockId() < NumberOfBlocks);
  return NumberOfBlocks;
}

uint32_t mapping::getNumberOfProcessorElements() {
  uint32_t NumberOfProcessorElements = impl::getNumHardwareThreadsInBlock();
  ASSERT(impl::getThreadIdInBlock() < NumberOfProcessorElements);
  return NumberOfProcessorElements;
}

///}

/// Execution mode
///
///{

// TODO: This is a workaround for initialization coming from kernels outside of
//       the TU. We will need to solve this more correctly in the future.
int __attribute__((used, retain, weak)) SHARED(IsSPMDMode);

void mapping::init(bool IsSPMD) {
  if (mapping::isInitialThreadInLevel0(IsSPMD))
    IsSPMDMode = IsSPMD;
}

bool mapping::isSPMDMode() { return IsSPMDMode; }

bool mapping::isGenericMode() { return !isSPMDMode(); }
///}

extern "C" {
__attribute__((noinline)) uint32_t __kmpc_get_hardware_thread_id_in_block() {
  FunctionTracingRAII();
  return mapping::getThreadIdInBlock();
}
__attribute__((noinline)) uint32_t __kmpc_get_hardware_thread_id_in_block_x() {
  return mapping::getThreadIdInBlock(/* Dim */ 0);
}
__attribute__((noinline)) uint32_t __kmpc_get_hardware_thread_id_in_block_y() {
  return mapping::getThreadIdInBlock(/* Dim */ 1);
}
__attribute__((noinline)) uint32_t __kmpc_get_hardware_thread_id_in_block_z() {
  return mapping::getThreadIdInBlock(/* Dim */ 2);
}
__attribute__((noinline)) uint32_t __kmpc_get_hardware_block_id_in_grid_x() {
  return mapping::getBlockId(/* Dim */ 0);
}
__attribute__((noinline)) uint32_t __kmpc_get_hardware_block_id_in_grid_y() {
  return mapping::getBlockId(/* Dim */ 1);
}
__attribute__((noinline)) uint32_t __kmpc_get_hardware_block_id_in_grid_z() {
  return mapping::getBlockId(/* Dim */ 2);
}
__attribute__((noinline)) uint32_t __kmpc_get_hardware_num_threads_in_block() {
  FunctionTracingRAII();
  return impl::getNumHardwareThreadsInBlock();
}
__attribute__((noinline)) uint32_t __kmpc_get_hardware_num_threads_in_block_x() {
  return impl::getNumHardwareThreadsInBlock(/* Dim */ 0);
}
__attribute__((noinline)) uint32_t __kmpc_get_hardware_num_threads_in_block_y() {
  return impl::getNumHardwareThreadsInBlock(/* Dim */ 1);
}
__attribute__((noinline)) uint32_t __kmpc_get_hardware_num_threads_in_block_z() {
  return impl::getNumHardwareThreadsInBlock(/* Dim */ 2);
}
__attribute__((noinline)) uint32_t __kmpc_get_hardware_num_blocks_in_grid_x() {
  return impl::getNumberOfBlocks(/* Dim */ 0);
}
__attribute__((noinline)) uint32_t __kmpc_get_hardware_num_blocks_in_grid_y() {
  return impl::getNumberOfBlocks(/* Dim */ 1);
}
__attribute__((noinline)) uint32_t __kmpc_get_hardware_num_blocks_in_grid_z() {
  return impl::getNumberOfBlocks(/* Dim */ 2);
}

__attribute__((noinline)) uint32_t __kmpc_get_warp_size() {
  FunctionTracingRAII();
  return impl::getWarpSize();
}
}
#pragma omp end declare target
