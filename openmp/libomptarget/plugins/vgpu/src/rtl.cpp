//===------RTLs/vgpu/src/rtl.cpp - Target RTLs Implementation ----- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL for virtual (x86) GPU
//
//===----------------------------------------------------------------------===//

#include <barrier>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <ffi.h>
#include <functional>
#include <gelf.h>
#include <link.h>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "Debug.h"
#include "ThreadEnvironment.h"
#include "ThreadEnvironmentImpl.h"
#include "omptarget.h"
#include "omptargetplugin.h"

#ifndef TARGET_NAME
#define TARGET_NAME Generic ELF - 64bit
#endif
#define DEBUG_PREFIX "TARGET " GETNAME(TARGET_NAME) " RTL"

#ifndef TARGET_ELF_ID
#define TARGET_ELF_ID 0
#endif

#include "elf_common.h"

#define OFFLOADSECTIONNAME "omp_offloading_entries"

#define DEBUG false

struct FFICallTy {
  ffi_cif CIF;
  std::vector<ffi_type *> ArgsTypes;
  std::vector<void *> Args;
  std::vector<void *> Ptrs;
  void (*Entry)(void);

  FFICallTy(int32_t ArgNum, void **TgtArgs, ptrdiff_t *TgtOffsets,
            void *TgtEntryPtr)
      : ArgsTypes(ArgNum, &ffi_type_pointer), Args(ArgNum), Ptrs(ArgNum) {
    for (int32_t i = 0; i < ArgNum; ++i) {
      Ptrs[i] = (void *)((intptr_t)TgtArgs[i] + TgtOffsets[i]);
      Args[i] = &Ptrs[i];
    }

    ffi_status status = ffi_prep_cif(&CIF, FFI_DEFAULT_ABI, ArgNum,
                                     &ffi_type_void, &ArgsTypes[0]);

    assert(status == FFI_OK && "Unable to prepare target launch!");

    *((void **)&Entry) = TgtEntryPtr;
  }
};

/// Array of Dynamic libraries loaded for this target.
struct DynLibTy {
  char *FileName;
  void *Handle;
};

/// Keep entries table per device.
struct FuncOrGblEntryTy {
  __tgt_target_table Table;
};

thread_local ThreadEnvironmentTy *ThreadEnvironment;

/// Class containing all the device information.
class RTLDeviceInfoTy {
  std::vector<std::list<FuncOrGblEntryTy>> FuncGblEntries;

public:
  std::list<DynLibTy> DynLibs;

  // Record entry point associated with device.
  void createOffloadTable(int32_t device_id, __tgt_offload_entry *begin,
                          __tgt_offload_entry *end) {
    assert(device_id < (int32_t)FuncGblEntries.size() &&
           "Unexpected device id!");
    FuncGblEntries[device_id].emplace_back();
    FuncOrGblEntryTy &E = FuncGblEntries[device_id].back();

    E.Table.EntriesBegin = begin;
    E.Table.EntriesEnd = end;
  }

  // Return true if the entry is associated with device.
  bool findOffloadEntry(int32_t device_id, void *addr) {
    assert(device_id < (int32_t)FuncGblEntries.size() &&
           "Unexpected device id!");
    FuncOrGblEntryTy &E = FuncGblEntries[device_id].back();

    for (__tgt_offload_entry *i = E.Table.EntriesBegin, *e = E.Table.EntriesEnd;
         i < e; ++i) {
      if (i->addr == addr)
        return true;
    }

    return false;
  }

  // Return the pointer to the target entries table.
  __tgt_target_table *getOffloadEntriesTable(int32_t device_id) {
    assert(device_id < (int32_t)FuncGblEntries.size() &&
           "Unexpected device id!");
    FuncOrGblEntryTy &E = FuncGblEntries[device_id].back();

    return &E.Table;
  }

  RTLDeviceInfoTy() : FuncGblEntries(1) {}

  ~RTLDeviceInfoTy() {
    // Close dynamic libraries
    for (auto &lib : DynLibs) {
      if (lib.Handle) {
        dlclose(lib.Handle);
        remove(lib.FileName);
      }
    }
  }
};

static RTLDeviceInfoTy DeviceInfo;

std::vector<CTAEnvironmentTy *> CTAEnvironments;
std::vector<WarpEnvironmentTy *> WarpEnvironments;

struct VGPUTy {
  struct KernelTy {
    FFICallTy *Call;
    int NumTeams;

    KernelTy(FFICallTy *Call, int NumTeams) : Call(Call), NumTeams(NumTeams) {}
  };

  struct VGPUStreamTy {
    std::queue<KernelTy> Kernels;
    std::mutex Mtx;

    void emplace(FFICallTy *Call, int NumTeams) {
      std::lock_guard Guard(Mtx);
      Kernels.emplace(Call, NumTeams);
    }

    KernelTy front() {
      std::lock_guard Guard(Mtx);
      return Kernels.front();
    }

    void pop() {
      std::lock_guard Guard(Mtx);
      Kernels.pop();
    }

    bool empty() {
      std::lock_guard Guard(Mtx);
      return Kernels.empty();
    }
  };

  struct AsyncInfoQueueTy {
    std::deque<__tgt_async_info *> Streams;
    std::mutex Mtx;

    bool empty() {
      std::lock_guard Guard(Mtx);
      return Streams.empty();
    }

    __tgt_async_info *front() {
      std::lock_guard Guard(Mtx);
      return Streams.front();
    }

    void pop() {
      std::lock_guard Guard(Mtx);
      Streams.pop_front();
    }

    void emplace(__tgt_async_info *AsyncInfo) {
      std::lock_guard Guard(Mtx);
      Streams.emplace_back(AsyncInfo);
    }
  } ExecutionQueue;

  VGPUStreamTy *getStream(__tgt_async_info *AsyncInfo) {
    assert(AsyncInfo != nullptr && "async_info ptr was null");

    if (!AsyncInfo->Queue)
      AsyncInfo->Queue = new VGPUStreamTy();

    return reinterpret_cast<VGPUStreamTy *>(AsyncInfo->Queue);
  }

  std::atomic<bool> Running;
  std::vector<std::thread> Threads;
  int WarpsPerCTA = -1;
  int NumCTAs = -1;
  int NumThreads = -1;

  std::unique_ptr<std::barrier<std::function<void(void)>>> Barrier;
  std::condition_variable WorkAvailable;
  std::mutex WorkDoneMtx;
  std::condition_variable WorkDone;

  void configureArchitecture() {
    int ThreadsPerWarp = -1;

    if (const char *Env = std::getenv("VGPU_NUM_THREADS"))
      NumThreads = std::stoi(Env);
    if (const char *Env = std::getenv("VGPU_THREADS_PER_WARP"))
      ThreadsPerWarp = std::stoi(Env);
    if (const char *Env = std::getenv("VGPU_WARPS_PER_CTA"))
      WarpsPerCTA = std::stoi(Env);

    if (NumThreads == -1)
      NumThreads = std::thread::hardware_concurrency();
    if (ThreadsPerWarp == -1)
      ThreadsPerWarp = NumThreads;
    if (WarpsPerCTA == -1)
      WarpsPerCTA = 1;

    NumCTAs = NumThreads / (ThreadsPerWarp * WarpsPerCTA);

    assert(NumThreads % ThreadsPerWarp == 0 && NumThreads % WarpsPerCTA == 0 &&
           "Invalid VGPU Config");

    DP("NumThreads: %d, ThreadsPerWarp: %d, WarpsPerCTA: %d\n", NumThreads,
       ThreadsPerWarp, WarpsPerCTA);

    CTAEnvironmentTy::configure(NumThreads, NumCTAs);
    WarpEnvironmentTy::configure(ThreadsPerWarp);
  }

  VGPUTy() : Running(true) {
    configureArchitecture();

    Barrier = std::make_unique<BarrierTy>(NumThreads, []() {});
    Threads.reserve(NumThreads);

    auto GlobalThreadIdx = 0;
    for (auto CTAIdx = 0; CTAIdx < CTAEnvironmentTy::NumCTAs; CTAIdx++) {
      auto *CTAEnv = new CTAEnvironmentTy();
      for (auto WarpIdx = 0; WarpIdx < WarpsPerCTA; WarpIdx++) {
        auto *WarpEnv = new WarpEnvironmentTy();
        for (auto ThreadIdx = 0; ThreadIdx < WarpEnvironmentTy::ThreadsPerWarp;
             ThreadIdx++) {
          Threads.emplace_back([this, GlobalThreadIdx, CTAEnv, WarpEnv]() {
            ThreadEnvironment = new ThreadEnvironmentTy(WarpEnv, CTAEnv);
            while (Running) {
              {
                std::unique_lock<std::mutex> UniqueLock(ExecutionQueue.Mtx);

                WorkAvailable.wait(UniqueLock, [&]() {
                  if (!Running)
                    return true;

                  bool IsEmpty = ExecutionQueue.Streams.empty();

                  return !IsEmpty;
                });
              }

              if (ExecutionQueue.empty())
                continue;

              while (!ExecutionQueue.empty()) {
                auto *Stream = getStream(ExecutionQueue.front());
                while (!Stream->empty()) {
                  auto [Call, NumTeams] = Stream->front();

                  runKernel(CTAEnv, Call, NumTeams);

                  if (GlobalThreadIdx == 0) {
                    Stream->pop();
                    delete Call;
                  }

                  Barrier->arrive_and_wait();
                }
                if (GlobalThreadIdx == 0) {
                  ExecutionQueue.pop();
                  WorkDone.notify_all();
                }
                Barrier->arrive_and_wait();
              }
            }
            delete ThreadEnvironment;
          });
          GlobalThreadIdx = (GlobalThreadIdx + 1) % NumThreads;
        }
        WarpEnvironments.push_back(WarpEnv);
      }
      CTAEnvironments.push_back(CTAEnv);
    }
  }

  void runKernel(CTAEnvironmentTy *CTAEnv, FFICallTy *Call, int NumTeams) {
    unsigned TeamIdx = 0;
    while (TeamIdx < NumTeams) {
      if (CTAEnv->getId() < NumTeams) {
        ThreadEnvironment->setBlockEnv(
            new ThreadBlockEnvironmentTy(TeamIdx + CTAEnv->getId(), NumTeams));
        ffi_call(&Call->CIF, Call->Entry, NULL, &(Call->Args)[0]);
        ThreadEnvironment->resetBlockEnv();
      }
      Barrier->arrive_and_wait();
      TeamIdx += NumCTAs;
    }
  }

  ~VGPUTy() {
    awaitAll();

    Running = false;
    WorkAvailable.notify_all();

    for (auto &Thread : Threads) {
      if (Thread.joinable())
        Thread.join();
    }

    for (auto *CTAEnv : CTAEnvironments)
      delete CTAEnv;

    for (auto *WarpEnv : WarpEnvironments)
      delete WarpEnv;
  }

  void await(__tgt_async_info *AsyncInfo) {
    std::unique_lock UniqueLock(getStream(AsyncInfo)->Mtx);
    WorkDone.wait(UniqueLock,
                  [&]() { return getStream(AsyncInfo)->Kernels.empty(); });
  }

  void awaitAll() {
    while (!ExecutionQueue.empty()) {
      await(ExecutionQueue.front());
    }
  }

  void scheduleAsync(__tgt_async_info *AsyncInfo, FFICallTy *Call,
                     int NumTeams) {
    if (NumTeams == 0)
      NumTeams = NumCTAs;
    auto *Stream = getStream(AsyncInfo);
    Stream->emplace(Call, NumTeams);
    ExecutionQueue.emplace(AsyncInfo);
    WorkAvailable.notify_all();
  }
};

VGPUTy VGPU;

#ifdef __cplusplus
extern "C" {
#endif

int32_t __tgt_rtl_is_valid_binary(__tgt_device_image *image) {
// If we don't have a valid ELF ID we can just fail.
#if TARGET_ELF_ID < 1
  return 0;
#else
  return elf_check_machine(image, TARGET_ELF_ID);
#endif
}

int32_t __tgt_rtl_number_of_devices() { return 1; }

int32_t __tgt_rtl_init_device(int32_t device_id) { return OFFLOAD_SUCCESS; }

__tgt_target_table *__tgt_rtl_load_binary(int32_t device_id,
                                          __tgt_device_image *image) {

  DP("Dev %d: load binary from " DPxMOD " image\n", device_id,
     DPxPTR(image->ImageStart));

  assert(device_id >= 0 && device_id < 1 && "bad dev id");

  size_t ImageSize = (size_t)image->ImageEnd - (size_t)image->ImageStart;
  size_t NumEntries = (size_t)(image->EntriesEnd - image->EntriesBegin);
  DP("Expecting to have %zd entries defined.\n", NumEntries);

  // Is the library version incompatible with the header file?
  if (elf_version(EV_CURRENT) == EV_NONE) {
    DP("Incompatible ELF library!\n");
    return NULL;
  }

  // Obtain elf handler
  Elf *e = elf_memory((char *)image->ImageStart, ImageSize);
  if (!e) {
    DP("Unable to get ELF handle: %s!\n", elf_errmsg(-1));
    return NULL;
  }

  if (elf_kind(e) != ELF_K_ELF) {
    DP("Invalid Elf kind!\n");
    elf_end(e);
    return NULL;
  }

  // Find the entries section offset
  Elf_Scn *section = 0;
  Elf64_Off entries_offset = 0;

  size_t shstrndx;

  if (elf_getshdrstrndx(e, &shstrndx)) {
    DP("Unable to get ELF strings index!\n");
    elf_end(e);
    return NULL;
  }

  while ((section = elf_nextscn(e, section))) {
    GElf_Shdr hdr;
    gelf_getshdr(section, &hdr);

    if (!strcmp(elf_strptr(e, shstrndx, hdr.sh_name), OFFLOADSECTIONNAME)) {
      entries_offset = hdr.sh_addr;
      break;
    }
  }

  if (!entries_offset) {
    DP("Entries Section Offset Not Found\n");
    elf_end(e);
    return NULL;
  }

  DP("Offset of entries section is (" DPxMOD ").\n", DPxPTR(entries_offset));

  // load dynamic library and get the entry points. We use the dl library
  // to do the loading of the library, but we could do it directly to avoid
  // the dump to the temporary file.
  //
  // 1) Create tmp file with the library contents.
  // 2) Use dlopen to load the file and dlsym to retrieve the symbols.
  char tmp_name[] = "/tmp/tmpfile_XXXXXX";
  int tmp_fd = mkstemp(tmp_name);

  if (tmp_fd == -1) {
    elf_end(e);
    return NULL;
  }

  FILE *ftmp = fdopen(tmp_fd, "wb");

  if (!ftmp) {
    elf_end(e);
    return NULL;
  }

  fwrite(image->ImageStart, ImageSize, 1, ftmp);
  fclose(ftmp);

  DynLibTy Lib = {tmp_name, dlopen(tmp_name, RTLD_NOW | RTLD_GLOBAL)};

  if (!Lib.Handle) {
    DP("Target library loading error: %s\n", dlerror());
    elf_end(e);
    return NULL;
  }

  DeviceInfo.DynLibs.push_back(Lib);

  struct link_map *libInfo = (struct link_map *)Lib.Handle;

  // The place where the entries info is loaded is the library base address
  // plus the offset determined from the ELF file.
  Elf64_Addr entries_addr = libInfo->l_addr + entries_offset;

  DP("Pointer to first entry to be loaded is (" DPxMOD ").\n",
     DPxPTR(entries_addr));

  // Table of pointers to all the entries in the target.
  __tgt_offload_entry *entries_table = (__tgt_offload_entry *)entries_addr;

  __tgt_offload_entry *entries_begin = &entries_table[0];
  __tgt_offload_entry *entries_end = entries_begin + NumEntries;

  if (!entries_begin) {
    DP("Can't obtain entries begin\n");
    elf_end(e);
    return NULL;
  }

  DP("Entries table range is (" DPxMOD ")->(" DPxMOD ")\n",
     DPxPTR(entries_begin), DPxPTR(entries_end));
  DeviceInfo.createOffloadTable(device_id, entries_begin, entries_end);

  elf_end(e);

  return DeviceInfo.getOffloadEntriesTable(device_id);
}

// Sample implementation of explicit memory allocator. For this plugin all
// kinds are equivalent to each other.
void *__tgt_rtl_data_alloc(int32_t device_id, int64_t size, void *hst_ptr,
                           int32_t kind) {
  void *ptr = NULL;

  switch (kind) {
  case TARGET_ALLOC_DEVICE:
  case TARGET_ALLOC_HOST:
  case TARGET_ALLOC_SHARED:
  case TARGET_ALLOC_DEFAULT:
    ptr = malloc(size);
    break;
  default:
    REPORT("Invalid target data allocation kind");
  }

  return ptr;
}

int32_t __tgt_rtl_data_submit(int32_t device_id, void *tgt_ptr, void *hst_ptr,
                              int64_t size) {
  VGPU.awaitAll();
  memcpy(tgt_ptr, hst_ptr, size);
  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_data_retrieve(int32_t device_id, void *hst_ptr, void *tgt_ptr,
                                int64_t size) {
  VGPU.awaitAll();
  memcpy(hst_ptr, tgt_ptr, size);
  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_data_delete(int32_t device_id, void *tgt_ptr) {
  free(tgt_ptr);
  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_synchronize(int32_t device_id, __tgt_async_info *async_info) {
  VGPU.await(async_info);
  delete (VGPUTy::VGPUStreamTy *)async_info->Queue;
  async_info->Queue = nullptr;
  return 0;
}

int32_t __tgt_rtl_run_target_team_region(int32_t device_id, void *tgt_entry_ptr,
                                         void **tgt_args,
                                         ptrdiff_t *tgt_offsets,
                                         int32_t arg_num, int32_t team_num,
                                         int32_t thread_limit,
                                         uint64_t loop_tripcount) {
  __tgt_async_info AsyncInfo;
  int rc = __tgt_rtl_run_target_team_region_async(
      device_id, tgt_entry_ptr, tgt_args, tgt_offsets, arg_num, team_num,
      thread_limit, loop_tripcount, &AsyncInfo);

  if (rc != OFFLOAD_SUCCESS)
    return OFFLOAD_FAIL;

  return __tgt_rtl_synchronize(device_id, &AsyncInfo);
}

int32_t __tgt_rtl_run_target_team_region_async(
    int32_t device_id, void *tgt_entry_ptr, void **tgt_args,
    ptrdiff_t *tgt_offsets, int32_t arg_num, int32_t team_num,
    int32_t thread_limit, uint64_t loop_tripcount /*not used*/,
    __tgt_async_info *async_info) {
  DP("Running entry point at " DPxMOD "...\n", DPxPTR(tgt_entry_ptr));

  auto Call = new FFICallTy(arg_num, tgt_args, tgt_offsets, tgt_entry_ptr);

  VGPU.scheduleAsync(async_info, std::move(Call), team_num);
  return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_run_target_region(int32_t device_id, void *tgt_entry_ptr,
                                    void **tgt_args, ptrdiff_t *tgt_offsets,
                                    int32_t arg_num) {
  return __tgt_rtl_run_target_team_region(device_id, tgt_entry_ptr, tgt_args,
                                          tgt_offsets, arg_num, 1, 1, 0);
}

int32_t __tgt_rtl_run_target_region_async(int32_t device_id,
                                          void *tgt_entry_ptr, void **tgt_args,
                                          ptrdiff_t *tgt_offsets,
                                          int32_t arg_num,
                                          __tgt_async_info *async_info) {
  return __tgt_rtl_run_target_team_region_async(device_id, tgt_entry_ptr,
                                                tgt_args, tgt_offsets, arg_num,
                                                1, 1, 0, async_info);
}

#ifdef __cplusplus
}
#endif
