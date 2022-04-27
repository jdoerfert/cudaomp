###===----------------------------------------------------------------------===##
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Build the VGPU plugin for virtual GPU offloading.
#
##===----------------------------------------------------------------------===#

if (NOT(LIBOMPTARGET_ENABLE_EXPERIMENTAL_VGPU_PLUGIN))
  return()
endif()

macro(build_generic_elf64_vgpu tmachine tmachine_name tmachine_libname tmachine_triple elf_machine_id)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "${tmachine}$")
  if(LIBOMPTARGET_DEP_LIBELF_FOUND)
    if(LIBOMPTARGET_DEP_LIBFFI_FOUND)
      libomptarget_say("Building ${tmachine_triple}-${tmachine_name} offloading plugin.")

      include_directories(${LIBOMPTARGET_DEP_LIBFFI_INCLUDE_DIR})
      include_directories(${LIBOMPTARGET_DEP_LIBELF_INCLUDE_DIR})
      include_directories(${LIBOMPTARGET_INCLUDE_DIR})

      # Define macro to be used as prefix of the runtime messages for this target.
      add_definitions("-DTARGET_NAME=${tmachine_name}")

      # Define macro with the ELF ID for this target.
      add_definitions("-DTARGET_ELF_ID=${elf_machine_id}")

      add_library("omptarget.rtl.${tmachine_libname}" SHARED
              ${CMAKE_CURRENT_SOURCE_DIR}/src/rtl.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/src/ThreadEnvironment.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/src/ThreadEnvironmentImpl.cpp)

      # Install plugin under the lib destination folder.
      install(TARGETS "omptarget.rtl.${tmachine_libname}"
              LIBRARY DESTINATION "${OPENMP_INSTALL_LIBDIR}")

      set_target_properties("omptarget.rtl.${tmachine_libname}" PROPERTIES CXX_STANDARD 20)

      target_link_libraries(
              "omptarget.rtl.${tmachine_libname}"
              elf_common
              ${LIBOMPTARGET_DEP_LIBFFI_LIBRARIES}
              ${LIBOMPTARGET_DEP_LIBELF_LIBRARIES}
              dl
              # ${OPENMP_PTHREAD_LIB}
              "-rdynamic"
              "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exports"
              )

      list(APPEND LIBOMPTARGET_TESTED_PLUGINS
              "omptarget.rtl.${tmachine_libname}")

      # Report to the parent scope that we are building a plugin.
      set(LIBOMPTARGET_SYSTEM_TARGETS
              "${LIBOMPTARGET_SYSTEM_TARGETS} ${tmachine_triple}" PARENT_SCOPE)
      set(LIBOMPTARGET_TESTED_PLUGINS
              "${LIBOMPTARGET_TESTED_PLUGINS}" PARENT_SCOPE)
    else(LIBOMPTARGET_DEP_LIBFFI_FOUND)
      libomptarget_say("Not building ${tmachine_name} offloading plugin: libffi dependency not found.")
    endif(LIBOMPTARGET_DEP_LIBFFI_FOUND)
  else(LIBOMPTARGET_DEP_LIBELF_FOUND)
    libomptarget_say("Not building ${tmachine_name} offloading plugin: libelf dependency not found.")
  endif(LIBOMPTARGET_DEP_LIBELF_FOUND)
else()
  libomptarget_say("Not building ${tmachine_name}-vgpu offloading plugin: machine not found in the system.")
endif()
endmacro()

build_generic_elf64_vgpu("x86_64" "vgpu" "vgpu" "x86_64-vgpu" "62")

