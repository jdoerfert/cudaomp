##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Build the DeviceRTL for all toolchains that are available
#
##===----------------------------------------------------------------------===##

set(LIBOMPTARGET_BUILD_DEVICERTL_BCLIB TRUE CACHE BOOL
  "Can be set to false to disable building this library.")

if (NOT LIBOMPTARGET_BUILD_DEVICERTL_BCLIB)
  libomptarget_say("Not building DeviceRTL: Disabled by LIBOMPTARGET_BUILD_DEVICERTL_BCLIB")
  return()
endif()

if (LLVM_DIR)
  # Builds that use pre-installed LLVM have LLVM_DIR set.
  find_program(CLANG_TOOL clang PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
  find_program(LINK_TOOL llvm-link PATHS ${LLVM_TOOLS_BINARY_DIR}
    NO_DEFAULT_PATH)
  find_program(OPT_TOOL opt PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
  if ((NOT CLANG_TOOL) OR (NOT LINK_TOOL) OR (NOT OPT_TOOL))
    libomptarget_say("Not building DeviceLib. Missing clang: ${CLANG_TOOL}, llvm-link: ${LINK_TOOL} or opt: ${OPT_TOOL}")
    return()
  else()
    libomptarget_say("Building DeviceRTL. Using clang: ${CLANG_TOOL}, llvm-link: ${LINK_TOOL} and opt: ${OPT_TOOL}")
  endif()
elseif (LLVM_TOOL_CLANG_BUILD AND NOT CMAKE_CROSSCOMPILING AND NOT OPENMP_STANDALONE_BUILD)
  # LLVM in-tree builds may use CMake target names to discover the tools.
  set(CLANG_TOOL $<TARGET_FILE:clang>)
  set(LINK_TOOL $<TARGET_FILE:llvm-link>)
  set(OPT_TOOL $<TARGET_FILE:opt>)
  libomptarget_say("Building DeviceRTL. Using clang from in-tree build")
else()
  libomptarget_say("Not building DeviceLib. No appropriate clang found")
  return()
endif()

# TODO: This part needs to be refined when libomptarget is going to support
# Windows!
# TODO: This part can also be removed if we can change the clang driver to make
# it support device only compilation.
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(aux_triple x86_64-unknown-linux-gnu)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "ppc64le")
  set(aux_triple powerpc64le-unknown-linux-gnu)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(aux_triple aarch64-unknown-linux-gnu)
else()
  libomptarget_say("Not building DeviceRTL: unknown host arch: ${CMAKE_HOST_SYSTEM_PROCESSOR}")
  return()
endif()

set(devicertl_base_directory ${CMAKE_CURRENT_SOURCE_DIR})
set(include_directory ${devicertl_base_directory}/include)
set(source_directory ${devicertl_base_directory}/src)

# Set flags for LLVM Bitcode compilation.
set(bc_flags -S -x c++ -std=c++17 -fvisibility=hidden
             -fno-exceptions -fno-rtti -O1
             -Xclang -emit-llvm-bc
             -Xclang -aux-triple -Xclang ${aux_triple}
             -I${include_directory}
             -I${devicertl_base_directory}/../include
             ${LIBOMPTARGET_LLVM_INCLUDE_DIRS_DEVICERTL}
)

function(compileDeviceRTLLibrary src_files target_name output_name)
  set(target_bc_flags ${ARGN})

  set(bc_files "")
  foreach(src ${src_files})
    get_filename_component(infile ${src} ABSOLUTE)
    get_filename_component(outfile ${src} NAME)
    set(outfile "${outfile}-${target_name}.bc")

    add_custom_command(OUTPUT ${outfile}
      COMMAND ${CLANG_TOOL}
      ${bc_flags}
      ${target_bc_flags}
      ${infile} -o ${outfile}
      DEPENDS ${infile} ${include_files}
      IMPLICIT_DEPENDS CXX ${infile}
      COMMENT "Building LLVM bitcode ${outfile}"
      VERBATIM
    )
    if("${CLANG_TOOL}" STREQUAL "$<TARGET_FILE:clang>")
      # Add a file-level dependency to ensure that clang is up-to-date.
      # By default, add_custom_command only builds clang if the
      # executable is missing.
      add_custom_command(OUTPUT ${outfile}
        DEPENDS clang
        APPEND
      )
    endif()
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${outfile})

    list(APPEND bc_files ${outfile})
  endforeach()

  set(bclib_name "libomptarget-${target_name}-${output_name}.bc")

  # Link to a bitcode library.
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/linked_${bclib_name}
      COMMAND ${LINK_TOOL}
        -o ${CMAKE_CURRENT_BINARY_DIR}/linked_${bclib_name} ${bc_files}
      DEPENDS ${bc_files}
      COMMENT "Linking LLVM bitcode ${bclib_name}"
  )

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name}
      COMMAND ${OPT_TOOL} ${link_opt_flags} ${CMAKE_CURRENT_BINARY_DIR}/linked_${bclib_name}
                      -o ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name}
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/linked_${bclib_name}
      COMMENT "Optimizing LLVM bitcode ${bclib_name}"
  )

  # Add a file-level dependency to ensure that llvm-link and opt are up-to-date.
  # By default, add_custom_command only builds the tool if the executable is missing
  if("${LINK_TOOL}" STREQUAL "$<TARGET_FILE:llvm-link>")
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/linked_${bclib_name}
      DEPENDS llvm-link
      APPEND)
  endif()
  if("${OPT_TOOL}" STREQUAL "$<TARGET_FILE:opt>")
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name}
      DEPENDS opt
      APPEND)
  endif()

  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${bclib_name})

  set(bclib_target_name "omptarget-${target_name}-${output_name}-bc")

  add_custom_target(${bclib_target_name} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name})

  # Copy library to destination.
  add_custom_command(TARGET ${bclib_target_name} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name}
                    ${LIBOMPTARGET_LIBRARY_DIR})

  # Install bitcode library under the lib destination folder.
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${bclib_name} DESTINATION "${OPENMP_INSTALL_LIBDIR}")
endfunction()

set(src_files ${source_directory}/OpenMPMath.cpp)

# Generate a Bitcode library for the OpenMP math function wrappers.
compileDeviceRTLLibrary(${src_files} nvptx "math-wrappers" -target nvptx64-nvidia-cuda -nogpulib)
compileDeviceRTLLibrary(${src_files} amdgpu "math-wrappers" -target amdgcn-amd-amdhsa -D__AMDGCN__ -nogpulib)

set(src_files ${source_directory}/DeviceLibm.cpp)

# Generate a Bitcode library for the device math function wrappers.
compileDeviceRTLLibrary(${src_files} nvptx "libm" -target nvptx64-nvidia-cuda -D__CUDA__ -nogpulib)
compileDeviceRTLLibrary(${src_files} amdgpu "libm" -target amdgcn-amd-amdhsa -D__AMDGCN__ -nogpulib)
