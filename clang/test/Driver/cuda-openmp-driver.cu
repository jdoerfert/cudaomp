// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// RUN: %clang -### -target x86_64-linux-gnu -nocudalib -ccc-print-bindings -fgpu-rdc \
// RUN:        -foffload-new-driver --offload-arch=sm_35 --offload-arch=sm_70 %s 2>&1 \
// RUN: | FileCheck -check-prefix CHECK %s

// CHECK: "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[PTX_SM_35:.+]]"
// CHECK: "nvptx64-nvidia-cuda" - "NVPTX::Assembler", inputs: ["[[PTX_SM_35]]"], output: "[[CUBIN_SM_35:.+]]"
// CHECK: "nvptx64-nvidia-cuda" - "NVPTX::Linker", inputs: ["[[CUBIN_SM_35]]", "[[PTX_SM_35]]"], output: "[[FATBIN_SM_35:.+]]"
// CHECK: "nvptx64-nvidia-cuda" - "clang", inputs: ["[[INPUT]]"], output: "[[PTX_SM_70:.+]]"
// CHECK: "nvptx64-nvidia-cuda" - "NVPTX::Assembler", inputs: ["[[PTX_SM_70:.+]]"], output: "[[CUBIN_SM_70:.+]]"
// CHECK: "nvptx64-nvidia-cuda" - "NVPTX::Linker", inputs: ["[[CUBIN_SM_70]]", "[[PTX_SM_70:.+]]"], output: "[[FATBIN_SM_70:.+]]"
// CHECK: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT]]", "[[FATBIN_SM_35]]", "[[FATBIN_SM_70]]"], output: "[[HOST_OBJ:.+]]"
// CHECK: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["[[HOST_OBJ]]"], output: "a.out"
