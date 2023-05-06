#!/bin/bash

# this scripts records the system info

# Get the current date and time
now=$(date +"%Y-%m-%d %H:%M:%S")

# Print the date and time to the console
echo "Date: $now"

# Save the output of each command to a variable
kernel_version=$(uname -r)
clang_version=$(clang --version | head -n 1)
bison_version=$(bison --version | head -n 1)
llvm_version=$(llvm-config --version)
cmake_version=$(cmake --version | head -n 1)
gcc_version=$(gcc --version | head -n 1)
ubuntu_version=$(cat /etc/lsb-release)
linker_version=$(ld -v)
make_version=$(make --version)
util_linux_version=$(fdformat --version)
module_init_tools=$(depmod -V)
packinst_list=$(dpkg -l)

# Print the system info to the console
echo "pwd: $PWD"
echo "kernel version: $kernel_version"
echo "clang version: $clang_version"
echo "bison version: $bison_version"
echo "LLVM version: $llvm_version"
echo "cMake version: $cmake_version"
echo "gcc version: $gcc_version"
echo "linker version: $linker_version"
echo "make version: $make_version"
echo "util-linux version: $util_linux_version"
echo "module-init-tools version: $module_init_tools"
echo "ubuntu version: $ubuntu_version"
echo "dpkg -l recorded"

# Save the system info to a file
echo "Date: $now" >> system_info.txt
echo "Kernel version: $kernel_version" >> system_info.txt
echo "Clang version: $clang_version" >> system_info.txt
echo "Bison version: $bison_version" >> system_info.txt
echo "LLVM version: $llvm_version" >> system_info.txt
echo "CMake version: $cmake_version" >> system_info.txt
echo "GCC version: $gcc_version" >> system_info.txt
echo "linker version: $linker_version" >> system_info.txt
echo "make version: $make_version" >> system_info.txt
echo "util-linux version: $util_linux_version" >> system_info.txt
echo "module-init-tools version: $module_init_tools" >> system_info.txt
echo "ubuntu version: $ubuntu_version" >> system_info.txt
echo "packages installed: $packinst_list" >> system_info.txt
