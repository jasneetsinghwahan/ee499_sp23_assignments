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

# Print the system info to the console
echo "pwd: $PWD"
echo "kernel version: $kernel_version"
echo "clang version: $clang_version"
echo "bison version: $bison_version"
echo "LLVM version: $llvm_version"
echo "cMake version: $cmake_version"
echo "gcc version: $gcc_version"

# Save the system info to a file
echo "Date: $now" >> system_info.txt
echo "Kernel version: $kernel_version" >> system_info.txt
echo "Clang version: $clang_version" >> system_info.txt
echo "Bison version: $bison_version" >> system_info.txt
echo "LLVM version: $llvm_version" >> system_info.txt
echo "CMake version: $cmake_version" >> system_info.txt
echo "GCC version: $gcc_version" >> system_info.txt
