cmake_minimum_required(VERSION 3.22)
project(nlm_denoise)

set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_SYSTEM_VERSION 19)
set(CMAKE_ANDROID_ARCH_ABI arm64-v8a)
set(CMAKE_ANDROID_STL_TYPE gnustl_shared)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O0 -fopenmp -march=armv8-a")

find_package(OpenMP REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}
        /home/ray/Android/Sdk/ndk/25.1.8937393/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/14.0.6/)

add_executable(nlm_denoise main.cpp)
target_link_libraries(nlm_denoise
        OpenMP::OpenMP_CXX)
