cmake_minimum_required(VERSION 3.16)
project(mjolnir C)

set(CMAKE_C_STANDARD 11)

# Include directories
include_directories(include)

# Math library: needed on Linux/macOS, not on MSVC
if(NOT MSVC)
    set(MATH_LIB m)
else()
    set(MATH_LIB "")
    # MSVC: use _CRT_SECURE_NO_WARNINGS to avoid snprintf warnings
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# CPU reference library
add_library(mjolnir_cpu STATIC
    src/cpu/geometric_product.c
    src/cpu/grade_ops.c
    src/cpu/sandwich.c
    src/cpu/bivector_exp.c
    src/cpu/batch_ops.c
)
if(MATH_LIB)
    target_link_libraries(mjolnir_cpu ${MATH_LIB})
endif()

# Build shared library too (for FFI)
add_library(mjolnir_cpu_shared SHARED
    src/cpu/geometric_product.c
    src/cpu/grade_ops.c
    src/cpu/sandwich.c
    src/cpu/bivector_exp.c
    src/cpu/batch_ops.c
)
if(MATH_LIB)
    target_link_libraries(mjolnir_cpu_shared ${MATH_LIB})
endif()
# Export all symbols on Windows for DLL
set_target_properties(mjolnir_cpu_shared PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    OUTPUT_NAME mjolnir_cpu
)

# Tests
enable_testing()

add_executable(test_multiplication_table tests/test_multiplication_table.c)
target_link_libraries(test_multiplication_table mjolnir_cpu ${MATH_LIB})
add_test(NAME test_multiplication_table COMMAND test_multiplication_table)

add_executable(test_geometric_product tests/test_geometric_product.c)
target_link_libraries(test_geometric_product mjolnir_cpu ${MATH_LIB})
add_test(NAME test_geometric_product COMMAND test_geometric_product)

add_executable(test_sandwich tests/test_sandwich.c)
target_link_libraries(test_sandwich mjolnir_cpu ${MATH_LIB})
add_test(NAME test_sandwich COMMAND test_sandwich)

add_executable(test_associativity tests/test_associativity.c)
target_link_libraries(test_associativity mjolnir_cpu ${MATH_LIB})
add_test(NAME test_associativity COMMAND test_associativity)

# CUDA kernels (optional)
option(BUILD_CUDA "Build CUDA kernels" OFF)
if(BUILD_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    # Default: A100 (sm_80). Override via -DCMAKE_CUDA_ARCHITECTURES=XX
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "80")
    endif()

    set(CUDA_SOURCES
        src/cuda/geometric_product.cu
        src/cuda/grade_ops.cu
        src/cuda/sandwich.cu
        src/cuda/bivector_exp.cu
        src/cuda/batch_ops.cu
        src/cuda/fused_kernels.cu
    )

    # Static library
    add_library(mjolnir_cuda STATIC ${CUDA_SOURCES})
    target_include_directories(mjolnir_cuda PUBLIC include)
    target_link_libraries(mjolnir_cuda CUDA::cudart)

    # Shared library for Python FFI
    add_library(mjolnir_cuda_shared SHARED ${CUDA_SOURCES})
    target_include_directories(mjolnir_cuda_shared PUBLIC include)
    set_target_properties(mjolnir_cuda_shared PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        OUTPUT_NAME mjolnir_cuda
    )
    target_link_libraries(mjolnir_cuda_shared CUDA::cudart)

    # --- Persistent training engine (optional) ---
    option(BUILD_PERSISTENT_ENGINE "Build persistent CUDA training engine" ON)
    if(BUILD_PERSISTENT_ENGINE)
        set(ENGINE_SOURCES
            src/cuda/persistent_engine.cu
        )

        # The persistent engine uses cooperative groups, which requires:
        # 1. -rdc=true for relocatable device code
        # 2. Linking against CUDA::cudart for cudaLaunchCooperativeKernel

        # Static library
        add_library(mjolnir_engine STATIC ${ENGINE_SOURCES})
        target_include_directories(mjolnir_engine PUBLIC include src/cuda)
        set_target_properties(mjolnir_engine PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )
        target_link_libraries(mjolnir_engine CUDA::cudart)

        # Shared library for Python FFI (includes both base kernels + engine)
        add_library(mjolnir_engine_shared SHARED ${CUDA_SOURCES} ${ENGINE_SOURCES})
        target_include_directories(mjolnir_engine_shared PUBLIC include src/cuda)
        set_target_properties(mjolnir_engine_shared PROPERTIES
            WINDOWS_EXPORT_ALL_SYMBOLS ON
            OUTPUT_NAME mjolnir_engine
            CUDA_SEPARABLE_COMPILATION ON
        )
        target_link_libraries(mjolnir_engine_shared CUDA::cudart)

        # Keep the engine shared object distinct from the base CUDA FFI library
        # so the runtime can explicitly load the persistent-engine build.
    endif()
endif()
