cmake_minimum_required(VERSION 3.16)
project(runtime_core
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "YGGDRASIL L2 Runtime â€” Clifford algebra computation graph runtime"
)

# ============================================================================
# C++ Standard
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Options
# ============================================================================
option(YGGDRASIL_BUILD_TESTS  "Build unit tests"       ON)
option(YGGDRASIL_HAS_CUDA     "Enable CUDA support"    OFF)

# ============================================================================
# Source files
# ============================================================================
set(YGGDRASIL_SOURCES
    src/device_cpu.cpp
    src/device_cuda.cpp
    src/memory_pool.cpp
    src/graph.cpp
    src/scheduler.cpp
    src/tape.cpp
    src/fusion.cpp
)

set(YGGDRASIL_HEADERS
    include/device.h
    include/memory_pool.h
    include/graph.h
    include/scheduler.h
    include/tape.h
)

# ============================================================================
# Library target
# ============================================================================
add_library(runtime_core STATIC ${YGGDRASIL_SOURCES} ${YGGDRASIL_HEADERS})

target_include_directories(runtime_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Threading support (for scheduler)
find_package(Threads REQUIRED)
target_link_libraries(runtime_core PUBLIC Threads::Threads)

# CUDA compile definition
if(YGGDRASIL_HAS_CUDA)
    target_compile_definitions(runtime_core PUBLIC YGGDRASIL_HAS_CUDA)
    # When CUDA is enabled, you would also do:
    # enable_language(CUDA)
    # find_package(CUDAToolkit REQUIRED)
    # target_link_libraries(runtime_core PUBLIC CUDA::cudart)
endif()

# Compiler warnings
if(MSVC)
    target_compile_options(runtime_core PRIVATE /W4)
else()
    target_compile_options(runtime_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ============================================================================
# Tests
# ============================================================================
if(YGGDRASIL_BUILD_TESTS)
    enable_testing()

    # test_graph
    add_executable(test_graph tests/test_graph.cpp)
    target_link_libraries(test_graph PRIVATE runtime_core)
    add_test(NAME test_graph COMMAND test_graph)

    # test_memory
    add_executable(test_memory tests/test_memory.cpp)
    target_link_libraries(test_memory PRIVATE runtime_core)
    add_test(NAME test_memory COMMAND test_memory)

    # test_fusion
    add_executable(test_fusion tests/test_fusion.cpp)
    target_link_libraries(test_fusion PRIVATE runtime_core)
    add_test(NAME test_fusion COMMAND test_fusion)

    # test_backward
    add_executable(test_backward tests/test_backward.cpp)
    target_link_libraries(test_backward PRIVATE runtime_core)
    add_test(NAME test_backward COMMAND test_backward)
endif()

# ============================================================================
# Install
# ============================================================================
install(TARGETS runtime_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(FILES ${YGGDRASIL_HEADERS} DESTINATION include/yggdrasil)

# Python bindings (informational â€” not built by CMake)
message(STATUS "Python bindings are in python/ â€” use directly with numpy")
