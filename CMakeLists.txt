cmake_minimum_required(VERSION 3.20)
project(MASS)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================================
# BUILD TYPE CONFIGURATION
# =============================================================================
# Default to Release build for maximum performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# =============================================================================
# COMPILER OPTIMIZATION FLAGS
# =============================================================================
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

# Release optimizations - these are safe and don't change simulation logic
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Additional optimization flags for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Enable native CPU optimizations (uses best instruction set for your CPU)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
        message(STATUS "Enabled -march=native for CPU-specific optimizations")
    endif()
    
    # Enable fast math (safe for physics simulations, significant speedup)
    # Note: This is slightly less precise but much faster
    option(ENABLE_FAST_MATH "Enable fast math optimizations" OFF)
    if(ENABLE_FAST_MATH)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math")
        message(STATUS "Enabled -ffast-math (less precise but faster)")
    endif()
    
    # Link-time optimization for better inlining across translation units
    option(ENABLE_LTO "Enable Link Time Optimization" OFF)
    if(ENABLE_LTO)
        check_cxx_compiler_flag("-flto" COMPILER_SUPPORTS_LTO)
        if(COMPILER_SUPPORTS_LTO)
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto")
            set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -flto")
            message(STATUS "Enabled Link Time Optimization")
        endif()
    endif()
endif()

message(STATUS "CXX Release flags: ${CMAKE_CXX_FLAGS_RELEASE}")

add_compile_definitions(MASS_ROOT_DIR="${CMAKE_SOURCE_DIR}")

# =============================================================================
# EIGEN OPTIMIZATIONS
# =============================================================================
# These flags help Eigen use SIMD instructions effectively
add_compile_definitions(EIGEN_NO_DEBUG)  # Disable Eigen debug assertions in release
add_compile_definitions(EIGEN_FAST_MATH)  # Enable Eigen fast math

# =============================================================================
# FIND PACKAGES
# =============================================================================
find_package(DART REQUIRED COMPONENTS dart gui collision-bullet utils)
find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(TinyXML REQUIRED IMPORTED_TARGET tinyxml)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Bullet REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${DART_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${TinyXML_INCLUDE_DIRS})
include_directories(${BULLET_INCLUDE_DIRS})

# =============================================================================
# CORE LIBRARY
# =============================================================================
add_library(MASS SHARED
    core/BVH.cpp
    core/Character.cpp
    core/DARTHelper.cpp
    core/Environment.cpp
    core/Muscle.cpp
)

target_include_directories(MASS PUBLIC 
    ${CMAKE_SOURCE_DIR}/core
    ${DART_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${TinyXML_INCLUDE_DIRS}
    ${BULLET_INCLUDE_DIRS}
)

target_link_libraries(MASS PUBLIC 
    ${DART_LIBRARIES}
    Eigen3::Eigen 
    PkgConfig::TinyXML
    ${BULLET_LIBRARIES}
)

# =============================================================================
# RENDER EXECUTABLE
# =============================================================================
add_executable(render 
    render/main.cpp 
    render/Window.cpp
)

target_link_libraries(render 
    MASS 
    OpenGL::GL 
    GLUT::GLUT
    pybind11::embed
)

# =============================================================================
# PYTHON MODULE
# =============================================================================
pybind11_add_module(pymss python/EnvManager.cpp)

target_include_directories(pymss PRIVATE 
    ${CMAKE_SOURCE_DIR}/python
)

# Link OpenMP for parallel environment simulation
target_link_libraries(pymss PRIVATE MASS OpenMP::OpenMP_CXX)

# =============================================================================
# PRINT CONFIGURATION SUMMARY
# =============================================================================
message(STATUS "")
message(STATUS "=== MASS Build Configuration ===")
message(STATUS "Build type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler:    ${CMAKE_CXX_COMPILER}")
message(STATUS "OpenMP enabled:  ${OpenMP_CXX_FOUND}")
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP version:  ${OpenMP_CXX_VERSION}")
endif()
message(STATUS "================================")