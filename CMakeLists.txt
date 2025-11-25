cmake_minimum_required(VERSION 3.20)
project(MASS)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

add_compile_definitions(MASS_ROOT_DIR="${CMAKE_SOURCE_DIR}")

find_package(DART REQUIRED COMPONENTS dart gui collision-bullet utils)
find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(TinyXML REQUIRED IMPORTED_TARGET tinyxml)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Bullet REQUIRED)

include_directories(${DART_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${TinyXML_INCLUDE_DIRS})
include_directories(${BULLET_INCLUDE_DIRS})

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

find_package(OpenMP REQUIRED)

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

pybind11_add_module(pymss python/EnvManager.cpp)

target_include_directories(pymss PRIVATE 
    ${CMAKE_SOURCE_DIR}/python
)

target_link_libraries(pymss PRIVATE MASS OpenMP::OpenMP_CXX)
