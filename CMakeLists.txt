cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(USE_CLANG)
  find_program(CLANG clang)
  if(CLANG)
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)
  else(USE_CLANG)
    message("Cannot find Clang. Please install if you want to use Clang.")
  endif()
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(EIGEN REQUIRED eigen3)

include_directories(${EIGEN_INCLUDE_DIRS})

add_subdirectory(examples)
