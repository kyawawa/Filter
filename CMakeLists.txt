cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
endif()

if(USE_CLANG)
  find_program(CLANG clang)
  find_program(CLANG++ clang++)
  if(CLANG AND CLANG++)
    set(CMAKE_C_COMPILER "${CLANG}")
    set(CMAKE_CXX_COMPILER "${CLANG++}")
  else(USE_CLANG)
    message("Cannot find clang or clang++. Please install if you want to use Clang.")
  endif()
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  add_compile_options(-fdiagnostics-color=always)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  add_compile_options(-fcolor-diagnostics)
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(EIGEN REQUIRED eigen3)

include_directories(${EIGEN_INCLUDE_DIRS})

add_subdirectory(examples)
