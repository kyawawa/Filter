# Download matplotlibcpp
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/matplotlibcpp.h")
  message("Downloading matplotlibcpp.h...")
  file(DOWNLOAD
    "https://raw.githubusercontent.com/lava/matplotlib-cpp/master/matplotlibcpp.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/matplotlibcpp.h"
    SHOW_PROGRESS)
endif()

find_package(PythonLibs 2.7 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
message("${PYTHON_INCLUDE_DIRS}")

set(target ball_trajectory)
add_executable(${target} ${target}.cpp)
target_link_libraries(${target} PRIVATE ${PYTHON_LIBRARIES})

set(target test_multivariate_normal_distribution)
add_executable(${target} ${target}.cpp)
target_link_libraries(${target} PRIVATE ${PYTHON_LIBRARIES})
