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

add_executable(ball_trajectory ball_trajectory.cpp)
target_link_libraries(ball_trajectory PRIVATE ${PYTHON_LIBRARIES})
