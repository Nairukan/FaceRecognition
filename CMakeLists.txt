cmake_minimum_required(VERSION 3.5)

project(LBPH LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV COMPONENTS REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(LBPH main.cpp myfacerecognition.cpp myfacerecognition.h)

target_link_libraries(LBPH ${OpenCV_LIBS})

install(TARGETS LBPH
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
