cmake_minimum_required(VERSION 3.28)
project(NetProject LANGUAGES CXX)
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

message("dir: "${CMAKE_CURRENT_SOURCE_DIR})

add_executable(app ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)


add_library(netWorkUtils ${SRC_DIR}/tcpUdp.cpp ${SRC_DIR}/networkBase.cpp)

target_link_libraries(app netWorkUtils)