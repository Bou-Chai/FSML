cmake_minimum_required(VERSION 3.28)
project(FSML CXX)

add_subdirectory(./external/Tables/)
#add_subdirectory(./src/evaluation/)
add_subdirectory(./src/models/)
add_subdirectory(./src/training/)
add_subdirectory(./tests/)
