add_library(libMLRModel ./MLRModel.cpp)
target_include_directories(libMLRModel PUBLIC ../../../include/ PRIVATE ./)
target_link_libraries(libMLRModel libTables)
