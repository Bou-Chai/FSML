add_library(libPlanarModels ./PlanarModel.cpp ./MLRModel.cpp)
target_include_directories(libPlanarModels PUBLIC ../../../include/ PRIVATE ./)
target_link_libraries(libPlanarModels libTables)
