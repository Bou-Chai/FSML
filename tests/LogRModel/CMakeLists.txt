add_executable(testLogRModel ./Test.cpp)
target_include_directories(testLogRModel PUBLIC ../../include PRIVATE ./)
target_link_libraries(testLogRModel libPlanarModels libGDTrainers)
