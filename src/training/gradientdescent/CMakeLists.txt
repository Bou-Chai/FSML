add_library(libSGDTrainer ./SGDTrainer.cpp)
target_include_directories(libSGDTrainer PUBLIC ../../../include/ PRIVATE ./)
target_link_libraries(libSGDTrainer libTables)
