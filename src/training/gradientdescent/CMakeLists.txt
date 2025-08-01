add_library(libGDTrainers ./GDTrainer.cpp ./SGDTrainer.cpp)
target_include_directories(libGDTrainers PUBLIC ../../../include PRIVATE ./)
target_link_libraries(libGDTrainers libTables)
