add_library(libNaiveBayesModel ./NaiveBayesModel.cpp)
target_include_directories(libNaiveBayesModel PUBLIC ../../../include/ PRIVATE ./)
target_link_libraries(libNaiveBayesModel libTables)
