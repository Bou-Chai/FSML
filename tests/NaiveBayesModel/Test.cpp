#include <string>
#include "FSML/models/NaiveBayesModel/NaiveBayesModel.h"
#include "FSML/evaluation/Evaluation.h"
#include "Tables/Table.h"
#include <cmath>
#include <unordered_map>

int main() {  
    tables::Table dataset;
    // Load data and convert relevant columns to double
    dataset.loadCSV("../tests/data/iris.csv", ',');
    std::vector<std::string> titleList = {"sepal_length", "sepal_width", "petal_length", "petal_width"};
    dataset.toDouble(titleList);
    dataset.reshuffle();

    // Calculate size of training data and split data for training and testing
    int trainSize = 0.6 * dataset.height();
    tables::Table trainingFeatures = dataset.copy(0, 4, 0, trainSize);
    tables::Table trainingTargets = dataset.copy(4, 5, 0, trainSize);
    tables::Table testFeatures = dataset.copy(0, 4, trainSize, dataset.height());
    tables::Table testTargets = dataset.copy(4, 5, trainSize, dataset.height());

    NaiveBayesModel model;
    model.train(trainingFeatures, trainingTargets);

    // Performance metrics

    // Generate column of zero-rule values
    std::string majorityClass = trainingTargets.col<std::string>("class").findMajority();
    tables::Column<std::string> targetPredicted0;
    for (int i = 0; i < testTargets.height(); i++) {
        targetPredicted0.add(majorityClass);
    }

    // Generate column of values predicted by the model based on the test feature data
    tables::Column<std::string> targetPredictedM;
    for (int i = 0; i < testTargets.height(); i++) {
        targetPredictedM.add(model.classify(testFeatures.getRow<double>(i)));
    }

    // Print the mean and standard deviation for each feature of each class
    std::cout << "Gaussian probability density function parameters:\n";
    model.printGpdfParams();
    std::cout << "\n";

    // Print performance metrics
    std::cout << "0-R Majority Class: " << majorityClass << "\n";
    std::cout << "Accuracy of 0-R: " << tables::eval::classificationAccuracy(testTargets.col<std::string>(0), targetPredicted0) << "\n";
    std::cout << "Accuracy of model: " << tables::eval::classificationAccuracy(testTargets.col<std::string>(0), targetPredictedM) << "\n";

    return 0;
}
