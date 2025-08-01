#include <string>
#include "FSML/models/PlanarModel/LogRModel.h"
#include "FSML/training/gradientdescent/SGDTrainer.h"
#include "FSML/evaluation/Evaluation.h"
#include "Tables/Table.h"
#include <cmath>

int main() {  
    tables::Table dataset;
    // Load data and convert relevant columns to double
    dataset.loadCSV("../tests/data/pima-indians-diabetes.csv", ',');
    std::vector<std::string> titleList = {"s","t","u","v","w","x","y","z","diabetes"};
    dataset.toDouble(titleList);

    // Normalize features
    titleList.pop_back();
    dataset.normalize<double>(titleList);

    // Calculate size of training data and split data for training and testing
    int trainSize = 0.6 * dataset.height();
    tables::Table trainingFeatures = dataset.copy(0, 8, 0, trainSize);
    tables::Table trainingTargets = dataset.copy(8, 9, 0, trainSize);
    tables::Table testFeatures = dataset.copy(0, 8, trainSize, dataset.height());
    tables::Table testTargets = dataset.copy(8, 9, trainSize, dataset.height());

    // Train model using gradient descent
    LogRModel model;

    SGDTrainer trainer;
    trainer.setLearningRate(0.3);
    trainer.train(model, trainingFeatures, trainingTargets, 100);

    // Print model weights
    std::cout << "Constant: " << model.getConstant() << "\n";
    std::vector<double> coeffs = model.getCoeffs();
    for (int i = 0; i < coeffs.size(); i++) {
        std::cout << "Coefficient " << i << ": " << coeffs[i] << "\n";
    }

    // Performance metrics

    // Generate column of zero-rule values
    double majorityClass = trainingTargets.col<double>("diabetes").findMajority();
    tables::Column<double> targetPredicted0;
    for (int i = 0; i < testTargets.height(); i++) {
        targetPredicted0.add(majorityClass);
    }

    // Generate column of values predicted by the model based on the test feature data
    tables::Column<double> targetPredictedM;
    for (int i = 0; i < testTargets.height(); i++) {
        targetPredictedM.add(std::round(model.estimate(testFeatures.getRow<double>(i))));
    }

    // Print performance metrics
    std::cout << "0-R Majority Class: " << majorityClass << "\n";
    std::cout << "Accuracy of 0-R: " << tables::eval::classificationAccuracy(testTargets.col<double>(0), targetPredicted0) << "\n";
    std::cout << "Accuracy of model: " << tables::eval::classificationAccuracy(testTargets.col<double>(0), targetPredictedM) << "\n";

    return 0;
}
