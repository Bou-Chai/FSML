#include <string>
#include "FSML/models/PlanarModel/MLRModel.h"
#include "FSML/training/gradientdescent/SGDTrainer.h"
#include "FSML/evaluation/Evaluation.h"
#include "Tables/Table.h"

int main() {  
    tables::Table dataset;
    // Load data and convert relevant columns to double
    dataset.loadCSV("../tests/data/winequality-red.csv", ';');
    std::vector<std::string> titleList = {"fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"};
    dataset.toDouble(titleList);

    // Normalize features
    titleList.pop_back();
    dataset.normalize<double>(titleList);

    // Calculate size of training data and split data for training and testing
    int trainSize = 0.6 * dataset.height();
    tables::Table trainingFeatures = dataset.copy(0, 11, 0, trainSize);
    tables::Table trainingTargets = dataset.copy(11, 12, 0, trainSize);
    tables::Table testFeatures = dataset.copy(0, 11, trainSize, dataset.height());
    tables::Table testTargets = dataset.copy(11, 12, trainSize, dataset.height());

    std::cout << "Training Features:" << trainingFeatures.height() << "\n";
    std::cout << "Test Features:" << testFeatures.height() << "\n";
    std::cout << "Training Targets:" << trainingTargets.height() << "\n";
    std::cout << "Test Targets:" << testTargets.height() << "\n";

    // Train model using gradient descent
    MLRModel model;
    model.setLearningRate(0.04);
    //model.train(trainingFeatures, trainingTargets, 5);
    SGDTrainer trainer;
    trainer.train(model, trainingFeatures, trainingTargets, 5);
    //model.train(tr, trainingFeatures, trainingTargets, 5);

    // Print model weights
    std::cout << "Constant: " << model.getConstant() << "\n";
    std::vector<double> coeffs = model.getCoeffs();
    for (int i = 0; i < coeffs.size(); i++) {
        std::cout << "Coefficient " << i << ": " << coeffs[i] << "\n";
    }

    // Performance metrics

    // Generate column of zero-rule values
    double mean = trainingTargets.col<double>("quality").getMean(0, trainingTargets.height());
    tables::Column<double> targetPredicted0;
    for (int i = 0; i < testTargets.height(); i++) {
        targetPredicted0.add(mean);
    }

    // Generate column of values predicted by the model based on the test feature data
    tables::Column<double> targetPredictedM;
    for (int i = 0; i < testTargets.height(); i++) {
        targetPredictedM.add(model.estimate(testFeatures.getRow<double>(i)));
    }

    // Print performance metrics
    std::cout << "0-R Mean: " << mean << "\n";
    std::cout << "MAE of 0-R: " << tables::eval::mae(testTargets.col<double>(0), targetPredicted0) << "\n";
    std::cout << "RMSE of 0-R: " << tables::eval::rmse(testTargets.col<double>(0), targetPredicted0) << "\n";
    std::cout << "MAE of model: " << tables::eval::mae(testTargets.col<double>(0), targetPredictedM) << "\n";
    std::cout << "RMSE of model: " << tables::eval::rmse(testTargets.col<double>(0), targetPredictedM) << "\n";

    return 0;
}
