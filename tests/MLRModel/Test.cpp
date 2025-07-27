#include <string>
#include "FSML/models/MLRModel/MLRModel.h"
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
    tables::Table trainingData = dataset.copy(0, 12, 0, trainSize);
    tables::Table featuresTest = dataset.copy(0, 11, trainSize, dataset.height());
    tables::Table targetTest = dataset.copy(11, 12, trainSize, dataset.height());

    // Train model using gradient descent
    MLRModel model;
    model.setLearningRate(0.04);
    model.train(trainingData, 0, 11, "quality", 5);

    // Print model weights
    std::cout << "Constant: " << model.getConstant() << "\n";
    std::vector<double> coeffs = model.getCoeffs();
    for (int i = 0; i < coeffs.size(); i++) {
        std::cout << "Coefficient " << i << ": " << coeffs[i] << "\n";
    }

    // Performance metrics

    // Generate column of zero-rule values
    double mean = trainingData.col<double>("quality").getMean(0, trainingData.height());
    tables::Column<double> targetPredicted0;
    for (int i = 0; i < targetTest.height(); i++) {
        targetPredicted0.add(mean);
    }

    // Generate column of values predicted by the model based on the test feature data
    tables::Column<double> targetPredictedM;
    for (int i = 0; i < targetTest.height(); i++) {
        targetPredictedM.add(model.estimate(featuresTest.getRow<double>(i)));
    }

    // Print performance metrics
    std::cout << "0-R Mean: " << mean << "\n";
    std::cout << "MAE of 0-R: " << tables::eval::mae(targetTest.col<double>(0), targetPredicted0) << "\n";
    std::cout << "RMSE of 0-R: " << tables::eval::rmse(targetTest.col<double>(0), targetPredicted0) << "\n";
    std::cout << "MAE of model: " << tables::eval::mae(targetTest.col<double>(0), targetPredictedM) << "\n";
    std::cout << "RMSE of model: " << tables::eval::rmse(targetTest.col<double>(0), targetPredictedM) << "\n";

    return 0;
}
