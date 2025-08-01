#include <cmath>
#include "FSML/models/PlanarModel/LogRModel.h"

double LogRModel::estimate(const std::vector<double>& featureVals) {
    double estimate = constant;
    for (int i = 0; i < coeffs.size(); i++) {
        estimate += coeffs[i] * featureVals[i];
    }

    estimate *= -1.0;
    estimate = 1.0 / (1.0 + exp(estimate));

    return estimate;
}

void LogRModel::updateWeightsGD(tables::Table& trainingFeatures, tables::Table& trainingTargets, int rowIndex, double learningRate) {
    // Calculate error
    double estimate = this->estimate(trainingFeatures.getRow<double>(rowIndex));
    double error = trainingTargets.at<double>(0, rowIndex) - estimate;
    // Update constant and coefficients
    constant = constant + learningRate * error * estimate * (1 - estimate);
    for (int j = 0; j < coeffs.size(); j++) {
        coeffs.at(j) = coeffs.at(j) + learningRate * error * estimate * (1 - estimate) * trainingFeatures.at<double>(j, rowIndex);
    }
}
