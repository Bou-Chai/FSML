#include <random>
#include "FSML/models/PlanarModel/MLRModel.h"

    double MLRModel::estimate(const std::vector<double>& featureVals) {
        double estimate = constant;
        for (int i = 0; i < coeffs.size(); i++) {
            estimate += coeffs[i] * featureVals[i];
        }
        return estimate;
    }

    void MLRModel::updateWeightsGD(tables::Table& trainingFeatures, tables::Table& trainingTargets, int rowIndex, double learningRate) {
        // Calculate error
        double error = this->estimate(trainingFeatures.getRow<double>(rowIndex)) - trainingTargets.at<double>(0, rowIndex);
        // Update constant and coefficients
        constant = constant - learningRate * error;
        for (int j = 0; j < coeffs.size(); j++) {
            coeffs.at(j) = coeffs.at(j) - learningRate * trainingFeatures.at<double>(j, rowIndex) * error;
        }
    }
