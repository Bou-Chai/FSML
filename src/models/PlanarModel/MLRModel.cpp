#include <random>
#include "FSML/models/PlanarModel/MLRModel.h"

    double MLRModel::estimate(const std::vector<double>& featureVals) {
        double estimate = constant;
        for (int i = 0; i < coeffs.size(); i++) {
            estimate += coeffs[i] * featureVals[i];
        }
        return estimate;
    }

    void MLRModel::train(tables::Table& trainingFeatures, tables::Table& trainingTargets, int epochs) {
        // Initialize constant and coefficients
        constant = 0;
        for (int i = 0; i < trainingFeatures.width(); i++) {
            coeffs.push_back(0);
        }

        // Train for n epochs
        for (int n = 0; n < epochs; n++) {
            // *Stochastic*
            std::random_device rd;
            int seed = rd();
            trainingFeatures.reshuffle(seed);
            trainingTargets.reshuffle(seed);

            // Go through all training data
            double error;
            for (int i = 0; i < trainingFeatures.height(); i++) {
                // Calculate error
                error = estimate(trainingFeatures.getRow<double>(i)) - trainingTargets.at<double>(0, i);
                // Update constant and coefficients
                constant = constant - learningRate * error;
                for (int j = 0; j < coeffs.size(); j++) {
                    coeffs.at(j) = coeffs.at(j) - learningRate * trainingFeatures.at<double>(j, i) * error;
                }
            }
        }
    }
    
    double MLRModel::getConstant() {
        return constant;
    }
    
    void MLRModel::setConstant(double constant) {
        this->constant = constant;
    }
    
    std::vector<double> MLRModel::getCoeffs() {
        return coeffs;
    }
    
    void MLRModel::setCoeffs(std::vector<double>) {
        this->coeffs = coeffs;
    }
    
    double MLRModel::getLearningRate() {
        return learningRate;
    }

    void MLRModel::setLearningRate(double learningRate) {
        this->learningRate = learningRate;
    }

