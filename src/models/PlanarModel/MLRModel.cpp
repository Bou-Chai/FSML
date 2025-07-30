#include <random>
#include "FSML/models/PlanarModel/MLRModel.h"
//#include "FSML/training/gradientdescent/GDTrainer.h"

    void MLRModel::updateWeightsGD(tables::Table& trainingFeatures, tables::Table& trainingTargets, int rowIndex) {
        // Calculate error
        double error = this->estimate(trainingFeatures.getRow<double>(rowIndex)) - trainingTargets.at<double>(0, rowIndex);
        // Update constant and coefficients
        constant = constant - learningRate * error;
        for (int j = 0; j < coeffs.size(); j++) {
            coeffs.at(j) = coeffs.at(j) - learningRate * trainingFeatures.at<double>(j, rowIndex) * error;
        }
    }
/*
    void MLRModel::train(GDTrainer& trainer, tables::Table& trainingFeatures, tables::Table& trainingTargets, int epochs) {
        trainer.train(*this, trainingFeatures, trainingTargets, epochs);
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
            unsigned int seed = rd();
            trainingFeatures.reshuffle(seed);
            trainingTargets.reshuffle(seed);

            // Go through all training data
            double error;
            for (int i = 0; i < trainingFeatures.height(); i++) {
                // Calculate error
                error = estimate(trainingFeatures.getRow<double>(i)) - trainingTargets.at<double>(0, i);
                std::cout <<"\n";
                // Update constant and coefficients
                constant = constant - learningRate * error;
                for (int j = 0; j < coeffs.size(); j++) {
                    coeffs.at(j) = coeffs.at(j) - learningRate * trainingFeatures.at<double>(j, i) * error;
                }
            }
        }
    }
*/
