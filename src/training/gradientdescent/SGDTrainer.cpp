#include "FSML/training/gradientdescent/SGDTrainer.h"

void SGDTrainer::train(PlanarModel& model, tables::Table& trainingFeatures, tables::Table& trainingTargets, int epochs) {
    // Initialize constant and coefficients
    int constant = 0;

    std::vector<double> coeffs;
    for (int i = 0; i < trainingFeatures.width(); i++) {
        coeffs.push_back(0);
    }

    model.setConstant(constant);
    model.setCoeffs(coeffs);

    // Train for n epochs
    for (int n = 0; n < epochs; n++) {
        // *Stochastic*
        std::random_device rd;
        unsigned int seed = rd();
        trainingFeatures.reshuffle(seed);
        trainingTargets.reshuffle(seed);

        // Go through all training data
        for (int i = 0; i < trainingFeatures.height(); i++) {
            model.updateWeightsGD(trainingFeatures, trainingTargets, i, learningRate);
        }
    }
}
