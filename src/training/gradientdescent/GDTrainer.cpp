#include "FSML/training/gradientdescent/GDTrainer.h"

    double GDTrainer::getLearningRate() {
        return learningRate;
    }

    void GDTrainer::setLearningRate(double learningRate) {
        this->learningRate = learningRate;
    }

