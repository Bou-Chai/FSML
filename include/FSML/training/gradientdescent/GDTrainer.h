#ifndef GDTRAINER_H
#define GDTRAINER_H

#include "FSML/models/PlanarModel/PlanarModel.h"
#include "Tables/Table.h"

class GDTrainer {
public:
    virtual ~GDTrainer() = default;
    virtual void train(PlanarModel& model, tables::Table& trainingFeatures, tables::Table& trainingTargets, int epochs) = 0;
    double getLearningRate();
    void setLearningRate(double learningRate);

protected:
    double learningRate = 0.1;
};

#endif
