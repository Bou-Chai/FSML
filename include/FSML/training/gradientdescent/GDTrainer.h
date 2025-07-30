#ifndef GDTRAINER_H
#define GDTRAINER_H

#include "FSML/models/PlanarModel/PlanarModel.h"
#include "Tables/Table.h"

class GDTrainer {
public:
    virtual ~GDTrainer() = default;
    virtual void train(PlanarModel& model, tables::Table& trainingFeatures, tables::Table& trainingTargets, int epochs) = 0;
};

#endif
