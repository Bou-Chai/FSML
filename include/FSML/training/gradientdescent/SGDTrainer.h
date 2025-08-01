#ifndef SGDTRAINER_H
#define SGDTRAINER_H

#include "FSML/training/gradientdescent/GDTrainer.h"
#include "FSML/models/PlanarModel/PlanarModel.h"
#include "Tables/Table.h"

class SGDTrainer: public GDTrainer {
public:
    void train(PlanarModel& model, tables::Table& trainingFeatures, tables::Table& trainingTargets, int epochs) override;
};

#endif
