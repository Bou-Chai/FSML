#ifndef LOGRMODEL_H
#define LOGRMODEL_H

#include <vector>
#include <Tables/Table.h>
#include "FSML/models/PlanarModel/PlanarModel.h"

class LogRModel : public PlanarModel {
public:
    double estimate(const std::vector<double>& featureVals) override;
    void updateWeightsGD(tables::Table& trainingFeatures, tables::Table& trainingTargets, int rowIndex, double learningRate) override;
};

#endif
