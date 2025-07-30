#ifndef MLRMODEL_H
#define MLRMODEL_H

#include <vector>
#include <Tables/Table.h>
#include "FSML/models/PlanarModel/PlanarModel.h"
//#include "FSML/training/gradientdescent/GDTrainer.h"

class MLRModel : public PlanarModel {
public:
    // Function to estimate target according to features
    //double estimate(const std::vector<double>& featureVals);
    // Function to train the model given a table of data, the range of the features column, the index of the target column, and the number of epochs
    //void train(tables::Table& trainingFeatures, tables::Table& trainingTargets, int epochs);
    //void train(GDTrainer& trainer, tables::Table& trainingFeatures, tables::Table& trainingTargets, int epochs);
    void updateWeightsGD(tables::Table& trainingFeatures, tables::Table& trainingTargets, int rowIndex);
    //double getConstant();
    //void setConstant(double constant);
    //std::vector<double> getCoeffs();
    //void setCoeffs(std::vector<double>);
    //double getLearningRate();
    //void setLearningRate(double learningRate);
};

#endif
