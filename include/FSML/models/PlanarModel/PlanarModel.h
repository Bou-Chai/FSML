#ifndef PLANARMODEL_H
#define PLANARMODEL_H

#include <vector>
#include <Tables/Table.h>

class PlanarModel {
public:
    virtual ~PlanarModel() = default;
    // Function to estimate target according to features
    virtual double estimate(const std::vector<double>& featureVals) = 0;
    // Function to update weights according to the model's gradient descent update rule
    virtual void updateWeightsGD(tables::Table& trainingFeatures, tables::Table& trainingTargets, int rowIndex, double learningRate) = 0;
    // Function to train the model given a table of data, the range of the features column, the index of the target column, and the number of epochs
    //virtual void train(GDTrainer& trainer, tables::Table& trainingFeatures, tables::Table& trainingTargets, int epochs);
    virtual double getConstant();
    virtual void setConstant(double constant);
    virtual std::vector<double> getCoeffs();
    virtual void setCoeffs(std::vector<double>);

protected:
    double constant;
    std::vector<double> coeffs;
};

#endif
