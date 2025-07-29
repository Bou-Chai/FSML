#ifndef PLANARMODEL_H
#define PLANARMODEL_H

#include <vector>
#include <Tables/Table.h>

class PlanarModel {
public:
    virtual ~PlanarModel() = default;
    // Function to estimate target according to features
    virtual double estimate(const std::vector<double>& featureVals);
    // Function to update weights according to the model's gradient descent update rule
    virtual void updateWeightsGD();
    // Function to train the model given a table of data, the range of the features column, the index of the target column, and the number of epochs
    virtual void train(tables::Table& trainingData, int featuresStart, int featuresEnd, std::string targetColIndex, int epochs);
    virtual double getConstant();
    virtual void setConstant(double constant);
    virtual std::vector<double> getCoeffs();
    virtual void setCoeffs(std::vector<double>);
    virtual double getLearningRate();
    virtual void setLearningRate(double learningRate);

private:
    double learningRate = 0.1;
    double constant;
    std::vector<double> coeffs;
};

#endif
