#ifndef PLANARMODEL_H
#define PLANARMODEL_H

#include <vector>
#include <Tables/Table.h>

class PlanarModel {
public:
    // Function to estimate target according to features
    double estimate(const std::vector<double>& featureVals);
    // Function to train the model given a table of data, the range of the features column, the index of the target column, and the number of epochs
    void train(tables::Table& trainingData, int featuresStart, int featuresEnd, std::string targetColIndex, int epochs);
    double getConstant();
    void setConstant(double constant);
    std::vector<double> getCoeffs();
    void setCoeffs(std::vector<double>);
    double getLearningRate();
    void setLearningRate(double learningRate);

private:
    double learningRate = 0.1;
    double constant;
    std::vector<double> coeffs;
    void updateWeights();
};

#endif
