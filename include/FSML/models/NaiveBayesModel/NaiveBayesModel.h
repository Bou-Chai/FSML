#ifndef NAIVEBAYESMODEL_H
#define NAIVEBAYESMODEL_H

#include <vector>
#include <unordered_map>
#include "Tables/Table.h"

class NaiveBayesModel {
public:
    double estimate(const std::vector<double>& featureVals);
    void train(tables::Table& trainingFeatures, tables::Table& trainingTargets);

private:
    std::unordered_map<int, std::vector<std::vector<double>>> gpdfParams;
    std::unordered_map<int, int> classInstanceCounts;
    double gpdf(double featureValue, double mean, double sd);
};

#endif
