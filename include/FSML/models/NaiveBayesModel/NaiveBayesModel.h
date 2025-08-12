#ifndef NAIVEBAYESMODEL_H
#define NAIVEBAYESMODEL_H

#include <string>
#include <vector>
#include <unordered_map>
#include "Tables/Table.h"

class NaiveBayesModel {
public:
    std::string classify(const std::vector<double>& featureVals);
    double estimate(const std::vector<double>& featureVals, std::string c);
    void train(tables::Table& trainingFeatures, tables::Table& trainingTargets);

private:
    // Maps to hold information on classes
    std::unordered_map<std::string, std::vector<std::vector<double>>> gpdfParams;
    std::unordered_map<std::string, int> classInstanceCounts;
    std::unordered_map<std::string, int> classProbabilities;
    double gpdf(double featureValue, double mean, double sd);
};

#endif
