#include <cmath>
#include "FSML/models/NaiveBayesModel/NaiveBayesModel.h"
#include "Tables/Table.h"

std::string NaiveBayesModel::classify(const std::vector<double>& featureVals) {
    std::string estimatedClass;
    int greatestProbability = 0;

    // Check probabilities for all classes and find the class with the highest probability
    std::unordered_map<std::string, int>::iterator it;
    for (it = classInstanceCounts.begin(); it != classInstanceCounts.end(); ++it) {
        if (estimate(featureVals, it->first) > greatestProbability) {
            estimatedClass = it->first;
        }
    }
    return estimatedClass;
}

double NaiveBayesModel::estimate(const std::vector<double>& featureVals, std::string c) {
    double estimate = 0;
    for (int i = 0; i < featureVals.size(); i++) {
        estimate *= gpdf(featureVals[i], gpdfParams.at(c)[i][0], gpdfParams.at(c)[i][1]);
    }
    estimate += classProbabilities.at(c);
    return estimate;
}

void NaiveBayesModel::train(tables::Table& trainingFeatures, tables::Table& trainingTargets) {
    // Count instances of classes
    for (int i = 0; i < trainingTargets.height(); i++) {
        classInstanceCounts[trainingTargets.at<std::string>(0, i)]++;
    }

    // Calculate class probabilities
    std::unordered_map<std::string, int>::iterator itClassInstanceCounts = classInstanceCounts.begin();
    for (itClassInstanceCounts = classProbabilities.begin(); itClassInstanceCounts != classProbabilities.end(); ++itClassInstanceCounts) {
        classProbabilities[itClassInstanceCounts->first] = itClassInstanceCounts->second / trainingTargets.height();
    }

    // Clear and initialize map
    gpdfParams.clear();
    for (int i = 0; i < classInstanceCounts.size(); i++) {
        std::unordered_map<std::string, int>::iterator it = classInstanceCounts.begin();
        for (int j = 0; j < trainingFeatures.width(); j++, ++it) {
            gpdfParams[it->first].push_back({0, 0});
        }
    }

    // Calculate feature means for each class
    for (int i = 0; i < trainingFeatures.height(); i++) {
        for (int j = 0; j < trainingFeatures.width(); j++) {
            gpdfParams.at(trainingTargets.at<std::string>(0, i))[j][0] += trainingFeatures.at<double>(j, i);
        }
    }

    std::unordered_map<std::string, std::vector<std::vector<double>>>::iterator itGpdfParams;
    for (itGpdfParams = gpdfParams.begin(); itGpdfParams != gpdfParams.end(); ++itGpdfParams) {
        for (int j = 0; j < itGpdfParams->second.size(); j++) {
            itGpdfParams->second.at(j)[0] /= classInstanceCounts.at(itGpdfParams->first);
        }
    }


    // Calculate feature standard deviations for each class
    for (int i = 0; i < trainingFeatures.height(); i++) {
        for (int j = 0; j < trainingFeatures.width(); j++) {
            gpdfParams.at(trainingTargets.at<std::string>(0, i))[j][1] += std::pow(trainingFeatures.at<double>(j, i) - gpdfParams[trainingTargets.at<std::string>(0, i)][j][0], 2);
        }
    }

    for (itGpdfParams = gpdfParams.begin(); itGpdfParams != gpdfParams.end(); ++itGpdfParams) {
        for (int j = 0; j < itGpdfParams->second.size(); j++) {
            itGpdfParams->second.at(j)[1] /= classInstanceCounts.at(itGpdfParams->first);
            itGpdfParams->second.at(j)[1] = std::sqrt(itGpdfParams->second.at(j)[1]);
        }
    }
}

//(1 / (sqrt(2 * PI) * sd)) * exp(-((x-mean^2)/(2*sd^2)))
double NaiveBayesModel::gpdf(double featureValue, double mean, double sd) {
    return ((sqrt(2 * M_PI) * sd)) * std::exp(-((featureValue - std::pow(mean, 2)) / (2 * std::pow(sd, 2))));
}
