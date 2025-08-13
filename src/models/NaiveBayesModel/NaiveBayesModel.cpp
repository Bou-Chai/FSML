#include <cmath>
#include "FSML/models/NaiveBayesModel/NaiveBayesModel.h"
#include "Tables/Table.h"

std::string NaiveBayesModel::classify(const std::vector<double>& featureVals) {
    std::string estimatedClass = "yo";
    double greatestProbability = 0;

    // Check probabilities for all classes and find the class with the highest probability
    std::unordered_map<std::string, int>::iterator it;
    for (it = classInstanceCounts.begin(); it != classInstanceCounts.end(); ++it) {
        double probability = estimate(featureVals, it->first);
        //std::cout << it->first << ": " << probability << "\n";
        if (probability > greatestProbability) {
            greatestProbability = probability;
            estimatedClass = it->first;
        }
    }

    for (double featureVal : featureVals) {
        std::cout << featureVal << ",";
    }
    std::cout << " " << estimatedClass << "\n";
    return estimatedClass;
}

double NaiveBayesModel::estimate(const std::vector<double>& featureVals, std::string c) {
    double estimate = 1;
    for (int i = 0; i < featureVals.size(); i++) {
        estimate *= gpdf(featureVals[i], gpdfParams.at(c)[i][0], gpdfParams.at(c)[i][1]);
        //std::cout << "Mean: " << gpdfParams.at(c)[i][0] << "\n";
        //std::cout << "SD: " << gpdfParams.at(c)[i][1] << "\n";
        //std::cout << "PDF: " << estimate << "\n";
    }
    estimate += classProbabilities.at(c);
    return estimate;
}

void NaiveBayesModel::train(tables::Table& trainingFeatures, tables::Table& trainingTargets) {
    // Count instances of classes
    for (int i = 0; i < trainingTargets.height(); i++) {
        classInstanceCounts[trainingTargets.at<std::string>(0, i)]++;
    }
std::cout << "HERE1\n";
    // Calculate class probabilities
    std::unordered_map<std::string, int>::iterator itClassInstanceCounts = classInstanceCounts.begin();
    for (itClassInstanceCounts = classInstanceCounts.begin(); itClassInstanceCounts != classInstanceCounts.end(); ++itClassInstanceCounts) {
        classProbabilities[itClassInstanceCounts->first] = itClassInstanceCounts->second / trainingTargets.height();
    }
std::cout << "HERE2\n";
    // Clear and initialize map
    gpdfParams.clear();
    for (itClassInstanceCounts = classInstanceCounts.begin(); itClassInstanceCounts != classInstanceCounts.end(); ++itClassInstanceCounts) {
        for (int j = 0; j < trainingFeatures.width(); j++) {
            gpdfParams[itClassInstanceCounts->first].push_back({0, 0});
        }
    }

std::cout << "HERE3\n";
    // Calculate feature means for each class
    for (int i = 0; i < trainingFeatures.height(); i++) {
        for (int j = 0; j < trainingFeatures.width(); j++) {
            gpdfParams.at(trainingTargets.at<std::string>(0, i))[j][0] += trainingFeatures.at<double>(j, i);
        }
    }
std::cout << "HERE4\n";
    std::unordered_map<std::string, std::vector<std::vector<double>>>::iterator itGpdfParams;
    for (itGpdfParams = gpdfParams.begin(); itGpdfParams != gpdfParams.end(); ++itGpdfParams) {
        for (int j = 0; j < itGpdfParams->second.size(); j++) {
            itGpdfParams->second.at(j)[0] /= classInstanceCounts.at(itGpdfParams->first);
        }
    }

std::cout << "HERE5\n";
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
std::cout << "HERE6\n";
}

//(1 / (sqrt(2 * PI) * sd)) * exp(-((x-mean^2)/(2*sd^2)))
double NaiveBayesModel::gpdf(double featureValue, double mean, double sd) {
    double result = (1 / (sqrt(2 * M_PI) * sd)) * std::exp(-(std::pow(featureValue - mean, 2) / (2 * std::pow(sd, 2))));
    std::cout << "PDF: " << result << "\n";
    return result;
}

void NaiveBayesModel::printGpdfParams() {
    std::unordered_map<std::string, std::vector<std::vector<double>>>::iterator itGpdfParams;
    for (itGpdfParams = gpdfParams.begin(); itGpdfParams != gpdfParams.end(); ++itGpdfParams) {
        std::cout << itGpdfParams->first << ": ";
        for (int j = 0; j < itGpdfParams->second.size(); j++) {
            std::cout << "(" << itGpdfParams->second.at(j)[0] << ", " << itGpdfParams->second.at(j)[1] << ") | ";
        }
        std::cout << "\n";
    }
}
