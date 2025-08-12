#include "FSML/models/NaiveBayesModel/NaiveBayesModel.h"
#include "Tables/Table.h"

double NaiveBayesModel::estimate(const std::vector<double>& featureVals) {

}

void NaiveBayesModel::train(tables::Table& trainingFeatures, tables::Table& trainingTargets) {
    // Count instances of classes
    for (int i = 0; i < trainingTargets.height(); i++) {
        classInstanceCounts[trainingTargets.at<double>(0, i)]++;
    }

    // Clear and initialize map
    gpdfParams.clear();
    for (int i = 0; i < classInstanceCounts.size(); i++) {
        std::unordered_map<int, int>::iterator it = classInstanceCounts.begin();
        for (int j = 0; j < trainingFeatures.width(); j++, ++it) {
            gpdfParams[it->first].push_back({0, 0});
        }
    }

    // Calculate feature means for each class
    for (int i = 0; i < trainingFeatures.height(); i++) {
        for (int j = 0; j < trainingFeatures.width(); j++) {
            gpdfParams[trainingTargets.at<int>(0, i)][j][0] += trainingFeatures.at<double>(j, i);
        }
    }

    // Calculate standard deviation for each class
    // Calculate feature means for each class
    for (int i = 0; i < trainingFeatures.height(); i++) {
        for (int j = 0; j < trainingFeatures.width(); j++) {
            gpdfParams[trainingTargets.at<int>(0, i)][j][1] += trainingFeatures.at<double>(j, i);
        }
    }
}

double NaiveBayesModel::gpdf(double featureValue, double mean, double sd) {

}
