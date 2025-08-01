#include "FSML/models/PlanarModel/PlanarModel.h"
    double PlanarModel::getConstant() {
        return constant;
    }
    
    void PlanarModel::setConstant(double constant) {
        this->constant = constant;
    }
    
    std::vector<double> PlanarModel::getCoeffs() {
        return coeffs;
    }
    
    void PlanarModel::setCoeffs(std::vector<double> coeffs) {
        this->coeffs = coeffs;
    }
