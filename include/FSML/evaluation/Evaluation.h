#ifndef EVALUATION_H
#define EVALUATION_H

#include <cmath>
#include "Tables/Column.h"

namespace tables {
    namespace eval {
        void checkColumns(const ColumnBase& col1, const ColumnBase& col2, std::string funcName) {
            if (col1.empty() || col2.empty()) {
                throw std::logic_error("eval::" + funcName + ": Input columns must not be empty");
            }
            if (col1.size() != col2.size()) {
                throw std::logic_error("eval::" + funcName + ": Input column sizes must be the same");
            }
        }

        template <typename T>
        double mae(Column<T>& yActual, Column<T>& yPredicted) {
            checkColumns(yActual, yPredicted, "mae");

            T errorSum = 0;
            for (int i = 0; i < yActual.size(); i++) {
                errorSum += std::abs(yActual.row(i) - yPredicted.row(i));
            }
            return errorSum / yActual.size();
        }

        template <typename T>
        double rmse(Column<T>& yActual, Column<T>& yPredicted) {
            checkColumns(yActual, yPredicted, "rmse");

            T errorSum = 0;
            for (int i = 0; i < yActual.size(); i++) {
                errorSum += std::pow(yActual.row(i) - yPredicted.row(i), 2);
            }
            return std::sqrt(errorSum / yActual.size());
        }
    }
}

#endif
