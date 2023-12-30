#pragma once
#include "matrix.h"
#include <cmath>

inline void feature_scale(Matrix &X) {
    // features iter
    for (int i = 0; i < X.rows(); ++i) {
        float mean = 0;
        for (int j = 0; j < X.cols(); ++j)
            mean += X.at(i, j);
        mean /= X.cols();

        float sd = 0;
        for (int j = 0; j < X.cols(); ++j)
            sd += pow(X.at(i, j) - mean, 2);
        sd /= X.cols();

        for (int j = 0; j < X.cols(); ++j)
            X.atref(i, j) = (X.at(i, j) - mean) / sd;
    }
}
