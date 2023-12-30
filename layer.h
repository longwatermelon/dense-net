#pragma once
#include "matrix.h"
#include <map>
#include <random>
using namespace std;

struct Layer {
    int n;
    // b is a 1-col matrix
    Matrix W, b;
    Matrix Z, A;
    Matrix dZ;
    bool relu;

    Layer(int n, int prev_n, int m, bool relu)
        : n(n), W(Matrix(n, prev_n)), b(Matrix(n, 1)),
          Z(Matrix(n, m)), A(Matrix(n, m)), dZ(Matrix(n, m)),
          relu(relu)
    {
        // random init W [-0.5,0.5]
        for (int r = 0; r < W.rows(); ++r) {
            for (int c = 0; c < W.cols(); ++c) {
                W.atref(r, c) = (double)(rand() % 100) / 100. - .5;
            }
        }
    }
};

inline void forward_prop(Layer &l, Layer &prev)
{
    l.Z = l.W * prev.A;
    for (int i = 0; i < l.Z.cols(); ++i) {
        vector<double> &col = l.Z.get_col(i);
        for (int j = 0; j < col.size(); ++j)
            col[j] += l.b.at(i, 0);
    }

    l.A = l.Z;
    if (l.relu) {
        for (int r = 0; r < l.A.rows(); ++r) {
            for (int c = 0; c < l.A.cols(); ++c) {
                l.A.atref(r, c) = max(0., l.Z.at(r, c));
            }
        }
    }
}

inline void back_prop(Layer &l, const Layer &prev, const Layer *next,
                      const Matrix &Y, Matrix &dW, Matrix &db)
{
    if (!next) {
        // output layer
        l.dZ = Y * -2. + l.A * 2.;
        /* l.dZ = l.A - Y; */
    } else {
        // hidden layer
        Matrix gprime(l.Z.rows(), l.Z.cols());
        if (l.relu) {
            for (int r = 0; r < gprime.rows(); ++r) {
                for (int c = 0; c < gprime.cols(); ++c) {
                    gprime.atref(r, c) = l.Z.at(r, c) > 0. ? 1. : 0.;
                }
            }
        } else {
            for (int r = 0; r < gprime.rows(); ++r) {
                for (int c = 0; c < gprime.cols(); ++c) {
                    gprime.atref(r, c) = 1.;
                }
            }
        }

        l.dZ = (next->W.transpose() * next->dZ).element_mul(gprime);
    }

    dW = l.dZ * prev.A.transpose() * (1. / Y.cols());
    db = Matrix(l.dZ.rows(), 1);
    for (int r = 0; r < l.dZ.rows(); ++r) {
        db.atref(r, 0) = 0;
        for (int c = 0; c < l.dZ.cols(); ++c) {
            db.atref(r, 0) += l.dZ.at(r, c);
        }
        db.atref(r, 0) = db.at(r, 0) * (1. / Y.cols());
    }
}