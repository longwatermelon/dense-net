#pragma once
#include "matrix.h"
#include <map>
#include <random>
#include <sstream>
using namespace std;

enum class Loss {
    BinaryCrossentropy,
    Mse,
};

struct Layer {
    int n;
    // b is a 1-col matrix
    Matrix W, b;
    Matrix Z, A;
    Matrix dZ;
    bool relu;

    Layer(int n, int prev_n, bool relu)
        : n(n), W(Matrix(n, prev_n)), b(Matrix(n, 1)),
          relu(relu)
    {
        // random init W [-0.5,0.5]
        for (int r=0; r<W.rows(); ++r) {
            for (int c=0; c<W.cols(); ++c) {
                W.atref(r,c)=(double)(rand()%100)/100.-.5;
            }
        }
    }

    Layer(const string &s) {
        stringstream ss(s);
        int prev_n;
        ss >> n >> prev_n;
        W=Matrix(n, prev_n);
        b=Matrix(n, 1);
        for (int i=0; i<n; ++i) {
            for (int j=0; j<prev_n; ++j) {
                ss >> W.atref(i,j);
            }
        }

        for (int i=0; i<n; ++i) {
            ss >> b.atref(i,0);
        }

        ss >> relu;
    }

    string save() const {
        stringstream ss;
        // n, prev_n
        ss << n << ' ' << W.cols() << ' ';
        for (int i=0; i<W.rows(); ++i) {
            for (int j=0; j<W.cols(); ++j) {
                ss << W.at(i,j) << ' ';
            }
        }

        for (int i=0; i<b.rows(); ++i) {
            ss << b.at(i,0) << ' ';
        }

        ss << relu;
        return ss.str();
    }
};

inline void forward_prop(Layer &l, Layer &prev)
{
    l.Z=l.W*prev.A;
    for (int i=0; i<l.Z.cols(); ++i) {
        vector<double> &col=l.Z.get_col(i);
        for (int j=0; j<col.size(); ++j)
            col[j]+=l.b.at(j,0);
    }

    l.A=l.Z;
    if (l.relu) {
        for (int r=0; r<l.A.rows(); ++r) {
            for (int c=0; c<l.A.cols(); ++c) {
                l.A.atref(r,c)=max(0.,l.Z.at(r,c));
            }
        }
    }
}

inline void back_prop(Layer &l, const Layer &prev, const Layer *next,
                      const Matrix &Y, Matrix &dW, Matrix &db, Loss lossfn=Loss::BinaryCrossentropy)
{
    if (!next) {
        // output layer
        switch (lossfn) {
            case Loss::BinaryCrossentropy: l.dZ=l.A-Y; break;
            case Loss::Mse: {
                l.dZ=(l.A-Y)*2.;
                if (l.relu) {
                    for (int r=0; r<l.dZ.rows(); ++r) {
                        for (int c=0; c<l.dZ.cols(); ++c) {
                            l.dZ.atref(r,c)=max(l.dZ.at(r,c),0.);
                        }
                    }
                }
            } break;
        }
    } else {
        // hidden layer
        Matrix gprime(l.Z.rows(), l.Z.cols());
        if (l.relu) {
            for (int r=0; r<gprime.rows(); ++r) {
                for (int c=0; c<gprime.cols(); ++c) {
                    gprime.atref(r,c)=l.Z.at(r,c)>0. ? 1. : 0.;
                }
            }
        } else {
            for (int r=0; r<gprime.rows(); ++r) {
                for (int c=0; c<gprime.cols(); ++c) {
                    gprime.atref(r,c)=1.;
                }
            }
        }

        l.dZ=(next->W.transpose()*next->dZ).element_mul(gprime);
    }

    dW=l.dZ*prev.A.transpose()*(1./Y.cols());
    db=Matrix(l.dZ.rows(), 1);
    for (int r=0; r<l.dZ.rows(); ++r) {
        db.atref(r,0)=0;
        for (int c=0; c<l.dZ.cols(); ++c) {
            db.atref(r,0)+=l.dZ.at(r,c);
        }
        db.atref(r,0)=db.at(r,0)*(1./Y.cols());
    }
}
