#include "layer.h"
#include "util.h"
#include <iostream>
#define NF 3
#define NF_OUT 1
#define M 20

double mse_cost(const Matrix &A, const Matrix &Y) {
    double res = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < NF_OUT; ++j) {
            res += pow(Y.at(j, i) - A.at(j, i), 2.);
        }
    }
    res /= M;

    return res;
}

int main() {
    srand(time(0));
    /* Matrix X(NF, M); */
    /* Matrix Y(1, M); */
    /* for (int i = 0; i < M; ++i) { */
    /*     X.atref(0, i) = i; */
    /*     X.atref(1, i) = i*i; */
    /*     X.atref(2, i) = i*i*i; */
    /*     Y.atref(0, i) = i + i*i + i*i*i; */
    /* } */

    Matrix X(NF, M);
    Matrix Y(NF_OUT, M);
    for (int i = 0; i < M; ++i) {
        X.atref(0, i) = (double)i / M;
        X.atref(1, i) = (double)i*i / (M*M);
        X.atref(2, i) = (double)i*i*i / (M*M*M);
        Y.atref(0, i) = (double)(i + i*i + i*i*i);
    }

    /* feature_scale(X); */

    vector<Layer> layers = {
        Layer(NF, 1, false),
        Layer(10, NF, false),
        /* Layer(15, 25, M, false), */
        Layer(NF_OUT, 10, false),
    };
    layers[0].A = X;
    vector<pair<Matrix, Matrix>> grads(layers.size()); // (dW, db)

    double a = 0.0005;
    double cost = 1e10;
    int epoch = 0;
    while (cost > 0.1) {
        ++epoch;
        for (int i = 1; i < layers.size(); ++i)
            forward_prop(layers[i], layers[i - 1]);

        for (int i = layers.size() - 1; i > 0; --i) {
            back_prop(
                layers[i], layers[i - 1], i == layers.size() - 1 ? nullptr : &layers[i + 1],
                Y, grads[i].first, grads[i].second, Loss::BinaryCrossentropy
            );
        }

        for (int i = 1; i < grads.size(); ++i) {
            layers[i].W = layers[i].W - grads[i].first * a;
            layers[i].b = layers[i].b - grads[i].second * a;
        }

        cost = mse_cost(layers.back().A, Y);
        if (isnan(cost)) {
            fprintf(stderr, "cost became nan");
            exit(1);
        }

        if (epoch % 1000 == 0) {
            printf("epoch %d | cost: %f\r", epoch, cost);
            fflush(stdout);
        }
    }
    cout << '\n';

    for (int i = 0; i < M; ++i) {
        cout << layers.back().A.at(0, i) << " VS " << Y.at(0, i) << '\n';
    }

    /* cout << hidden0.W.at(0, 0) << ' ' << hidden0.b.at(0, 0) << '\n'; */
    /* cout << w << ' ' << b << '\n'; */
}
