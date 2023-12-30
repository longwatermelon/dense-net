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

    Layer input(NF, 1, M, false);
    input.A = X;
    Layer hidden0(10, NF, M, false);
    Layer output(NF_OUT, 10, M, false);

    float w = hidden0.W.at(0, 0);
    float b = hidden0.b.at(0, 0);

    double a = 0.0001;
    for (int i = 0; i < 100000; ++i) {
        forward_prop(hidden0, input);
        forward_prop(output, hidden0);

        Matrix dW_out, db_out;
        back_prop(output, hidden0, nullptr, Y, dW_out, db_out, Loss::Mse);
        Matrix dW_hid, db_hid;
        back_prop(hidden0, input, &output, Y, dW_hid, db_hid);

        output.W = output.W - dW_out * a;
        output.b = output.b - db_out * a;
        hidden0.W = hidden0.W - dW_hid * a;
        hidden0.b = hidden0.b - db_hid * a;

        if ((i + 1) % 10000 == 0) {
            printf("epoch %d | cost: %f\r", i + 1, mse_cost(output.A, Y));
            fflush(stdout);
        }
    }
    cout << '\n';

    for (int i = 0; i < M; ++i) {
        cout << output.A.at(0, i) << " VS " << Y.at(0, i) << '\n';
    }

    /* cout << hidden0.W.at(0, 0) << ' ' << hidden0.b.at(0, 0) << '\n'; */
    /* cout << w << ' ' << b << '\n'; */
}
