#include "layer.h"
#include <iostream>
#define NF 3
#define M 20

double cost(const Matrix &A, const Matrix &Y) {
    double res = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < NF; ++j) {
            res += pow(Y.at(j, i) - A.at(j, i), 2.);
        }
    }
    res /= M;

    return res;
}

int main() {
    Matrix X(NF, M);
    Matrix Y(1, M);
    for (int i = 0; i < M; ++i) {
        X.atref(0, i) = i;
        X.atref(1, i) = i*i;
        X.atref(2, i) = i*i*i;
        Y.atref(0, i) = i + i*i + i*i*i;
    }

    Layer input(NF, 1, M, false);
    input.A = X;
    Layer hidden0(25, NF, M, false);
    Layer output(1, 25, M, false);

    double a = 0.01;
    for (int i = 0; i < 10; ++i) {
        forward_prop(hidden0, input);
        forward_prop(output, hidden0);

        Matrix dW_out, db_out;
        back_prop(output, hidden0, nullptr, Y, dW_out, db_out);
        Matrix dW_hid, db_hid;
        back_prop(hidden0, input, &output, Y, dW_hid, db_hid);

        output.W = output.W - dW_out * a;
        output.b = output.b - db_out * a;
        hidden0.W = hidden0.W - dW_hid * a;
        hidden0.b = hidden0.b - db_hid * a;

        /* if ((i + 1) % 100 == 0) { */
            cout << "epoch " << i + 1 << " | cost: " << cost(output.A, Y) << '\n';
        /* } */
    }

    for (int r = 0; r < output.A.rows(); ++r) {
        for (int c = 0; c < output.A.cols(); ++c) {
            cout << output.A.at(r, c) << ' ';
        }
        cout << '\n';
    }
}
