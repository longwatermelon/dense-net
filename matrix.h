#pragma once
#include <vector>
using namespace std;

class Matrix {
public:
    // column major
    Matrix(int rows, int cols) {
        for (int i=0; i<cols; ++i) {
            m_data.emplace_back(rows,0.);
        }
    }
    Matrix()=default;

    Matrix transpose() const {
        Matrix res(cols(), rows());
        for (int r=0; r<rows(); ++r) {
            for (int c=0; c<cols(); ++c) {
                res.atref(c, r)=at(r,c);
            }
        }

        return res;
    }

    Matrix element_mul(const Matrix &other) const {
        Matrix res=*this;
        for (int r=0; r<rows(); ++r) {
            for (int c=0; c<cols(); ++c) {
                res.atref(r,c)*=other.at(r,c);
            }
        }

        return res;
    }

    Matrix operator+(const Matrix &other) const {
        Matrix res=*this;
        for (int r=0; r<rows(); ++r) {
            for (int c=0; c<cols(); ++c) {
                res.atref(r,c)+=other.at(r,c);
            }
        }

        return res;
    }

    Matrix operator-(const Matrix &other) const {
        Matrix res=*this;
        for (int r=0; r<rows(); ++r) {
            for (int c=0; c<cols(); ++c) {
                res.atref(r,c) -= other.at(r,c);
            }
        }

        return res;
    }

    Matrix operator*(const Matrix &other) const {
        Matrix res=Matrix(rows(), other.cols());

        for (int r=0; r<rows(); ++r) {
            for (int c=0; c<other.cols(); ++c) {
                res.atref(r,c)=0;
                for (int i=0; i<cols(); ++i) {
                    res.atref(r,c)+=at(r, i)*other.at(i, c);
                }
            }
        }

        return res;
    }

    Matrix operator*(float s) const {
        Matrix res=*this;

        for (int r=0; r<rows(); ++r) {
            for (int c=0; c<cols(); ++c) {
                res.atref(r,c)*=s;
            }
        }

        return res;
    }

    void set(int r, int c, double value) { m_data[c][r]=value; }
    double &atref(int r, int c) { return m_data[c][r]; }
    double at(int r, int c) const { return m_data[c][r]; }
    vector<double> &get_col(int c) { return m_data[c]; }
    int rows() const { return m_data[0].size(); }
    int cols() const { return m_data.size(); }

private:
    vector<vector<double>> m_data;
};
