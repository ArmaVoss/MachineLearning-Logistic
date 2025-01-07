//
// Created by Arman Vossoughi on 1/7/25.
//
#ifndef PARAMETER_H
#define PARAMETER_H
#include "Matrix.h"
template<typename T>
class Parameter{
private:
    Matrix<T> val;
    Matrix<T> grad;
public:
    Parameter(const Matrix<T>& val_, const Matrix<T>& grad_)  : val(val_), grad(grad_) {}

    Parameter& reset() {
        for (int i = 0; i < grad.shape().first; i++) {
            for (int j = 0; j < grad.shape().second; j++) {
                grad[i][j] = 0;
            }
        }
        return *this;
    }

    Parameter& step(const Matrix<double>& gradient_update) {
        if (grad.shape() != val.shape() || gradient_update.shape() != val.shape()) {
            throw std::invalid_argument("size of new gradient does not size of val");
        }
        for (int i = 0; i < val.shape().first; i++) {
            for (int j = 0; j < val.shape().second; j++) {
                val[i][j] -= gradient_update[i][j];
            }
        }
        return reset();
    }

    void setGradient(const Matrix<double>& newGrad) {
        if (newGrad.shape() != grad.shape()) {
            throw std::invalid_argument("New gradient size must match the current gradient size.");
        }
        grad = newGrad;
    }

    Matrix<T>& getValue() { return val; }
    const Matrix<T>& getValue() const { return val; }
    const Matrix<T>& getGradient() const { return grad; }
};

#endif //PARAMETER_H
