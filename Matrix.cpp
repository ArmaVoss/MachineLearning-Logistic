//
// Created by Arman Vossoughi on 12/26/24.
//

#include "Matrix.h"
#include "Vec.h"
// Access a row (modifiable)
template<typename T>
std::vector<T>& Matrix<T>::operator[](int row) {
     if (row < 0 || row >= data.size()) {
         throw std::out_of_range("Row index out of range");
     }
     return data[row];
 }

// Access a row (read-only)
template<typename T>
const std::vector<T>& Matrix<T>::operator[](int row) const {
    if (row < 0 || row >= data.size()) {
        throw std::out_of_range("Row index out of range");
    }
    return data[row];
 }

template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& matrix) {
    if (this == &matrix) {
        return *this;
    }
    data = matrix.data;
    return *this;
}

template<typename T>
bool Matrix<T>::operator==(const Matrix<T>& matrix) const{
    return (data == matrix.data);
}

template<typename T>
std::pair<int, int> Matrix<T>::shape() const {
    if (data.empty()) {
        return {0, 0};
    }
    return (std::pair<int, int> (data.size(), data[0].size()));
}

template<typename T>
Vec<double> Matrix<T>::sum(int axis) const {
    if (axis < 0 || axis > 1) {
        throw::std::invalid_argument("Axis must be 0 or 1");
    }
    if (data.empty()) {
        if (axis == 0) {
            return Vec<double> (std::vector<double>());
        }
        return Vec<double> (std::vector<double>());
    }

    if (axis == 0) {
        std::vector<double> result(data[0].size(),0);
        for (int i = 0; i < data[0].size(); i++) {
            double sum_ = 0;
            for (int j = 0; j < data.size(); j++) {
                sum_ += data[j][i];
            }
            result[i] = sum_;
        }
        return Vec(result);
    }
    else {
        std::vector<double> result(data.size(),0);
        for (int i = 0; i < data.size(); i++) {
            double sum_ = 0;
            for (int j = 0; j < data[i].size(); j++) {
                sum_ += data[i][j];
            }
            result[i] = sum_;
        }
        return Vec(result);
    }
}

template class Matrix<int>;
template class Matrix<float>;
template class Matrix<double>;