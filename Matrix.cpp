//
// Created by Arman Vossoughi on 12/26/24.
//

#include "Matrix.h"
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

template class Matrix<int>;
template class Matrix<float>;
template class Matrix<double>;