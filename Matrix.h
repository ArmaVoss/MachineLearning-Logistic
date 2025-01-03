//
// Created by Arman Vossoughi on 12/26/24.
//

#ifndef MATRIX_H
#define MATRIX_H
#include <vector>
#include <iostream>
#include "Vec.h"

template <typename T>
class Vec;

template <typename T>
class Matrix {
    public:
        std::vector<std::vector<T>> data;

        //constructor for matrix
        Matrix(const std::vector<std::vector<T>>& matrix) {
            data = matrix;
        }

        //matrix transpose
        Matrix<T> transpose() const {
            int numRows = data.size();
            int numColumns = data[0].size();
            std::vector<std::vector<T>> transposed(numColumns, std::vector<T>(numRows, T()));
            for (int i = 0; i<numRows; i++) {
                for (int j = 0; j<numColumns; j++) {
                    transposed[j][i] = data[i][j];
                }
            }
            return Matrix<T>(transposed);
        }

        friend std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix) {
                os << "[";
                for (int i = 0; i < matrix.data.size(); i++) {
                    if (i > 0) {
                        os << " ";
                    }
                    os << "[";
                    for (int j = 0; j < matrix.data[i].size(); j++) {
                        os << matrix.data[i][j];
                        if (j<matrix.data[i].size() - 1) {
                            os << ", ";
                        }
                    }
                    os << "]";
                    if (i < matrix.data.size() - 1) {
                        os <<", " << std::endl;
                    }
                }
                os << "]";
                return os;
            }

        template <typename U>
        Matrix<typename std::common_type<T, U>::type> operator+(const U& scalar) const {
            using ResultType = typename std::common_type<T, U>::type;
            Matrix<ResultType> result(data);
            for (int i = 0; i < result.data.size(); i++) {
                for (int j = 0; j < result.data[i].size(); j++) {
                    result.data[i][j] = result.data[i][j] + scalar;
                }
            }
            return result;
        }

        template <typename U>
          Matrix<typename std::common_type<T, U>::type> operator-(const U& scalar) const {
                using ResultType = typename std::common_type<T, U>::type;
                Matrix<ResultType> result(data);
                for (int i = 0; i < result.data.size(); i++) {
                    for (int j = 0; j < result.data[i].size(); j++) {
                        result.data[i][j] = result.data[i][j] - scalar;
                    }
                }
                return result;
            }
        template <typename U>
        Matrix<typename std::common_type<T, U>::type> operator*(const U& scalar) const {
                using ResultType = typename std::common_type<T, U>::type;
                Matrix<ResultType> result(data);
                for (int i = 0; i < result.data.size(); i++) {
                    for (int j = 0; j < result.data[i].size(); j++) {
                        result.data[i][j] = result.data[i][j] * scalar;
                    }
                }
                return result;
        }

        //matrix multiply
        template <typename U>
        Matrix<typename std::common_type<T, U>::type>
        operator*(const Matrix<U>& matrix) {
            int m = data.size(); //number of rows in data
            int n = matrix.data[0].size(); //length of columns of matrix.data
            int n_two = data[0].size(); //length of columns of data
            if (m != n) {
                throw std::invalid_argument("number of rows does not match the number of columns");
            }

            //ensure we only have ints, floats, and doubles
            if constexpr (!std::is_same_v<T, int> && !std::is_same_v<T, double> && !std::is_same_v<T, float> &&
                !std::is_same_v<U, int> && !std::is_same_v<U, double> && !std::is_same_v<U, float>) {
                throw std::invalid_argument("Matrix must be made of numbers");
                }

            //figure out return type
            using ResultType = typename std::common_type<T, U>::type;
            std::vector<std::vector<ResultType>> result(m, std::vector<ResultType>(n, ResultType()));

            //matrix multiplication
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    for (int k = 0; k < n_two; k++) {
                        result[i][j] += data[i][k] * matrix.data[k][j];
                    }
                }
            }
            return Matrix<ResultType>(result);
        }

        //dividing elements in one matrix by the elements in another
        template <typename U>
        Matrix<std::common_type_t<T,U>> operator/ (const Matrix<U>& matrix) const{
            if (shape() != matrix.shape()) {
                throw std::invalid_argument("Matrices do not have the same shape");
            }

            using ResultType = typename std::common_type<T, U>::type;
            Matrix<ResultType> temp(data);
            for (int i = 0; i < data.size(); i++) {
                for (int j = 0; j < matrix.data[i].size(); j++) {
                    if (matrix.data[i][j] != 0) {
                        temp[i][j] = temp[i][j] / matrix[i][j];
                    }
                    else {
                        throw::std::invalid_argument("division by zero");
                    }
                }
            }
            return temp;
        }

    //subtracting elements in one matrix by the elements in another
    template <typename U>
    Matrix<std::common_type_t<T,U>> operator- (const Matrix<U>& matrix) const{
            if (shape() != matrix.shape()) {
                throw std::invalid_argument("Matrices do not have the same shape");
            }

            using ResultType = typename std::common_type<T, U>::type;
            Matrix<ResultType> temp(data);
            for (int i = 0; i < data.size(); i++) {
                for (int j = 0; j < matrix.data[i].size(); j++) {
                    temp[i][j] = temp[i][j] - matrix[i][j];
                }
            }
            return temp;
        }

        std::vector<T>& operator[](int row);

        const std::vector<T>& operator[](int row) const;

        Matrix<T> &operator=(const Matrix<T> &matrix);

        bool operator==(const Matrix<T> &matrix) const;

        std::pair<int, int> shape() const;

        Vec<double> sum(int axis) const;
};



#endif //MATRIX_H
