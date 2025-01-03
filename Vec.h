//
// Created by Arman Vossoughi on 12/26/24.
//

#ifndef VEC_H
#define VEC_H
#include <iostream>
#include <vector>

#include "Matrix.h"

template <typename T>
class Vec {
    private:
        std::vector<T> data; //vector
    public:
        //constructor to create vector in Vec
        Vec(const std::vector<T>& vec){
            data = vec;
        }

        friend std::ostream& operator<<(std::ostream& os, const Vec<T>& vec) {
                os << "[";
                for (int i = 0; i < vec.data.size(); ++i) {
                    os << vec.data[i];
                    if (i < vec.data.size() - 1) {
                        os << ", ";
                    }
                }
                os << "]";
                return os;
        }

        //broadcasting functions
        //add single number across all elements in vector
        template <typename U>
        Vec<T> operator+(const U& scalar) const {
            static_assert(std::is_same<T,U>::value, "Can't add u to vector of type v");

            std::vector<T> result = data;
            for (int i = 0; i<data.size(); i++) {
                result[i] += scalar;
            }
            return Vec<T>(result);
        }

        //multiply single number across all elements in vector
        template <typename U>
        Vec<T> operator*(const U& scalar) const {
                static_assert(std::is_same<T,U>::value, "Can't multiply u to vector of type v");

                std::vector<T> result = data;
                for (int i = 0; i<data.size(); i++) {
                    result[i] *= scalar;
                }
                return Vec<T>(result);
        }

        //subtract single number across all elements in vector
        template <typename U>
        Vec<T> operator-(const U& scalar) const {
                static_assert(std::is_same<T,U>::value, "Can't subtract u from vector of type v");

                std::vector<T> result = data;
                for (int i = 0; i<data.size(); i++) {
                    result[i] -= scalar;
                }
                return Vec<T>(result);
        }

        //divide single number across all elements in vector
        template <typename U>
        Vec<T> operator/(const U& scalar) const {
                static_assert(std::is_same<T,U>::value, "Can't divide u from vector of type v");

                std::vector<T> result = data;
                for (int i = 0; i<data.size(); i++) {
                    result[i] /= scalar;
                }
                return Vec<T>(result);
        }

        //vector operations
        //add vectors element-wise [1,2,3] + [1,2,3] = [2,4,6]
        template <typename U>
        Vec<T> operator+(const Vec<U>& vec) const {
                if (data.size() != vec.data.size()) {
                    throw std::invalid_argument("Vectors must be of the same size for addition.");
                }


                std::vector<T> result = data;
                for (int i = 0; i<data.size(); i++) {
                    result[i] += vec.data[i];
                }
                return Vec<T>(result);
        }

        template <typename U>
        Vec<T> operator-(const Vec<U>& vec) const {
                if (data.size() != vec.data.size()) {
                    throw std::invalid_argument("Vectors must be of the same size for addition.");
                }

                std::vector<T> result = data;
                for (int i = 0; i<data.size(); i++) {
                    result[i] -= vec.data[i];
                }
                return Vec<T>(result);
        }

        template <typename U>
        double operator*(const Vec<U>& vec) const {
            if (data.size() == 0) {
                throw std::invalid_argument("Cannot dot product empty vector.");
            }
            if (data.size() != vec.data.size()) {
                throw std::invalid_argument("Vectors must be of the same size for dot product.");
            }

            double sum = 0.0;
            std::vector<T> result = data;
            for (int i = 0; i<data.size(); i++) {
                sum += (vec.data[i] * result[i]);
            }

            return sum;
        }

        Vec& operator=(const Vec& vec);

        bool operator==(const Vec& vec) const;

        //Read the value
        const T& operator[](const int index) const;

        //Update the value
        T& operator[](int index);

        T sum() const;

};



#endif //VEC_H
