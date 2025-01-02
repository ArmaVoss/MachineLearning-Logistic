#include "Vec.h"

//Read
template<typename T>
const T& Vec<T>::operator[](const int index) const {
    return data.at(index);
}

//Update
template<typename T>
T& Vec<T>::operator[](int index) {
    return data.at(index);
}

template<typename T>
Vec<T>& Vec<T>::operator=(const Vec<T>& vec) {
    if (this != &vec) {
        data = vec.data;
    }
    return *this;
}

template<typename T>
bool Vec<T>::operator==(const Vec<T>& vec) const {
    return (data == vec.data);
}

// Explicit template instantiation for supported types
template class Vec<int>;
template class Vec<double>;
template class Vec<float>;
