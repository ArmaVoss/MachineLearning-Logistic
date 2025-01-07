#include "Vec.h"
#include "Parameter.h"
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

template<typename T>
T Vec<T>::sum() const {
    static_assert(std::is_arithmetic<T>::value, "sum() is only valid for arithmetic types.");
    T sum_ = 0; // Only works for arithmetic types like int, float, double.
    for (int i = 0; i < data.size(); ++i) {
        sum_ += data[i];
    }
    return sum_;
}

template<typename T>
int Vec<T>::size () const {
    return data.size();
}
// Explicit template instantiation for supported types
template class Vec<int>;
template class Vec<double>;
template class Vec<float>;
