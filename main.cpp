#include "Vec.h"
#include "Matrix.h"

template <typename T>
class Parameter{
    private:
        Matrix<T> val;
        Matrix<T> grad;
    public:
        Parameter(Matrix<T> val_, Matrix<T> grad_) {
            val = val_;
            grad = grad_;
        }

        Parameter& reset() {
            //reset the gradient
            for (int i = 0; i < grad.size(); i++) {
                for (int j = 0; j < grad[i].size(); j++) {
                    grad[i][j] = 0;
                }
            }
            return *this;
        }

        Parameter& step(const Matrix<T>& gradient_update) {
                if (grad.size() != val.size() || grad[0].size() != val[0].size() ||
                    gradient_update.size() != val.size() || gradient_update[0].size() != val[0].size()) {
                    throw std::invalid_argument("size of new gradient does not size of val");
                    }

                for (int i = 0; i < val.size(); i++) {
                    for (int j = 0; j < val[i].size(); j++) {
                        val[i][j] -= gradient_update[i][j];
                    }
                }
                return reset();
            }


        // Set a new gradient
        void setGradient(const Matrix<T>& newGrad) {
                if (newGrad.size() != grad.size() || newGrad[0].size() != grad[0].size()) {
                    throw std::invalid_argument("New gradient size must match the current gradient size.");
                }
                grad = newGrad;
        }

        const Matrix<T>& getValue() const { return val; }

        const Matrix<T>& getGradient() const { return grad; }
};

//abstract base class for Module
template <typename T>
class Module {
    public:
        virtual ~Module() = default;

        virtual Matrix<T> forward(const Matrix<T>& X) = 0;

        virtual Matrix<T> backward(const Matrix<T>& X, const Matrix<T>& dLoss_dOutput) = 0;

        virtual Vec<Parameter<T>> parameters() = 0;
};

//abstract base class for Loss Functions
template <typename T>
class LossFunction {
    public:
        virtual ~LossFunction() = default;
        virtual double forward(const Matrix<T>& yHat, const Matrix<T>& yGT) = 0;
        virtual Matrix<T> backward(const Matrix<T>& yHat, const Matrix<T>& yGT) = 0;
};



int main() {
    Matrix<int> m1 ({{1,2},{3,4}});
    Matrix<int> m2 ({{1,2},{3,4}});
    Matrix m3 = m1;
    m3[0][1] += 1;
    std::cout << m1 << '\n';
    std::cout << m3 << '\n';
}