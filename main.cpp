#include "Vec.h"
#include "Matrix.h"
#include <cmath>
#include <random>
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

        //reset the gradient
        Parameter& reset() {
            for (int i = 0; i < grad.size(); i++) {
                for (int j = 0; j < grad[i].size(); j++) {
                    grad[i][j] = 0;
                }
            }
            return *this;
        }

        //step parameter
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

        //computes the forward method
        virtual Matrix<T> forward(Matrix<T>& X) = 0;

        //computes the partial derivative using chain rule
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

class BCE : public LossFunction<double> {
public:
    virtual ~BCE() = default;

    bool checkinput(const Matrix<double>& yHat, const Matrix<double>& yGT) {
        return (yHat.shape() == yGT.shape());
    }
    double forward(const Matrix<double>& yHat, const Matrix<double>& yGT) override {
        if (!checkinput(yHat, yGT)) {
            throw std::invalid_argument("size of yHat does not match size of yGT.");
        }
        std::vector<int> probabilitiesZero;
        std::vector<int> probabilitiesOne;
        int lenRow = yGT.shape().first;
        int lenColumn = yGT.shape().second;
        double loss = 0;
        for (int i = 0; i < lenRow; i++) {
            for (int j = 0; j < lenColumn; j++) {
                double y = yGT[i][j];
                double y_hat = yHat[i][j];

                loss += y * std::log(y_hat) + (1 - y) * std::log(1 - y_hat);
            }
        }
        return loss / (lenRow * lenColumn);
    }

    Matrix<double> backward(const Matrix<double>& yHat, const Matrix<double>& yGT) override {
        if (!checkinput(yHat, yGT)) {
            throw std::invalid_argument("yHat does not match size of yGT.");
        }

        Matrix<double> dLoss_dY_hat = (yGT / (yHat * yHat.shape().first)) -
                                      (yGT + (-1)) / ((yGT + (-1)) * yHat.shape().first);

        return dLoss_dY_hat;
    }
};

template <typename T>
class Optimizer {
    public:
        Vec<Parameter<T>> parameters;
        double learningRate;

        Optimizer(Vec<Parameter<T>> parameters_, double learningRate_) {
            parameters = parameters_;
            learningRate = learningRate_;
        }

        void Reset() {
            for (auto& param : parameters) {
                param.reset();
            }
        }

        virtual void step() = 0;
};

template <typename T>
class GradientDescentOptimizer : public Optimizer<T> {
    public:
        void step() override {
            for (auto& param: this->parameters) {
                param.step(param.getGradient() * this->learningRate);
            }
        }
};

template <typename T>
class Sigmoid: public Module<T> {
    public:
        //need to implement more matrix operations to do this more easily
        Matrix<T> forward(Matrix<T>& X) override{
            Matrix copy = X;
            for (int i = 0; i < X.shape().first; i++) {
                for (int j = 0; j < X.shape().second; j++) {
                    copy[i][j] = 1 / (1 + exp(-copy[i][j]));
                }
            }
            return copy;
        }

        Matrix<T> backward(const Matrix<T>& X, const Matrix<T>& dLoss_dModule) override {
            Matrix sig = forward(X);
            Matrix dModule_dx = sig * (sig + -1);
            Matrix dLoss_dx = dModule_dx * dLoss_dModule;
            if (X.shape() != dLoss_dx.shape()) {
                throw::std::invalid_argument("Size of dLoss_dX does not match size of X.");
            }
            return dLoss_dx;
        }

        Vec<Parameter<T>> parameters() override {
            std::vector<Parameter<T>> v;
            return Vec<Parameter<T>>(v);
        }
};

template <typename T>
class LinearRegression: public Module<T> {
    private:
        Matrix<T> betas;
    public:
        LinearRegression(int num_features) {
            std::vector<std::vector<T>> features((num_features+1), std::vector<T>(1));
            Matrix<double> values(features);
            std::default_random_engine generator;
            std::normal_distribution<double> distribution(0.0, 1.0);

            for (int i = 0; i < num_features + 1; ++i) {
                values[i][0] = distribution(generator);
            }
        }

};

int main() {
    Matrix<int> m1 ({{1,2},{3,4}});
    Vec<double> v1 = m1.sum(0);
    Vec<double> v2 = m1.sum(1);
    std::cout << v1 << '\n';
    std::cout << v2 << '\n';
}