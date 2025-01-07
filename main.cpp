#include "Vec.h"
#include "Matrix.h"
#include "Parameter.h"
#include <cmath>
#include <random>

template <typename T>
Matrix<T> elementwiseMultiply(const Matrix<T>& a, const Matrix<T>& b) {
    auto shapeA = a.shape();
    auto shapeB = b.shape();

    if (shapeA != shapeB) {
        throw std::invalid_argument("Matrices must have the same shape for element-wise multiplication.");
    }

    std::vector<std::vector<T>> resultData(a.data.size(), std::vector<T>(a.data[0].size()));

    for (int i = 0; i < a.data.size(); ++i) {
        for (int j = 0; j < a.data[i].size(); ++j) {
            resultData[i][j] = a.data[i][j] * b.data[i][j];
        }
    }

    return Matrix<T>(resultData);
}

template <typename T>
void printShape(const std::string& name, const Matrix<T>& mat) {
    auto shape = mat.shape();
    std::cout << name << " shape: (" << shape.first << ", " << shape.second << ")" << std::endl;
}

template <typename T>
void clampMatrix(Matrix<T>& mat, T min_val, T max_val) {
    for (int i = 0; i < mat.shape().first; ++i) {
        for (int j = 0; j < mat.shape().second; ++j) {
            mat[i][j] = std::max(std::min(mat[i][j], max_val), min_val);
        }
    }
}

template <typename T>
class Module {
    public:
        virtual ~Module() = default;
        virtual Matrix<T> forward(const Matrix<T>& X) = 0;
        virtual Matrix<T> backward(const Matrix<T>& X, const Matrix<T>& dLoss_dOutput) = 0;
        virtual std::vector<Parameter<T>> parameters() = 0;
};

template <typename T>
class LossFunction {
    public:
        virtual ~LossFunction() = default;
        virtual double forward(const Matrix<T>& yHat, const Matrix<T>& yGT) = 0;
        virtual Matrix<double> backward(const Matrix<T>& yHat, const Matrix<T>& yGT) = 0;
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
        int lenRow = yGT.shape().first;
        int lenColumn = yGT.shape().second;
        double loss = 0;
        double eps = 1e-7;
        for (int i = 0; i < lenRow; i++) {
            for (int j = 0; j < lenColumn; j++) {
                double y = yGT[i][j];
                double y_hat = yHat[i][j];
                double y_hat_safe = std::max(std::min(y_hat, 1.0 - eps), eps);
                loss += y * std::log(y_hat_safe) + (1.0 - y) * std::log(1.0 - y_hat_safe);
            }
        }
        return loss / (lenRow * lenColumn);
    }

    Matrix<double> backward(const Matrix<double> &yHat, const Matrix<double> &yGT) override {

        if (!checkinput(yHat, yGT)) {
            throw std::invalid_argument("yHat does not match size of yGT.");
        }
        Matrix yHC = yHat;
        Matrix yGTC = yGT;
        double eps = 1e-7;
        clampMatrix(yHC, eps, 1.0-eps);
        clampMatrix(yGTC, eps, 1.0-eps);

        Matrix<double> dLoss_dY_hat = (yGTC / (yHC * yHC.shape().first))
                                      - (yGTC + (-1)) / ((yGTC + (-1)) * yHat.shape().first);
        return dLoss_dY_hat;
    }
};

template <typename T>
class Optimizer {
    public:
        std::vector<Parameter<T>> parameters;
        double learningRate;

        Optimizer(std::vector<Parameter<T>> parameters_, double learningRate_)
            : parameters(parameters_), learningRate(learningRate_) {}

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
        GradientDescentOptimizer(std::vector<Parameter<T>> parameters_, double learningRate_)
           : Optimizer<T>(parameters_, learningRate_) {}

        void step() override {
            for (auto& param: this->parameters) {
                param.step(param.getGradient() * this->learningRate);
            }
        }
};

template <typename T>
class Sigmoid: public Module<T> {
    public:
        Matrix<T> forward(const Matrix<T>& X) override {
            Matrix<T> copy = X;
            for (int i = 0; i < X.shape().first; i++) {
                for (int j = 0; j < X.shape().second; j++) {
                    copy[i][j] = 1.0 / (1.0 + std::exp(-copy[i][j]));
                }
            }
            return copy;
        }

    Matrix<T> backward(const Matrix<T>& X, const Matrix<T>& dLoss_dModule) override {
            Matrix<T> sig = forward(X);
            Matrix<T> sig_minus_one = sig + (-1.0);  // Element-wise subtraction
            Matrix<T> dModule_dx = elementwiseMultiply(sig, sig_minus_one);
            Matrix<T> dLoss_dx = elementwiseMultiply(dModule_dx, dLoss_dModule);
            if (X.shape() != dLoss_dx.shape()) {
                throw std::invalid_argument("Size of dLoss_dX does not match size of X.");
            }
            return dLoss_dx;
        }

        std::vector<Parameter<T>> parameters() override {
            return {};
        }
};

template <typename T>
class LinearRegression: public Module<T> {
    private:
        Parameter<double> betas;
    public:
        LinearRegression(int num_features)
            : betas(
                  Matrix<double>(std::vector<std::vector<double>>(num_features + 1, std::vector<double>(1))),
                  Matrix<double>(std::vector<std::vector<double>>(num_features + 1, std::vector<double>(1, 0.0)))
              )
        {
            std::default_random_engine generator;
            std::normal_distribution<double> distribution(0.0, 1.0);
            Matrix<double>& values = betas.getValue();
            for (int i = 0; i < num_features + 1; ++i) {
                values[i][0] = distribution(generator);
            }
            Matrix<double> gradient(std::vector<std::vector<double>>((num_features + 1), std::vector<double>(1, 0)));
        }

        Matrix<double> forward(const Matrix<double>& X) override {
            Matrix<double> weights = betas.getValue();
            std::vector<std::vector<double>> featureWeights(weights.data.begin(), weights.data.end() - 1);
            Matrix<double> featureMatrix(featureWeights);
            double intercept = betas.getValue()[betas.getValue().shape().first - 1][0];
            Matrix<double> weightedSum = X * featureMatrix;
            for (int i = 0; i < weightedSum.shape().first; i++) {
                for (int j = 0; j < weightedSum.shape().second; j++) {
                    weightedSum[i][j] += intercept;
                }
            }
            return weightedSum;
        }

        Matrix<double> backward(const Matrix<double>& X, const Matrix<double>& dLoss_dModule) override {
            Matrix<double> dModule_dBeta = X.transpose();
            std::vector<double> onesRow(dModule_dBeta.shape().second, 1.0);
            dModule_dBeta = dModule_dBeta.vstack(onesRow);
            Matrix<double> dLoss_dBeta = dModule_dBeta * dLoss_dModule;
            betas.setGradient(betas.getGradient() + dLoss_dBeta);
            Matrix<double> dLoss_dX = dLoss_dModule * betas.getValue().transpose();
            return dLoss_dX;
        }

        std::vector<Parameter<T>> parameters() override {
            return {betas};
        }
};

template <typename T>
class LogisticRegression : public Module<T> {
    private:
        int num_features;
        LinearRegression<T> linear_regression;
        Sigmoid<T> sigmoid = Sigmoid<T>();
    public:
        LogisticRegression(int num_features_)
            : num_features(num_features_),
              linear_regression(LinearRegression<T>(num_features)),
              sigmoid(Sigmoid<T>()) {}

        Matrix<double> forward(const Matrix<double>& X) override {
            Matrix<double> copy = linear_regression.forward(X);
            return sigmoid.forward(copy);
        }

        Matrix<double> backward(const Matrix<double>& X, const Matrix<double>& dLoss_dModule) override {
            Matrix<double> A = linear_regression.forward(X);
            Matrix<double> dLoss_dA = sigmoid.backward(A, dLoss_dModule);
            Matrix<double> dLoss_dx = linear_regression.backward(X, dLoss_dA);
            return dLoss_dx;
        }

        std::vector<Parameter<T>> parameters() override {
            std::vector<Parameter<T>> x = linear_regression.parameters();
            std::vector<Parameter<T>> y = sigmoid.parameters();
            std::vector<Parameter<T>> param;
            for (int i = 0; i < x.size(); i++){
                param.push_back(x[i]);
            }
            for (int i = 0; i < y.size(); i++){
                param.push_back(y[i]);
            }
            return param;
        }
};

int main() {
    std::default_random_engine generator(12345);
    int num_features = 100;
    int num_examples = 200;
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    std::vector<std::vector<double>> xData(num_examples, std::vector<double>(num_features));
    for (int i = 0; i < num_examples; ++i) {
        for (int j = 0; j < num_features; ++j) {
            xData[i][j] = normal_dist(generator);
        }
    }
    Matrix<double> X(xData);
    std::uniform_int_distribution<int> binary_dist(0, 1);
    std::vector<std::vector<double>> yData(num_examples, std::vector<double>(1));
    for (int i = 0; i < num_examples; ++i) {
        yData[i][0] = static_cast<double>(binary_dist(generator));
    }
    Matrix<double> Y_gt(yData);
    auto xShape = X.shape();
    auto yShape = Y_gt.shape();
    std::cout << "X shape: (" << xShape.first << ", " << xShape.second << ")\n";
    std::cout << "Y_gt shape: (" << yShape.first << ", " << yShape.second << ")\n";

    double lr = 0.1;
    int max_epochs = 1000;
    LogisticRegression<double> m(num_features);
    GradientDescentOptimizer<double> optim(m.parameters(), lr);
    BCE loss_func;

    std::vector<double> losses;
    for (int i = 0; i < max_epochs; ++i) {
        optim.Reset();
        Matrix<double> y_hat = m.forward(X);
        losses.push_back(loss_func.forward(y_hat, Y_gt));
        Matrix<double> dLoss = loss_func.backward(y_hat, Y_gt);
        m.backward(X, dLoss);
        optim.step();
    }
    Vec loss(losses);
    std::cout << loss;
    return 0;
}
