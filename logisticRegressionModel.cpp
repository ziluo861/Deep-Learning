#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <print>
#include <random>
#include <utility>
#include <vector>

struct Data {
  std::vector<double> x;
  double y;
};

enum class RegType {
  None,
  Lasso,
  Ridge,
  ElasticNet,
};

class LogisticRegression {
public:
  explicit LogisticRegression(std::vector<Data> &&data, size_t wSize,
                              size_t batchSize, RegType type = RegType::None,
                              double learningrate = 0.01, double b = 0.0,
                              double lamdba1 = 0.01, double lamdba2 = 0.01)
      : data_(std::move(data)), nSize_(data_.size()), wSize_(wSize),
        batch_(batchSize), w_(wSize_, 0), type_(type),
        learningrate_(learningrate), b_(b), lamdba1_(lamdba1),
        lamdba2_(lamdba2), rng_(std::random_device{}()) {
    for (const auto &d : data_) {
      if (d.x.size() != wSize_) {
        throw std::runtime_error("feature dimension mismatch");
      }
    }
  }
  double predict(const std::vector<double> &x) const {
    double z = dot(x) + b_;
    if (z >= 0) {
      double expz = std::exp(-z);
      return 1.0 / (1.0 + expz);
    }
    double value = std::exp(z);
    return value / (1.0 + value);
  }
  void train(size_t epochs = 2000, double tolerance = 1e-9) {
    double prevloss = std::numeric_limits<double>::max();
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
      std::shuffle(data_.begin(), data_.end(), rng_);
      for (size_t i = 0; i < nSize_; i += batch_) {
        size_t start = i;
        size_t currentsize = std::min(batch_, nSize_ - i);
        std::vector<double> gradw(wSize_, 0.0);
        double gradb = 0.0;
        computeGradient(gradw, gradb, start, currentsize);
        for (size_t k = 0; k < wSize_; ++k) {
          w_[k] -= learningrate_ * gradw[k];
        }
        b_ -= learningrate_ * gradb;
      }

      double currentloss = computeLoss();
      if (epoch == 0 || ((epoch + 1) % 10) == 0) {
        std::print("epoch = {}, w = [", epoch + 1);
        for (size_t i = 0; i < wSize_; ++i) {
          i &&std::printf(" ");
          std::print("{}", w_[i]);
        }
        std::println("], b = {}, loss = {}", b_, currentloss);
      }
      if (std::abs((currentloss - prevloss)) < tolerance) {
        std::println("Converged at epoch {}", epoch + 1);
        break;
      }
      prevloss = currentloss;
    }
  }

  const std::vector<double> &getW() const { return w_; }
  double getB() const { return b_; }

private:
  std::vector<Data> data_;
  size_t nSize_, wSize_, batch_;
  std::vector<double> w_;
  RegType type_;
  double learningrate_;
  double b_;
  double lamdba1_, lamdba2_;
  std::mt19937 rng_;

  double dot(const std::vector<double> &x) const {
    assert(x.size() == wSize_);
    double sum = 0.0;
    for (size_t i = 0; i < wSize_; ++i) {
      sum += w_[i] * x[i];
    }
    return sum;
  }

  double sign(double x) const {
    if (x > 0)
      return 1.0;
    if (x < 0)
      return -1.0;
    return 0.0;
  }

  double computeLasso() const {
    double lasso = 0.0;
    for (size_t i = 0; i < wSize_; ++i) {
      lasso += lamdba1_ * std::abs(w_[i]);
    }
    return lasso / static_cast<double>(nSize_);
  }

  double computeRidge() const {
    double rigde = 0.0;
    for (size_t i = 0; i < wSize_; ++i) {
      rigde += lamdba2_ * w_[i] * w_[i];
    }
    return rigde / (2.0 * static_cast<double>(nSize_));
  }

  double computeLoss() const {
    double dataloss = 0.0, sumloss = 0.0, eps = 1e-15;
    for (const auto &cur : data_) {
      double y = cur.y;
      double p = std::clamp(predict(cur.x), eps, 1 - eps);
      dataloss += (y * std::log(p) + (1 - y) * std::log(1 - p));
    }
    sumloss = -(dataloss / static_cast<double>(nSize_));
    switch (type_) {
    case RegType::Lasso:
      sumloss += computeLasso();
      break;
    case RegType::Ridge:
      sumloss += computeRidge();
      break;
    case RegType::ElasticNet:
      sumloss += computeLasso() + computeRidge();
      break;
    default:
      break;
    }
    return sumloss;
  }

  void computeGradient(std::vector<double> &gradw, double &gradb, size_t start,
                       size_t currentsize) const {
    std::vector<double> sum_dw(wSize_, 0.0);
    double sum_db = 0.0;
    for (size_t i = 0; i < currentsize; ++i) {
      const auto &cur = data_[start + i];
      double value = predict(cur.x) - cur.y;
      for (size_t j = 0; j < wSize_; ++j) {
        sum_dw[j] += value * cur.x[j];
      }
      sum_db += value;
    }
    for (size_t i = 0; i < wSize_; ++i) {
      sum_dw[i] /= static_cast<double>(currentsize);
    }

    double inv_n = 1.0 / static_cast<double>(nSize_);
    switch (type_) {
    case RegType::Lasso:
      for (size_t i = 0; i < wSize_; ++i) {
        sum_dw[i] += lamdba1_ * sign(w_[i]) * inv_n;
      }
      break;
    case RegType::Ridge:
      for (size_t i = 0; i < wSize_; ++i) {
        sum_dw[i] += lamdba2_ * w_[i] * inv_n;
      }
      break;
    case RegType::ElasticNet:
      for (size_t i = 0; i < wSize_; ++i) {
        sum_dw[i] += (lamdba1_ * sign(w_[i]) + lamdba2_ * w_[i]) * inv_n;
      }
      break;
    default:
      break;
    }

    gradw.swap(sum_dw);
    gradb = sum_db / static_cast<double>(currentsize);
  }
};
