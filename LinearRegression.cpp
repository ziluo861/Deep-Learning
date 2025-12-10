#include <cassert>
#include <cstdlib>
#include <limits>
#include <print>
#include <random>
#include <utility>
#include <vector>

enum class RegType {
  None,
  Lasso,
  Ridge,
  ElasticNet,
};

struct Data {
  std::vector<double> x;
  double y;
};

class LinearRegression {
public:
  LinearRegression(std::vector<Data> &&data, size_t wSize,
                   RegType type = RegType::None, double learningrate = 0.01,
                   double b = 0.0, double lambda1 = 0.01, double lambda2 = 0.01)
      : datas_(std::move(data)), nSize_(datas_.size()), wSize_(wSize),
        w_(wSize_, 0), type_(type), learningrate_(learningrate), b_(b),
        lambda1_(lambda1), lambda2_(lambda2) {
    for (const auto &d : datas_) {
      assert(d.x.size() == wSize_ && "Feature dimension mismatch!");
    }
    if (type_ == RegType::None) {
      lambda1_ = lambda2_ = 0.0;
    } else if (type_ == RegType::Lasso) {
      lambda2_ = 0;
    } else if (type_ == RegType::Ridge) {
      lambda1_ = 0;
    }
  }
  void train(size_t epochs = 2000, double tolerance = 1e-6) {
    double prevloss = std::numeric_limits<double>::max();
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
      std::vector<double> gradw(wSize_, 0);
      double gradb = 0.0;
      computeGradient(gradw, gradb);

      for (size_t i = 0; i < wSize_; ++i) {
        w_[i] -= learningrate_ * gradw[i];
      }
      b_ -= learningrate_ * gradb;

      double currentloss = computeLoss();

      if (epoch == 0 || ((epoch + 1) % 10) == 0) {
        std::print("epoch = {}, w = [", epoch + 1);
        for (size_t i = 0; i < wSize_; ++i) {
          i &&std::printf(" ");
          std::print("{}", w_[i]);
        }
        std::println("], b = {}, loss = {}", b_, currentloss);
      }

      if (std::abs(currentloss - prevloss) < tolerance) {
        std::println("Converged at epoch {}", epoch + 1);
        break;
      }

      prevloss = currentloss;
    }
  }
  double predict(const std::vector<double> &x) const { return dot(w_, x) + b_; }
  const std::vector<double> &getW() const { return w_; }
  double getB() const { return b_; }

private:
  std::vector<Data> datas_;
  size_t nSize_, wSize_;
  std::vector<double> w_;
  RegType type_;
  double learningrate_;
  double b_;
  double lambda1_, lambda2_;

  double dot(const std::vector<double> &a, const std::vector<double> &b) const {
    assert(a.size() == b.size());
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
      sum += a[i] * b[i];
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
      lasso += lambda1_ * std::abs(w_[i]);
    }
    return lasso;
  }
  double computeRidge() const {
    double ridge = 0.0;
    for (size_t i = 0; i < wSize_; ++i) {
      ridge += w_[i] * w_[i];
    }
    return ridge * (lambda2_ / 2.0);
  }
  double computeLoss() const {
    double sumloss = 0.0, dataloss = 0.0;
    for (const auto &cur : datas_) {
      double y_hat = dot(w_, cur.x) + b_;
      double err = y_hat - cur.y;
      dataloss += err * err;
    }
    dataloss /= (2.0 * static_cast<double>(nSize_));
    if (type_ == RegType::None) {
      sumloss = dataloss;
    } else if (type_ == RegType::Lasso) {
      sumloss = dataloss + computeLasso();
    } else if (type_ == RegType::Ridge) {
      sumloss = dataloss + computeRidge();
    } else if (type_ == RegType::ElasticNet) {
      sumloss = dataloss + computeLasso() + computeRidge();
    }
    return sumloss;
  }
  void computeGradient(std::vector<double> &gradw, double &gradb) const {
    std::vector<double> sum_dw(wSize_, 0);
    double sum_db = 0;

    for (const auto &cur : datas_) {
      double y_hat = dot(w_, cur.x) + b_;
      double err = y_hat - cur.y;
      for (size_t i = 0; i < wSize_; ++i) {
        sum_dw[i] += err * cur.x[i];
      }
      sum_db += err;
    }
    for (size_t i = 0; i < wSize_; ++i) {
      sum_dw[i] /= static_cast<double>(nSize_);
    }
    if (type_ == RegType::Lasso) {
      for (size_t i = 0; i < wSize_; ++i) {
        sum_dw[i] += lambda1_ * sign(w_[i]);
      }
    } else if (type_ == RegType::Ridge) {
      for (size_t i = 0; i < wSize_; ++i) {
        sum_dw[i] += lambda2_ * w_[i];
      }
    } else if (type_ == RegType::ElasticNet) {

      for (size_t i = 0; i < wSize_; ++i) {
        sum_dw[i] += lambda1_ * sign(w_[i]);
        sum_dw[i] += lambda2_ * w_[i];
      }
    }
    gradw = std::move(sum_dw);
    gradb = sum_db / static_cast<double>(nSize_);
  }
};

int main() {
  std::vector<Data> data;
  {
    std::mt19937 rng_(42);
    std::normal_distribution<double> noise(0.0, 0.1);
    for (int i = 0; i < 100; ++i) {
      double x1 = static_cast<double>(i) / 10.0;
      double x2 = static_cast<double>(100 - i) / 10.0;
      double y_true = 3.0 * x1 - 2.0 * x2 + 1.0;
      double y = y_true + noise(rng_);
      data.emplace_back(Data{{x1, x2}, y});
    }
  }
  const int wSize = 2;
  RegType type = RegType::ElasticNet;
  const double learning_rate = 0.01;
  const double b = 0.0;
  const double lambda1 = 0.02;
  const double lambda2 = 0.03;
  LinearRegression linear(std::move(data), wSize, type, learning_rate, b,
                          lambda1, lambda2);
  std::println("Start training...");
  linear.train();
  std::println("training finish");
  std::print("Learned parameters:\nw = [");
  const auto &w = linear.getW();
  for (int i = 0; i < wSize; ++i) {
    i &&std::printf(" ");
    std::print("{}", w[i]);
  }
  std::println("], b = {}", linear.getB());
  std::println("test predictions");
  std::vector<std::vector<double>> test_x = {
      {0.0, 0.0}, {1.0, 1.0}, {2.0, 1.0}, {1.0, 2.0}, {5.0, 3.0}};
  for (const auto &x : test_x) {
    double y_hat = linear.predict(x);
    std::println("x = [{}, {}], y_hat = {}", x[0], x[1], y_hat);
  }
  return 0;
}
