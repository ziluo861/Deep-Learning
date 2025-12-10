#include <cassert>
#include <cstdlib>
#include <limits>
#include <print>
#include <random>
#include <utility>
#include <vector>

struct Data {
  std::vector<double> x; // 特征向量
  double y;              // 标签
};

class MultiLinearRegressionModel {
public:
  explicit MultiLinearRegressionModel(std::vector<Data> &&data, int wSize,
                                      double learning_rate, double lambda,
                                      double b = 0)
      : datas_(std::move(data)), nSize_(static_cast<int>(datas_.size())),
        wSize_(wSize), w_(wSize, 0), learning_rate_(learning_rate),
        lambda_(lambda), b_(b) {
    for (const auto &d : datas_) {
      assert(static_cast<int>(d.x.size()) == wSize_);
    }
  }

  void trainModel(int epochs = 2000, double tolerance = 1e-6) {
    double prevloss = std::numeric_limits<double>::max();
    for (int epoch = 0; epoch < epochs; ++epoch) {

      std::vector<double> gradw(wSize_, 0);
      double gradb = 0.0;
      computeGradientL2(gradw, gradb);

      for (int i = 0; i < wSize_; ++i) {
        w_[i] -= learning_rate_ * gradw[i];
      }
      b_ -= learning_rate_ * gradb;

      double currentloss = computeLossL2();
      if (epoch == 0 || (epoch + 1) % 10 == 0) {
        std::print("epoch = {}, w = [", epoch + 1);
        for (int i = 0; i < wSize_; ++i) {
          i &&std::printf(" ");
          std::print("{}", w_[i]);
        }
        std::println("], b = {}, loss = {}", b_, currentloss);
      }
      if (std::abs(prevloss - currentloss) < tolerance) {
        std::println("Converged at epoch {}", epoch + 1);
        return;
      }
      prevloss = currentloss;
    }
  }
  double prediction(const std::vector<double> &x) const {
    return dot(w_, x) + b_;
  }
  const std::vector<double> &getw() const { return w_; }
  double getb() const { return b_; }

private:
  std::vector<Data> datas_;
  int nSize_, wSize_;
  std::vector<double> w_;
  double learning_rate_;
  double lambda_;
  double b_;

  double dot(const std::vector<double> &a, const std::vector<double> &b) const {
    assert(a.size() == b.size());
    double sum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      sum += a[i] * b[i];
    }
    return sum;
  }

  double computeLoss() const {
    double sumloss = 0;
    for (const auto &cur : datas_) {
      double y_hat = dot(w_, cur.x) + b_;
      double err = y_hat - cur.y;
      sumloss += err * err;
    }
    return sumloss / (2.0 * static_cast<double>(nSize_));
  }

  void computeGradient(std::vector<double> &gradw, double &gradb) const {
    std::vector<double> sum_dw(wSize_, 0);
    double sum_db = 0;
    for (const auto &cur : datas_) {
      const auto &x = cur.x;
      double y_hat = dot(w_, x) + b_;
      double err = y_hat - cur.y;
      for (int i = 0; i < wSize_; ++i) {
        sum_dw[i] += err * x[i];
      }
      sum_db += err;
    }
    for (int i = 0; i < wSize_; ++i) {
      sum_dw[i] /= static_cast<double>(nSize_);
    }
    gradw = std::move(sum_dw);

    gradb = sum_db / static_cast<double>(nSize_);
  }

  double computeLossL2() const {
    double dataloss = 0.0;
    for (const auto &cur : datas_) {
      double y_hat = dot(w_, cur.x) + b_;
      double err = y_hat - cur.y;
      dataloss += err * err;
    }
    double sumloss = dataloss / (2.0 * static_cast<double>(nSize_));

    double reg = 0.0;
    for (int i = 0; i < wSize_; ++i) {
      reg += w_[i] * w_[i];
    }
    reg *= (lambda_ / 2.0);

    return sumloss + reg;
  }

  void computeGradientL2(std::vector<double> &gradw, double &gradb) const {
    std::vector<double> sum_dw(wSize_, 0);
    double sum_db = 0.0;

    for (const auto &cur : datas_) {
      double y_hat = dot(w_, cur.x) + b_;
      double err = y_hat - cur.y;
      for (int i = 0; i < wSize_; ++i) {
        sum_dw[i] += err * cur.x[i];
      }
      sum_db += err;
    }

    for (int i = 0; i < wSize_; ++i) {
      sum_dw[i] /= static_cast<double>(nSize_);
      sum_dw[i] += lambda_ * w_[i];
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
  const double learning_rate = 0.01;
  const double lambda = 0.03;
  MultiLinearRegressionModel linear(std::move(data), wSize, learning_rate,
                                    lambda);

  std::println("Start training...");
  linear.trainModel();
  std::println("training finish");
  std::print("Learned parameters:\nw = [");
  const auto &w = linear.getw();
  for (int i = 0; i < wSize; ++i) {
    i &&std::printf(" ");
    std::print("{}", w[i]);
  }
  std::println("], b = {}", linear.getb());

  std::println("test predictions");
  std::vector<std::vector<double>> test_x = {
      {0.0, 0.0}, {1.0, 1.0}, {2.0, 1.0}, {1.0, 2.0}, {5.0, 3.0}};
  for (const auto &x : test_x) {
    double y_hat = linear.prediction(x);
    std::println("x = [{}, {}], y_hat = {}", x[0], x[1], y_hat);
  }
  return 0;
}
