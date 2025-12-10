#include <algorithm>
#include <limits>
#include <print>
#include <random>
#include <utility>
#include <vector>

struct Data {
  std::vector<double> x;
  double y;
};

class MiniBatch {
public:
  MiniBatch(std::vector<Data> &&data, size_t wSize, double b = 0.0,
            double learningrate = 0.01, size_t batch = 32)
      : datas_(std::move(data)), nSize_(datas_.size()), wSize_(wSize),
        w_(wSize, 0.0), b_(b), learningrate_(learningrate), batch_(batch),
        rng_(std::random_device{}()) {
    for (const auto &d : datas_) {
      if (d.x.size() != wSize_) {
        throw std::runtime_error("feature dimension mismatch");
      }
    }
  }
  void train(size_t epochs = 2000, double tolerance = 1e-6) {
    double prevloss = std::numeric_limits<double>::max();
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
      std::shuffle(datas_.begin(), datas_.end(), rng_);
      for (size_t i = 0; i < nSize_; i += batch_) {
        size_t first = i;
        size_t current_batch = std::min(nSize_ - i, batch_);
        std::vector<double> gradw(wSize_, 0.0);
        double gradb = 0.0;
        computeGradient(gradw, gradb, first, current_batch);

        for (size_t k = 0; k < wSize_; ++k) {
          w_[k] -= learningrate_ * gradw[k];
        }
        b_ -= learningrate_ * gradb;
      }
      double currentloss = computeLoss();
      if (std::abs(currentloss - prevloss) < tolerance) {
        std::println("Converged at epoch {}", epoch + 1);
        break;
      }
      if (epoch == 0 || ((epoch + 1) % 10) == 0) {
        std::print("epoch = {}, w = [", epoch + 1);
        for (size_t i = 0; i < wSize_; ++i) {
          i &&std::printf(" ");
          std::print("{}", w_[i]);
        }
        std::println("], b = {}, loss = {}", b_, currentloss);
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
  double b_;
  double learningrate_;
  size_t batch_;
  std::mt19937 rng_;
  double dot(const std::vector<double> &a, const std::vector<double> &b) const {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
      sum += a[i] * b[i];
    }
    return sum;
  }
  double computeLoss() const {
    double dataloss = 0.0;
    for (const auto &cur : datas_) {
      double y_hat = dot(w_, cur.x) + b_;
      double err = y_hat - cur.y;
      dataloss += err * err;
    }
    return dataloss / (2.0 * static_cast<double>(nSize_));
  }
  void computeGradient(std::vector<double> &gradw, double &gradb, size_t first,
                       size_t current_batch) const {
    std::vector<double> sum_dw(wSize_, 0.0);
    double sum_db = 0.0;
    for (size_t i = 0; i < current_batch; ++i) {
      const auto &cur = datas_[first + i];
      double y_hat = dot(w_, cur.x) + b_;
      double err = y_hat - cur.y;
      for (size_t j = 0; j < wSize_; ++j) {
        sum_dw[j] += err * cur.x[j];
      }
      sum_db += err;
    }
    for (size_t i = 0; i < wSize_; ++i) {
      sum_dw[i] /= static_cast<double>(current_batch);
    }
    gradw = std::move(sum_dw);
    gradb = sum_db / static_cast<double>(current_batch);
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
  const double b = 0.0;
  MiniBatch linear(std::move(data), wSize, b, learning_rate);
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
