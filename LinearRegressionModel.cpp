#include <print>
#include <random>
#include <vector>

struct Data {
  double x, y;
};

class LinearRegressionModel {
public:
  explicit LinearRegressionModel(std::vector<Data> &&data, double learning_rate,
                                 double w = 0, double b = 0)
      : datas_(data), n(static_cast<int>(datas_.size())),
        learning_rate_(learning_rate), w_(w), b_(b) {}

  void trainLinearModel(int epochs = 2000) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
      double gradw = 0.0, gradb = 0.0;
      computeGradient(gradw, gradb);

      w_ -= learning_rate_ * gradw;
      b_ -= learning_rate_ * gradb;

      if (epoch == 0 || (epoch + 1) % 100 == 0) {
        std::println("epoch = {}, w = {}, b = {}, loss = {}", epoch, w_, b_,
                     computeLoss());
      }
    }
  }

  double predict(double x) { return w_ * x + b_; }

  double getGradW() { return w_; }
  double getGradB() { return b_; }

private:
  std::vector<Data> datas_;
  int n;
  double learning_rate_;
  double w_, b_;

  double computeLoss() {
    double sumloss = 0.0;
    for (const auto &cur : datas_) {
      double y_hat = w_ * cur.x + b_;
      double err = y_hat - cur.y;
      sumloss += err * err;
    }
    return sumloss / (2.0 * static_cast<double>(n));
  }

  void computeGradient(double &w, double &b) {
    double sum_dw = 0.0, sum_db = 0.0;

    for (const auto &cur : datas_) {
      double y_hat = w_ * cur.x + b_;
      double err = y_hat - cur.y;

      sum_dw += err * cur.x;
      sum_db += err;
    }
    w = sum_dw / static_cast<double>(n);
    b = sum_db / static_cast<double>(n);
  }
};

int main() {
  std::vector<Data> data{{0.0, 1.0}, {1.0, 3.1}, {2.0, 4.9},
                         {3.0, 7.2}, {4.0, 9.0}, {5.0, 11.1}};
  std::println("Start training");
  LinearRegressionModel linear(std::move(data), 0.01);

  linear.trainLinearModel();

  std::println("\nTraining finished.\n");
  std::println("Learned parameters:\n  w = {}, b = {}", linear.getGradW(),
               linear.getGradB());

  std::println("Test prections");
  for (double x : {0.0, 1.0, 2.0, 3.0, 10.0}) {
    double y_hat = linear.predict(x);
    std::println("x = {} -> y_hat = {}", x, y_hat);
  }
  return 0;
}
