#include <array>
#include <iostream>
#include <optional>
#include <ostream>
#include <random>
#include <tuple>
#include <vector>

struct Position {
  int x, y;
  bool operator==(const Position &obj) const {
    return this->x == obj.x && this->y == obj.y;
  }
  friend std::ostream &operator<<(std::ostream &out, const Position &obj);
};
std::ostream &operator<<(std::ostream &out, const Position &obj) {
  out << "(" << obj.x << ", " << obj.y << ")";
  return out;
}

using Grid = std::vector<std::vector<int>>;

enum Action { Up = 0, Down = 1, Left = 2, Right = 3, ActionCount = 4 };

const std::array<Position, ActionCount> Directions{
    {{0, 1}, {1, 0}, {0, -1}, {-1, 0}}};

class LearningAgent {
public:
  LearningAgent(int rows, int cols, double alpha, double gamma, double epsilon)
      : rows_(rows), cols_(cols), alpha_(alpha), gamma_(gamma),
        epsilon_(epsilon), rng_(std::random_device{}()), dist01_(0.0, 1.0),
        actionDist_(0, ActionCount - 1) {
    Q.assign(rows, std::vector<std::array<double, ActionCount>>(
                       cols, {0.0, 0.0, 0.0, 0.0}));
  }

  void train(const Grid &grid, const Position &start, const Position &goal,
             int episodes, int maxStepPerEpisode = 200) {
    for (int i = 0; i < episodes; ++i) {
      Position current{start};
      for (int step = 0; step < maxStepPerEpisode; ++step) {
        int bestAction = selectAction(current);
        Position next;
        double reward;
        bool done = false;
        std::tie(next, reward, done) =
            stepEnvironment(grid, current, goal, bestAction);

        double &qsa = Q[current.x][current.y][bestAction];
        double nextBestAction = bestValueFromAction(next);
        qsa += alpha_ * (reward + gamma_ * nextBestAction - qsa);

        current = next;

        if (done) {
          break;
        }
      }
    }
  }

  std::optional<std::vector<Position>> planGreedy(const Grid &grid,
                                                  const Position &start,
                                                  const Position &goal,
                                                  int maxStep = 200) {
    if (!inBound(start) || !inBound(goal)) {
      return std::nullopt;
    }
    std::vector<Position> Path;
    Position current{start};
    Path.emplace_back(current);
    for (int step = 0; step < maxStep; ++step) {
      if (current == goal) {
        break;
      }
      int bestAction = bestIndexFromAction(current);
      Position next{current.x + Directions[bestAction].x,
                    current.y + Directions[bestAction].y};
      if (!inBound(next) || grid[next.x][next.y] == 1) {
        return std::nullopt;
      }
      Path.emplace_back(next);
      current = next;
      if (current == goal) {
        break;
      }
    }
    if (!(current == goal)) {
      return std::nullopt;
    }
    return Path;
  }

private:
  int rows_, cols_;
  double alpha_, gamma_, epsilon_;
  mutable std::mt19937 rng_;
  mutable std::uniform_real_distribution<double> dist01_;
  mutable std::uniform_int_distribution<int> actionDist_;
  std::vector<std::vector<std::array<double, ActionCount>>> Q;

  bool inBound(const Position &obj) const {
    return obj.x >= 0 && obj.x < rows_ && obj.y >= 0 && obj.y < cols_;
  }

  int selectAction(const Position &currentPos) {
    auto r = dist01_(rng_);
    if (r < epsilon_) {
      return actionDist_(rng_);
    }
    return bestIndexFromAction(currentPos);
  }

  int bestIndexFromAction(const Position &currentPos) {
    int index = 0;
    double bestValue = Q[currentPos.x][currentPos.y][0];
    for (int i = 1; i < ActionCount; ++i) {
      if (Q[currentPos.x][currentPos.y][i] > bestValue) {
        bestValue = Q[currentPos.x][currentPos.y][i];
        index = i;
      }
    }
    return index;
  }

  double bestValueFromAction(const Position &currentPos) {
    double bestValue = Q[currentPos.x][currentPos.y][0];
    for (int i = 1; i < ActionCount; ++i) {
      if (Q[currentPos.x][currentPos.y][i] > bestValue) {
        bestValue = Q[currentPos.x][currentPos.y][i];
      }
    }
    return bestValue;
  }

  std::tuple<Position, double, bool> stepEnvironment(const Grid &grid,
                                                     const Position &currentPos,
                                                     const Position &goal,
                                                     int action) const {
    Position next{currentPos.x + Directions[action].x,
                  currentPos.y + Directions[action].y};
    if (!inBound(next) || grid[next.x][next.y] == 1) {
      return {currentPos, -5, false};
    }
    if (next == goal) {
      return {next, 0, true};
    }
    return {next, -1, false};
  }
};

int main() {
  Grid grid = {
      {0, 0, 0, 0, 0}, {0, 1, 1, 0, 0}, {0, 0, 0, 0, 1},
      {1, 1, 0, 0, 0}, {0, 0, 0, 1, 0},
  };

  Position start{0, 0};
  Position goal{4, 4};

  int rows = static_cast<int>(grid.size());
  int cols = static_cast<int>(grid[0].size());

  LearningAgent agent(rows, cols, 0.1, 0.95, 0.2);
  agent.train(grid, start, goal, 5000, 200);

  auto pathOpt = agent.planGreedy(grid, start, goal);
  if (!pathOpt) {
    std::cout << "Not path found from " << start << " to " << goal << std::endl;
    return 0;
  }

  const auto &path = pathOpt.value();

  std::vector<std::vector<char>> views(rows, std::vector<char>(cols, '.'));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      if (grid[i][j] == 1)
        views[i][j] = '#';
    }
  }

  for (const auto &pos : path) {
    views[pos.x][pos.y] = '*';
  }

  views[start.x][start.y] = 'S';
  views[goal.x][goal.y] = 'G';

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      j &&std::printf(" ");
      std::cout << views[i][j];
    }
    std::printf("\n");
  }

  return 0;
}
