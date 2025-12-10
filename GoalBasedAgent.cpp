#include <algorithm>
#include <array>
#include <iostream>
#include <optional>
#include <ostream>
#include <queue>
#include <vector>

struct Position {
  int row, col;
  bool operator==(const Position &obj) const {
    return this->row == obj.row && this->col == obj.col;
  }
  friend std::ostream &operator<<(std::ostream &out, const Position &obj);
};
std::ostream &operator<<(std::ostream &out, const Position &obj) {
  out << "(" << obj.row << ", " << obj.col << ")";
  return out;
}

using Grid = std::vector<std::vector<int>>;

class GoalBasedAgent {
public:
  std::optional<std::vector<Position>>
  Plan(const Grid &grid, const Position &start, const Position &goal) {
    if (grid.empty() || start == goal)
      return std::nullopt;
    rows = grid.size();
    cols = grid[0].size();
    std::queue<Position> que{};
    std::vector<std::vector<bool>> Visited(rows,
                                           std::vector<bool>(cols, false));
    std::vector<std::vector<Position>> Parent(
        rows, std::vector<Position>(cols, Position{}));
    bool HasPath = false;
    Visited[start.row][start.col] = true;
    que.emplace(start);
    while (!que.empty()) {
      const auto &current = que.front();
      que.pop();
      if (current == goal) {
        HasPath = true;
        break;
      }
      for (const auto &dir : directions) {
        int xx = current.row + dir.row;
        int yy = current.col + dir.col;
        Position next{xx, yy};
        if (!InBound(next))
          continue;
        if (Visited[xx][yy])
          continue;
        if (grid[xx][yy] == 1)
          continue;
        Visited[xx][yy] = true;
        Parent[xx][yy] = current;
        que.emplace(next);
      }
    }

    if (!HasPath)
      return std::nullopt;

    std::vector<Position> ans{};
    auto current = goal;
    while (!(current == start)) {
      ans.emplace_back(current);
      current = Parent[current.row][current.col];
    }
    ans.emplace_back(start);
    std::reverse(ans.begin(), ans.end());

    return ans;
  }

private:
  int rows{}, cols{};
  bool InBound(const Position &pos) {
    return pos.row >= 0 && pos.row < rows && pos.col >= 0 && pos.col < cols;
  }
  const std::array<Position, 4> directions = {
      {{0, 1}, {1, 0}, {0, -1}, {-1, 0}}};
};

int main() {
  Grid grid = {
      {0, 0, 0, 0, 0}, {0, 1, 1, 0, 0}, {0, 0, 0, 0, 1},
      {1, 1, 0, 0, 0}, {0, 0, 0, 1, 0},
  };

  Position start{0, 0};
  Position goal{4, 4};

  GoalBasedAgent agent{};
  auto PathOpt = agent.Plan(grid, start, goal);
  if (!PathOpt) {
    std::cout << "No Path found from " << start << " to " << goal << "\n";
    return 0;
  }

  const auto &Path = PathOpt.value();

  std::cout << "Path from " << start << " to " << goal << "\n";
  for (const auto &pos : Path) {
    std::cout << pos << " ";
  }
  std::cout << std::endl;

  int row = grid.size(), col = grid[0].size();
  std::vector<std::vector<char>> views(row, std::vector<char>(col, '.'));
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      if (grid[i][j] == 1)
        views[i][j] = '#';
    }
  }
  for (const auto &pos : Path) {
    views[pos.row][pos.col] = '*';
  }

  views[start.row][start.col] = 'S';
  views[goal.row][goal.col] = 'G';

  for (const auto &view : views) {
    for (const auto &ch : view) {
      std::cout << ch << " ";
    }
    std::cout << "\n";
  }

  return 0;
}
