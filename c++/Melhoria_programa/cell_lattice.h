#ifndef CELL_LATTICE_H
#define CELL_LATTICE_H

#include <vector>
#include <utility> // std::pair
#include "obstacle.h"

class CellLattice {
private:
    int width;
    int height;

    // Armazena posições de obstáculos
    std::vector<std::pair<int, int>> obstacles;

public:
    CellLattice(int width, int height);

    int getWidth() const;
    int getHeight() const;

    void addObstacle(int x, int y);
    bool isObstacle(int x, int y) const;
    void clearObstacles();
};

#endif // CELL_LATTICE_H