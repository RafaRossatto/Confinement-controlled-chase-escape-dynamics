#ifndef CELL_LATTICE_H
#define CELL_LATTICE_H

#include <vector>
#include <utility> // std::pair
#include "obstacle.h"

class CellLattice {
    private:
        int width;
        int height;
    
    public:
        CellLattice(int width_, int height_);
    
        bool loadObstacles(std::vector<Obstacle>& obstacles, int numObstacles, std::string& line) const;
    };

#endif // CELL_LATTICE_H