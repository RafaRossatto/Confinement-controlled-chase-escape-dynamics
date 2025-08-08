#pragma once
#ifndef CELL_LATTICE_H
#define CELL_LATTICE_H
#include "cell_lattice.h"
#include "utils.h" // for fileExists, logError
#include "cell.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <random>            // para std::mt19937
#include "obstacle.h"        // para classe Obstacle

class Cell;

class CellLattice 
{
    private:
        int width;
        int height;
        std::vector<std::vector<std::string>> grid;
    
    public:
        CellLattice(int width_, int height_);
    
        bool loadObstacles(std::vector<Obstacle>& obstacles, int numObstacles, std::string& line) const;
        static double calculateDistance(int x1, int y1, int x2, int y2);

        void printGrid() const;

        void setGridValue(int x, int y, const std::string& value);
        std::string getGridValue(int x, int y) const;
        
        int getWidth() const { return width; }
        int getHeight() const { return height; }

    bool placeObjects(std::vector<Obstacle>& obstacles,
                  std::vector<Cell>& normalCells,
                  std::vector<Cell>& cancerCells,
                  int numObstacles, int numNormal, int numCancer,
                  std::mt19937& rng,
                  int sr_normal, int sr_cancer,int run);
    
    template<typename T, typename U>
    bool overlapsWithList(const T& obj, const std::vector<U>& list) const;

    template<typename T>
    bool generalOverlap(const T& obj,
                        const std::vector<Obstacle>& obstacles,
                        const std::vector<Cell>& normalCells,
                        const std::vector<Cell>& cancerCells) const;
       

    bool isOccupied(int x, int y,
                    const std::vector<Cell>& normalCells,
                    const std::vector<Cell>& cancerCells,
                    const std::vector<Obstacle>& obstacles,
                    bool checkCancer) const;
};


template<typename T, typename U>
bool CellLattice::overlapsWithList(const T& obj, const std::vector<U>& list) const {
    for (const auto& item : list) {
        if (obj.getCoordenadaX() == item.getCoordenadaX() &&
            obj.getCoordenadaY() == item.getCoordenadaY()) {
            return true;
        }
    }
    return false;
}

template<typename T>
bool CellLattice::generalOverlap(const T& obj,
                                 const std::vector<Obstacle>& obstacles,
                                 const std::vector<Cell>& normalCells,
                                 const std::vector<Cell>& cancerCells) const {
    return overlapsWithList(obj, obstacles) ||
           overlapsWithList(obj, normalCells) ||
           overlapsWithList(obj, cancerCells);
}


#endif // CELL_LATTICE_H