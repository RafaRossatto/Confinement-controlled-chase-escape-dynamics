#include "cell_lattice.h"
#include "utils.h" // for fileExists, logError
#include <fstream>
#include <sstream>
#include <iostream>

CellLattice::CellLattice(int width_, int height_) : width(width_), height(height_) {}

bool CellLattice::loadObstacles(std::vector<Obstacle>& obstacles, int numObstacles, std::string& line) const
{
    if (numObstacles == 0) 
    {
        return true; // nothing to do
    }

    std::string filename = "obstacules_" + std::to_string(numObstacles) + ".txt";
    if (!fileExists(filename)) 
    {
        logError("Error: File " + filename + " does not exist.");
        std::cin.get();
        return false;
    }

    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) 
    {
        logError("Error opening file: " + filename);
        std::cin.get();
        return false;
    }

    int id_counter = 1;
    int x, y;

    while (std::getline(inputFile, line)) 
    {
        std::istringstream lineStream(line);
        if (lineStream >> x >> y) 
        {
            Obstacle new_obstacle("C", id_counter++, x, y);
            obstacles.push_back(new_obstacle);
        }
    }

    inputFile.close();
    return true;
}
