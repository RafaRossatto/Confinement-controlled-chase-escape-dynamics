#include "cell_lattice.h"

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

double CellLattice::calculateDistance(int x1, int y1, int x2, int y2) 
{
    int dx = std::abs(x2 - x1);
    if (dx > WIDTH / 2) dx = WIDTH - dx;

    int dy = std::abs(y2 - y1);
    if (dy > HEIGHT / 2) dy = HEIGHT - dy;

    return dx + dy; // Manhattan distance with periodic boundaries
}

bool CellLattice::placeObjects(std::vector<Obstacle>& obstacles,
                        std::vector<Cell>& normalCells,
                      std::vector<Cell>& cancerCells,
                      int numObstacles, int numNormal, int numCancer,
                      std::mt19937& rng,int sr_normal, int sr_cancer)
{
    const int maxTries = 100;

    // Place normal cells
    int triesNormal = 0;
    while (triesNormal < maxTries && normalCells.size() < static_cast<size_t>(numNormal)) {
        std::uniform_int_distribution<int> distX(0, width - 1);
        std::uniform_int_distribution<int> distY(0, height - 1);

        int x = distX(rng);
        int y = distY(rng);
        int id = normalCells.size() + 1;

        Cell candidate("N", id, x, y);
        candidate.setSearchRadius(sr_normal);

        if (!generalOverlap(candidate, obstacles, normalCells, cancerCells)) {
            normalCells.push_back(candidate);
            triesNormal = 0;
        } else {
            ++triesNormal;
        }
    }

    // Place cancer cells
    int triesCancer = 0;
    while (triesCancer < maxTries && cancerCells.size() < static_cast<size_t>(numCancer)) {
        std::uniform_int_distribution<int> distX(0, width - 1);
        std::uniform_int_distribution<int> distY(0, height - 1);

        int x = distX(rng);
        int y = distY(rng);
        int id = cancerCells.size() + 1;

        Cell candidate("O", id, x, y);
        candidate.setSearchRadius(sr_cancer);

        if (!generalOverlap(candidate, obstacles, normalCells, cancerCells)) {
            cancerCells.push_back(candidate);
            triesCancer = 0;
        } else {
            ++triesCancer;
        }
    }

    return normalCells.size() == static_cast<size_t>(numNormal) &&
           cancerCells.size() == static_cast<size_t>(numCancer);
}

bool CellLattice::isOccupied(int x, int y,
                             const std::vector<Cell>& normalCells,
                             const std::vector<Cell>& cancerCells,
                             const std::vector<Obstacle>& obstacles,
                             bool checkCancer) const
{
    for (const auto& cell : normalCells) {
        if (cell.getCoordenadaX() == x && cell.getCoordenadaY() == y) {
            return true;
        }
    }

    if (checkCancer) {
        for (const auto& cancer : cancerCells) {
            if (cancer.getCoordenadaX() == x && cancer.getCoordenadaY() == y) {
                return true;
            }
        }
    }

    for (const auto& obs : obstacles) {
        if (obs.getCoordenadaX() == x && obs.getCoordenadaY() == y) {
            return true;
        }
    }

    return false;
}