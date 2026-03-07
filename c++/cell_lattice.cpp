#include "cell_lattice.h"
#include <iomanip>
#include <algorithm> 
#include <array> 

/**
 * @brief Constructs a new CellLattice object
 * 
 * @param width Width of the grid
 * @param height Height of the grid
 */
CellLattice::CellLattice(int width, int height) : m_width(width), m_height(height) {
    m_grid.resize(height, std::vector<std::string>(width, "L")); // Initialize with "L" for free
}

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

    int idCounter = 1;
    int x, y;

    while (std::getline(inputFile, line)) 
    {
        std::istringstream lineStream(line);
        if (lineStream >> x >> y) 
        {
            Obstacle newObstacle("C", idCounter++, x, y);
            obstacles.push_back(newObstacle);
        }
    }

    inputFile.close();
    return true;
}

/**
 * @brief Calculates Manhattan distance with toroidal wrapping
 */
double CellLattice::calculateDistance(int x1, int y1, int x2, int y2) const
{
    int dx = std::abs(x2 - x1);
    if (dx > m_width / 2) dx = m_width - dx;

    int dy = std::abs(y2 - y1);
    if (dy > m_height / 2) dy = m_height - dy;

    return dx + dy;
}


/**
 * @brief Places objects on the grid from files
 */
bool CellLattice::placeObjects(std::vector<Obstacle>& obstacles,
    std::vector<Cell>& normalCells,
    std::vector<Cell>& cancerCells,
    int numObstacles, int numNormal, int numCancer,
    std::mt19937& rng, int searchRadiusNormal, int searchRadiusCancer, int run)
{
    // Format strings for file naming
    std::ostringstream ossNormal;
    ossNormal << std::setw(2) << std::setfill('0') << numNormal;
    std::string normalStr = "nC_" + ossNormal.str();

    std::ostringstream ossRun;
    ossRun << std::setw(2) << std::setfill('0') << run;
    std::string runStr = "run_" + ossRun.str();

    std::ostringstream ossObs;
    ossObs << std::setw(2) << std::setfill('0') << numObstacles;
    std::string obsStr = "obs_" + ossObs.str();

    // Load obstacles
    if (numObstacles > 0)
    {
        std::string filePath = "../" + obsStr + "/" + normalStr + "/" + runStr + "/obstacules.txt";
        std::ifstream file(filePath);
        if (!file.is_open()) 
        {
            std::cerr << "Error opening file: " << filePath << std::endl;
            return false;
        }

        int x, y;
        int id = 1;
        while (file >> x >> y) 
        {
            if (x < 0 || x >= m_width || y < 0 || y >= m_height) 
            {
                std::cerr << "Invalid position in file: (" << x << ", " << y << ")\n";
                continue;
            }

            Obstacle obstacle("C", id++, x, y);
            if (!generalOverlap(obstacle, obstacles, normalCells, cancerCells)) 
            {
                obstacles.push_back(obstacle);
                setGridValue(x, y, "C");
            }
        }
        file.close();
    }

    // Load normal cells (chasers)
    {
        std::string filePath = "../" + obsStr + "/" + normalStr + "/" + runStr + "/chasers.txt";
        std::ifstream file(filePath);
        if (!file.is_open()) 
        {
            std::cerr << "Error opening file: " << filePath << std::endl;
            return false;
        }
        
        int x, y;
        int id = 1;
        while (file >> x >> y) 
        {
            if (x < 0 || x >= m_width || y < 0 || y >= m_height) 
            {
                std::cerr << "Invalid position in file: (" << x << ", " << y << ")\n";
                continue;
            }
            
            Cell candidate("N", id++, x, y);
            if (!generalOverlap(candidate, obstacles, normalCells, cancerCells)) 
            {
                normalCells.push_back(candidate);
                setGridValue(x, y, "N");
            }
        }
        file.close();
    }

    // Load cancer cells (escapers)
    {
        std::string filePath = "../" + obsStr + "/" + normalStr + "/" + runStr + "/escapers.txt";
        std::ifstream file(filePath);
        if (!file.is_open()) 
        {
            std::cerr << "Error opening file: " << filePath << std::endl;
            return false;
        }
        
        int x, y;
        int id = 1;
        while (file >> x >> y) 
        {
            if (x < 0 || x >= m_width || y < 0 || y >= m_height) 
            {
                std::cerr << "Invalid position in file: (" << x << ", " << y << ")\n";
                continue;
            }
            
            Cell candidate("O", id++, x, y);
            if (!generalOverlap(candidate, obstacles, normalCells, cancerCells)) 
            {
                cancerCells.push_back(candidate);
                setGridValue(x, y, "O");
            }
        }
        file.close();
    }
    
    return obstacles.size() == static_cast<size_t>(numObstacles) &&
           normalCells.size() == static_cast<size_t>(numNormal) &&
           cancerCells.size() == static_cast<size_t>(numCancer);
}


/**
 * @brief Checks if a position is occupied
 */
bool CellLattice::isOccupied(int x, int y,
    const std::vector<Cell>& normalCells,
    const std::vector<Cell>& cancerCells,
    const std::vector<Obstacle>& obstacles,
    bool checkCancer) const
{
    for (const auto& cell : normalCells) 
    {
    if (cell.getPositionX() == x && cell.getPositionY() == y) 
    {
        return true;
    }
    }

    if (checkCancer) 
    {
        for (const auto& cancer : cancerCells) 
        {
            if (cancer.getPositionX() == x && cancer.getPositionY() == y) 
            {
                return true;
            }
        }
    }

    for (const auto& obs : obstacles) 
    {
        if (obs.getPositionX() == x && obs.getPositionY() == y) 
        {
            return true;
        }
    }
    return false;
}


/**
 * @brief Sets a value in the grid
 */
void CellLattice::setGridValue(int x, int y, const std::string& value) 
{
    if (x >= 0 && x < m_width && y >= 0 && y < m_height) 
    {
        m_grid[y][x] = value;
    } else 
    {
        std::cerr << "[ERROR] Grid access out of bounds in setGridValue: (" << x << "," << y << ")\n";
    }
}

/**
 * @brief Gets a value from the grid
 */
std::string CellLattice::getGridValue(int x, int y) const {
    if (x >= 0 && x < m_width && y >= 0 && y < m_height) {
        return m_grid[y][x];
    } else {
        std::cerr << "[ERROR] Grid access out of bounds in getGridValue: (" << x << "," << y << ")\n";
        return "!";
    }
}


/**
 * @brief Prints the grid to console
 */
void CellLattice::printGrid() const 
{
    for (int y = 0; y < m_height; ++y) 
    {
        for (int x = 0; x < m_width; ++x) 
        {
            std::cout << m_grid[y][x] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "---------------------------\n";
}

/**
 * @brief Counts targets around a position
 */
int CellLattice::countTargetsAround(int x, int y,
    const std::vector<Cell>& agents,
    const std::vector<std::string>& types,
    int searchRadius) const
{
    int count = 0;
    for (const auto& agent : agents) 
    {
        if (std::find(types.begin(), types.end(), agent.getType()) == types.end())
            continue;

        double distance = calculateDistance(x, y, agent.getPositionX(), agent.getPositionY());
        if (distance <= searchRadius + 1e-6)
            ++count;
    }
    return count;
}

/**
 * @brief Moves a cancer cell according to its behavior
 */
void CellLattice::moveCancerCell(Cell& cell,
    std::vector<Cell>& normalCells, std::vector<Cell>& cancerCells,
    std::vector<Obstacle>& obstacles,
    std::mt19937& rng, bool checkCancer, int searchRadius)
{
    int x = cell.getPositionX();
    int y = cell.getPositionY();
    int newX = x, newY = y;

    const std::array<Direction, 4> directions = {NORTH, EAST, SOUTH, WEST};
    std::array<Direction, 4> shuffledDirections = directions;
    std::shuffle(shuffledDirections.begin(), shuffledDirections.end(), rng);

    // 1. Collect all adjacent hunters
    std::vector<std::pair<int, int>> adjacentHunters;

    for (Direction dir : shuffledDirections)
    {
        int adjX = x, adjY = y;
        cell.randomWalk(adjX, adjY, dir);
        for (const auto& hunter : normalCells)
        {
            if (hunter.getPositionX() == adjX && hunter.getPositionY() == adjY)
            {
                adjacentHunters.emplace_back(adjX, adjY);
                break;
            }
        }
    }

    // 2. If there are adjacent hunters → choose random one and flee to free position
    if (!adjacentHunters.empty())
    {
        std::shuffle(adjacentHunters.begin(), adjacentHunters.end(), rng);
        
        // Try to flee to a free direction
        std::vector<Direction> freeDirections;
        for (Direction dir : shuffledDirections)
        {
            int tempX = x, tempY = y;
            cell.randomWalk(tempX, tempY, dir);

            if (!isOccupied(tempX, tempY, normalCells, cancerCells, obstacles, checkCancer))
                freeDirections.push_back(dir);
        }

        if (!freeDirections.empty())
        {
            std::shuffle(freeDirections.begin(), freeDirections.end(), rng);
            Direction fleeDirection = freeDirections.front();
            cell.randomWalk(newX, newY, fleeDirection);

            setGridValue(x, y, "L");
            cell.changePosition(newX, newY);
            setGridValue(newX, newY, "O");
        }
        return;
    }   

    // 3. If no adjacent hunter → density strategy
    std::vector<Cell> allCells;
    allCells.insert(allCells.end(), normalCells.begin(), normalCells.end());
    allCells.insert(allCells.end(), cancerCells.begin(), cancerCells.end());

    int currentDensity = countTargetsAround(x, y, allCells, {"N"}, searchRadius);
    int minDensity = currentDensity;
    std::vector<Direction> bestDirections;

    for (Direction dir : shuffledDirections)
    {
        int tempX = x, tempY = y;
        cell.randomWalk(tempX, tempY, dir);
        if (isOccupied(tempX, tempY, normalCells, cancerCells, obstacles, checkCancer))
            continue;

        int neighborDensity = countTargetsAround(tempX, tempY, allCells, {"N"}, searchRadius);
        if (neighborDensity < minDensity)
        {
            minDensity = neighborDensity;
            bestDirections.clear();
            bestDirections.push_back(dir);
        }
        else if (neighborDensity == minDensity)
        {
            bestDirections.push_back(dir);
        }
    }

    if (!bestDirections.empty())
    {
        std::shuffle(bestDirections.begin(), bestDirections.end(), rng);
        Direction bestDir = bestDirections.front();
        cell.randomWalk(newX, newY, bestDir);
    }
    else
    {
        std::uniform_int_distribution<int> dirDist(0, 3);
        cell.randomWalk(newX, newY, static_cast<Direction>(dirDist(rng)));
    }

    if (!isOccupied(newX, newY, normalCells, cancerCells, obstacles, checkCancer))
    {
        setGridValue(x, y, "L");
        cell.changePosition(newX, newY);
        setGridValue(newX, newY, "O");
    }
}

/**
 * @brief Moves a normal cell according to its behavior
 */
int CellLattice::moveNormalCell(Cell& cell,
    std::vector<Cell>& normalCells, std::vector<Cell>& cancerCells,
    std::vector<Obstacle>& obstacles,
    std::mt19937& rng, bool checkCancer, int searchRadius)
    
{
    int capturedId = -1;
    int x = cell.getPositionX();
    int y = cell.getPositionY();
    int newX = x, newY = y;
    
    // Probability for intelligent movement (should be defined elsewhere)
    const double CT_PROBABILITY = 0.7; // Example value
    std::bernoulli_distribution dist(CT_PROBABILITY);
    bool intelligentMovement = dist(rng);

    const std::array<Direction, 4> directions = {NORTH, EAST, SOUTH, WEST};
    std::array<Direction, 4> shuffledDirections = directions;
    std::shuffle(shuffledDirections.begin(), shuffledDirections.end(), rng);

    if (intelligentMovement)
    {
        // 1. Check for adjacent prey
        std::vector<Direction> adjacentPrey;
        for (Direction dir : shuffledDirections)
        {
            int tempX = x, tempY = y;
            cell.randomWalk(tempX, tempY, dir);
            for (const auto& prey : cancerCells)
            {
                if (prey.getPositionX() == tempX && prey.getPositionY() == tempY)
                {
                    adjacentPrey.push_back(dir);
                    break;
                }
            }
        }

        // 2. If there's adjacent prey → capture random one
        if (!adjacentPrey.empty())
        {
            std::shuffle(adjacentPrey.begin(), adjacentPrey.end(), rng);
            Direction dir = adjacentPrey.front();
            cell.randomWalk(newX, newY, dir);

            // Capture prey at that position
            for (auto it = cancerCells.begin(); it != cancerCells.end(); )
            {
                if (it->getPositionX() == newX && it->getPositionY() == newY)
                {
                    capturedId = it->getId(); 
                    setGridValue(it->getPositionX(), it->getPositionY(), "L");    
                    it = cancerCells.erase(it);
                }
                else
                {
                    ++it;
                }
            }

            setGridValue(x, y, "L");
            cell.changePosition(newX, newY);
            setGridValue(newX, newY, "N");
            return capturedId;
        }

        // 3. If no adjacent prey → follow density strategy
        std::vector<Cell> allCells;
        allCells.insert(allCells.end(), normalCells.begin(), normalCells.end());
        allCells.insert(allCells.end(), cancerCells.begin(), cancerCells.end());

        int currentDensity = countTargetsAround(x, y, allCells, {"O"}, searchRadius);
        int maxDensity = currentDensity;
        std::vector<Direction> bestDirections;

        for (Direction dir : shuffledDirections)
        {
            int tempX = x, tempY = y;
            cell.randomWalk(tempX, tempY, dir);
            if (isOccupied(tempX, tempY, normalCells, cancerCells, obstacles, checkCancer))
                continue;

            int neighborDensity = countTargetsAround(tempX, tempY, allCells, {"O"}, searchRadius);
            if (neighborDensity > maxDensity)
            {
                maxDensity = neighborDensity;
                bestDirections.clear();
                bestDirections.push_back(dir);
            }
            else if (neighborDensity == maxDensity)
            {
                bestDirections.push_back(dir);
            }
        }

        if (!bestDirections.empty())
        {
            std::shuffle(bestDirections.begin(), bestDirections.end(), rng);
            Direction bestDir = bestDirections.front();
            cell.randomWalk(newX, newY, bestDir);
        }
        else
        {
            std::uniform_int_distribution<int> dirDist(0, 3);
            cell.randomWalk(newX, newY, static_cast<Direction>(dirDist(rng)));
        }
    }
    else
    {
        // Completely random movement
        std::uniform_int_distribution<int> dirDist(0, 3);
        cell.randomWalk(newX, newY, static_cast<Direction>(dirDist(rng)));
    }

    if (!isOccupied(newX, newY, normalCells, cancerCells, obstacles, checkCancer))
    {
        setGridValue(x, y, "L");
        cell.changePosition(newX, newY);
        setGridValue(newX, newY, "N");
    }
    return capturedId;
}
