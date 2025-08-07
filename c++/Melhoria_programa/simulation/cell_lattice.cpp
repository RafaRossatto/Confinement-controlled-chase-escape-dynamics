#include "cell_lattice.h"
#include <iomanip>


CellLattice::CellLattice(int width_, int height_) : width(width_), height(height_) {
    grid.resize(height, std::vector<std::string>(width, "L")); // Inicializa com "L" de livre
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

/*
double CellLattice::calculateDistance(int x1, int y1, int x2, int y2) 
{
    int dx = std::abs(x2 - x1);
    if (dx > WIDTH / 2) dx = WIDTH - dx;

    int dy = std::abs(y2 - y1);
    if (dy > HEIGHT / 2) dy = HEIGHT - dy;

    return std::hypot(dx, dy); // Distância Euclidiana com contorno periódico
}  
*/

// distancia de Manhattan   
double CellLattice::calculateDistance(int x1, int y1, int x2, int y2) 
{
    int dx = std::abs(x2 - x1);
    if (dx > WIDTH / 2) dx = WIDTH - dx;

    int dy = std::abs(y2 - y1);
    if (dy > HEIGHT / 2) dy = HEIGHT - dy;

    return dx + dy;
}


bool CellLattice::placeObjects(std::vector<Obstacle>& obstacles,
    std::vector<Cell>& normalCells,
    std::vector<Cell>& cancerCells,
    int numObstacles, int numNormal, int numCancer,
    std::mt19937& rng, int sr_normal, int sr_cancer, int run)
{

        std::ostringstream oss_nc;
        oss_nc << std::setw(2) << std::setfill('0') << numNormal;
        std::string ncStr = "nC_" + oss_nc.str();  // ex: nC_05

        std::ostringstream oss_run;
        oss_run << std::setw(2) << std::setfill('0') << run;
        std::string runStr = "run_" + oss_run.str();  // ex: run_03

        std::ostringstream oss_obs;
        oss_obs << std::setw(2) << std::setfill('0') << numObstacles;
        std::string ObsStr = "obs_" + oss_obs.str();  // por exemplo, o_15

  
    if (numObstacles > 0)
{
    std::string caminhoArquivo = "../" + ObsStr + "/" + ncStr + "/"  + runStr + "/obstacules.txt";
    std::ifstream arquivo(caminhoArquivo);
    if (!arquivo.is_open()) 
    {
        std::cerr << "Erro ao abrir o arquivo: " << caminhoArquivo << std::endl;
        return false;
    }

    int x, y;
    int id = 1;
    while (arquivo >> x >> y) 
    {
        if (x < 0 || x >= width || y < 0 || y >= height) 
        {
            std::cerr << "Posição inválida no arquivo: (" << x << ", " << y << ")\n";
            continue;
        }

        Obstacle obst("C", id++, x, y);
        if (!generalOverlap(obst, obstacles, normalCells, cancerCells)) 
        {
            obstacles.push_back(obst);
            setGridValue(x, y, "C");
        }
    }

    arquivo.close();
}

    {
        std::string caminhoArquivo = "../" + ObsStr + "/" + ncStr + "/"  + runStr + "/chasers.txt";
        std::ifstream arquivo(caminhoArquivo);
        if (!arquivo.is_open()) 
        {
            std::cerr << "Erro ao abrir o arquivo: " << caminhoArquivo << std::endl;
            return false;
        }
        int x, y;
        int id = 1;
        while (arquivo >> x >> y) 
        {
            if (x < 0 || x >= width || y < 0 || y >= height) 
            {
                std::cerr << "Posição inválida no arquivo: (" << x << ", " << y << ")\n";
                continue;
            }
            Cell candidate("N", id++, x, y);
            if (!generalOverlap(candidate, obstacles, normalCells, cancerCells)) 
            {
            normalCells.push_back(candidate);
            setGridValue(x, y, "N"); // marca na grade
            }
        }

        arquivo.close();
    }

    {

        std::string caminhoArquivo = "../" + ObsStr + "/" + ncStr + "/" + runStr + "/escapers.txt";
        std::ifstream arquivo(caminhoArquivo);
        if (!arquivo.is_open()) 
        {
            std::cerr << "Erro ao abrir o arquivo: " << caminhoArquivo << std::endl;
            return false;
        }
        int x, y;
        int id = 1;
        while (arquivo >> x >> y) 
        {
            if (x < 0 || x >= width || y < 0 || y >= height) 
            {
                std::cerr << "Posição inválida no arquivo: (" << x << ", " << y << ")\n";
                continue;
            }
            Cell candidate("O", id++, x, y);
            if (!generalOverlap(candidate, obstacles, normalCells, cancerCells)) 
            {
            cancerCells.push_back(candidate);
            setGridValue(x, y, "O"); // marca na grade
            }
        }

        arquivo.close();
    }
    
//    std::cerr << "[DEBUG RUN " << run << "] "
  //        << "Esperado: " << numObstacles << " obstáculos, "
    //      << numNormal << " caçadores, "
      //    << numCancer << " presas.\n";

   // std::cerr << "[DEBUG RUN " << run << "] "
     //     << "Obtido: " << obstacles.size() << " obstáculos, "
       //   << normalCells.size() << " caçadores, "
         // << cancerCells.size() << " presas.\n";
//          std:: cin.get();
    
return obstacles.size() == static_cast<size_t>(numObstacles) &&
normalCells.size() == static_cast<size_t>(numNormal) &&
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

void CellLattice::setGridValue(int x, int y, const std::string& value) {
    if (x >= 0 && x < width && y >= 0 && y < height) {
        grid[y][x] = value;
    } else {
        std::cerr << "[ERRO] Tentativa de acesso fora dos limites do grid em setGridValue: (" << x << "," << y << ")\n";
    }
}

std::string CellLattice::getGridValue(int x, int y) const {
    if (x >= 0 && x < width && y >= 0 && y < height) {
        return grid[y][x];
    } else {
        std::cerr << "[ERRO] Tentativa de acesso fora dos limites do grid em getGridValue: (" << x << "," << y << ")\n";
        return "!";
    }
}

void CellLattice::printGrid() const {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            std::cout << grid[y][x] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "---------------------------\n";
}
