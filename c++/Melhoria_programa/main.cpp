#include "obstacle.h"
#include "config.h"
#include "utils.h"
#include "cell.h"
#include "cell_lattice.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include<utility>
#include <tuple>
#include <algorithm> 
#include <array> 
std::random_device rd;
unsigned int GLOBAL_SEED = rd();
double CTPROBABILITY = 1.00; // probabilidade de capturar ou procurar comida
double CCPROBABILITY = 1.00; // probabilidade de escapar ou procurar comida

void setCTPROBABILITY(double newValue)
{
    CTPROBABILITY = newValue;
}

void setCCPROBABILITY(double newValue)
{
    CCPROBABILITY = newValue;
}

void moverCelulaRuim(CellLattice& lattice, Cell& cell,
                    std::vector<Cell>& cT, std::vector<Cell>& cC,
                    std::vector<Obstacle>& obstacles,
                    std::mt19937& rng, bool checkcC) 
{    
    int x = cell.getCoordenadaX();
    int y = cell.getCoordenadaY();
    int newX = x, newY = y;

    std::vector<Cell> celulas;
    celulas.insert(celulas.end(), cT.begin(), cT.end());
    celulas.insert(celulas.end(), cC.begin(), cC.end());
    auto alvos = cell.findNearestTarget(celulas, lattice, {"N"});

    int menorDist = std::numeric_limits<int>::max();
    int alvoX = 0, alvoY = 0;
    bool find = false;

    // Passo 1: Encontra a "N" mais próxima
    for (const auto& [ax, ay] : alvos) 
    {
        for (const auto& agente : celulas) 
        {
            if (agente.getCoordenadaX() == ax && agente.getCoordenadaY() == ay) 
            {
                int dist = lattice.calculateDistance(x, y, ax, ay);
                if (dist < menorDist) 
                {
                    menorDist = dist;
                    alvoX = ax; alvoY = ay;
                    find = true;
                }
            }
        }
    }

    // Passo 2: Foge ou move aleatoriamente
    if (find) 
    {
        const std::array<Direction, 4> direcoes = {NORTH, EAST, SOUTH, WEST};
        std::array<Direction, 4> direcoesEmbaralhadas = direcoes;
        std::shuffle(direcoesEmbaralhadas.begin(), direcoesEmbaralhadas.end(), rng);

        double maiorDistancia = -1.0;
        Direction melhorDirecao = NORTH;
        int tempX_final = x, tempY_final = y;

        for (Direction dir : direcoesEmbaralhadas) 
        {
            int tempX = x;
            int tempY = y;
            cell.randonWalk(tempX, tempY, dir);
            double dist = lattice.calculateDistance(tempX, tempY, alvoX, alvoY);

            if (dist > maiorDistancia) 
            {
                maiorDistancia = dist;
                melhorDirecao = dir;
                tempX_final = tempX;
                tempY_final = tempY;
            }
        }
        newX = tempX_final;
        newY = tempY_final;
    } 
    else 
    {
        // Movimento aleatório (sem alvos)
        std::uniform_int_distribution<int> dir(0, 3);
        cell.randonWalk(newX, newY, static_cast<Direction>(dir(rng)));
        logInfo("[O] Random movement (no targets found)");
    }

    // Verifica colisão
    if (!lattice.isOccupied(newX, newY, cT, cC, obstacles, checkcC)) 
    {
        cell.changePosition(newX, newY);
    } 
}

void moverCelulaBoa(CellLattice& lattice, Cell& cell,
                   std::vector<Cell>& cT, std::vector<Cell>& cC,
                   std::vector<Obstacle>& obstacles,
                   std::mt19937& rng, bool checkcC)
{    
    int x = cell.getCoordenadaX();
    int y = cell.getCoordenadaY();
    int newX = x, newY = y;

    std::bernoulli_distribution d(CTPROBABILITY);
    bool movimentoInteligente = d(rng);

    if (movimentoInteligente) 
    {
        std::vector<Cell> cells;
        cells.insert(cells.end(), cT.begin(), cT.end());
        cells.insert(cells.end(), cC.begin(), cC.end());

        // Busca alvos do tipo "O" (inimigas) e "N" (companheiras)
        auto targets = cell.findNearestTarget(cells, lattice, {"O", "N"});

        int menorDist_O = std::numeric_limits<int>::max(); // Distância para "O" mais próxima
        int menorDist_N = std::numeric_limits<int>::max(); // Distância para "N" mais próxima
        int alvoX_O = 0, alvoY_O = 0;
        int alvoX_N = 0, alvoY_N = 0;

        // Passo 1: Identifica alvos mais próximos de cada tipo
        for (const auto& [ax, ay] : targets) 
        {
            for (const auto& agent : cells) 
            {
                if (agent.getCoordenadaX() == ax && agent.getCoordenadaY() == ay) 
                {
                    int dist = lattice.calculateDistance(x, y, ax, ay);    
                    if (agent.getTipo() == "O" && dist < menorDist_O) 
                    {
                        menorDist_O = dist;
                        alvoX_O = ax; alvoY_O = ay;
                    } 
                    else if (agent.getTipo() == "N" && dist < menorDist_N) 
                    {
                        menorDist_N = dist;
                        alvoX_N = ax; alvoY_N = ay;
                    }
                }
            }
        }

        // Passo 2: Toma decisão (prioriza perseguir "O" se existir)
        if (menorDist_O < menorDist_N) 
        {
            // Persegue "O" mais próxima
            const std::array<Direction, 4> direcoes = {NORTH, EAST, SOUTH, WEST};
            std::array<Direction, 4> direcoesEmbaralhadas = direcoes;
            std::shuffle(direcoesEmbaralhadas.begin(), direcoesEmbaralhadas.end(), rng);

            double menorDistancia = std::numeric_limits<double>::max();
            Direction melhorDirecao = NORTH;
            int tempX_final = x, tempY_final = y;

            for (Direction dir : direcoesEmbaralhadas) 
            {
                int tempX = x;
                int tempY = y;
                cell.randonWalk(tempX, tempY, dir);
                double dist = lattice.calculateDistance(tempX, tempY, alvoX_O, alvoY_O);

                if (dist < menorDistancia) 
                {
                    menorDistancia = dist;
                    melhorDirecao = dir;
                    tempX_final = tempX;
                    tempY_final = tempY;
                }
            }
            newX = tempX_final;
            newY = tempY_final;

            // Verifica se alcançou a célula O (captura)
            if (newX == alvoX_O && newY == alvoY_O) 
            {
                // Remove a célula O capturada do vetor cC
                for (auto it = cC.begin(); it != cC.end(); ) 
                {
                    if (it->getCoordenadaX() == alvoX_O && it->getCoordenadaY() == alvoY_O) 
                    {
                        it = cC.erase(it);
                        break;
                    } 
                    else 
                    {
                        ++it;
                    }
                }
            }
        } 
        else if (menorDist_N != std::numeric_limits<int>::max()) 
        {
            // Foge da "N" mais próxima
            const std::array<Direction, 4> direcoes = {NORTH, EAST, SOUTH, WEST};
            std::array<Direction, 4> direcoesEmbaralhadas = direcoes;
            std::shuffle(direcoesEmbaralhadas.begin(), direcoesEmbaralhadas.end(), rng);
            double maiorDistancia = -1.0;
            Direction melhorDirecao = NORTH;
            int tempX_final = x, tempY_final = y;

            for (Direction dir : direcoesEmbaralhadas) 
            {
                int tempX = x;
                int tempY = y;
                cell.randonWalk(tempX, tempY, dir);
                double dist = lattice.calculateDistance(tempX, tempY, alvoX_N, alvoY_N);

                if (dist > maiorDistancia) 
                {
                    maiorDistancia = dist;
                    melhorDirecao = dir;
                    tempX_final = tempX;
                    tempY_final = tempY;
                }
            }
            newX = tempX_final;
            newY = tempY_final;
        } 
        else 
        {
            std::uniform_int_distribution<int> dir(0, 3);
            cell.randonWalk(newX, newY, static_cast<Direction>(dir(rng)));
            logInfo("[N] Random movement (no targets found)");
        }
    } 
    else 
    {
        // Movimento aleatório (quando não é inteligente)
        std::uniform_int_distribution<int> dir(0, 3);
        cell.randonWalk(newX, newY, static_cast<Direction>(dir(rng)));
    }

    // Verifica colisão com obstáculos ou outras células
    if (!lattice.isOccupied(newX, newY, cT, cC, obstacles, checkcC)) 
    {
        cell.changePosition(newX, newY);
    } 
}







/*
void executarRodadas(CellLattice& lattice,int count, int numCT, int numcC, int numPoint,
    double ncNoise_ct, double ncNoise_cc,
    const std::string& fileName, const std::vector<Obstacle>& point,int sr_normal,int sr_cancer)
{
    //int remaining_cC;
    std::ofstream outputFile(fileName);
    if (!outputFile.is_open()) 
    {
        logError("Failed to open file: " + fileName);
        std::cin.get();
        return;
    }
    outputFile << "run,steps,escapers,seed\n";

    // Sorteia uma run para salvar evolução temporal
//int runSorteada = std::uniform_int_distribution<int>(1, count)(std::mt19937(GLOBAL_SEED + numPoint + numcC));
	std::mt19937 rng_run_selector(GLOBAL_SEED + numPoint + numcC + 99999);
    std::uniform_int_distribution<int> dist_run(1, count);
    //int runSorteada = dist_run(rng_run_selector);

    #pragma omp parallel for schedule(dynamic)
    for (int run = 1; run <= count; ++run) 
    {
        unsigned int seed_run = GLOBAL_SEED + 10* sr_cancer + 10 * sr_normal + run + 1000 * numcC + 100000 * numPoint;
        std::mt19937 rng_local(seed_run);
        std::vector<Cell> cT_local;
        std::vector<Cell> cC_local;
        std::vector<Obstacle> point_local = point;

        if (!lattice.placeObjects(point_local, cT_local, cC_local, numPoint, numCT, numcC, rng_local,sr_normal,sr_cancer)) 
        {
            #pragma omp critical
            logError("Failed to place the objects in the execution " + std::to_string(run));
            continue;
        }

        std::vector<int> presasPorPasso;
        const int intervalo = 100000;

        int steps = 1;
        bool verificacC;
        while (steps < 50000) 
        {
            if (!cT_local.empty()) {
                for (int i = cT_local.size() - 1; i >= 0; --i) {
                    verificacC = false;
                    moverCelulaBoa(lattice,cT_local[i], cT_local, cC_local, point_local, rng_local,verificacC);
                }
            }
            if (!cC_local.empty()) {
                for (int i = cC_local.size() - 1; i >= 0; --i) {
                    verificacC = true;
                    moverCelulaRuim(lattice,cC_local[i], cT_local, cC_local, point_local, rng_local,verificacC);
                }
            }
            if (cC_local.empty()) break;

            
            if (steps % intervalo == 0)
	        {
                presasPorPasso.push_back(static_cast<int>(cC_local.size()));
            }                

            steps++;
        }  	
	#pragma omp critical
	
    {
    	outputFile << run << "," << steps << "," << cC_local.size() << "," << seed_run << "\n";
        }
        
    }
    outputFile.close();
    logInfo("Results saved to file: " + fileName);
}
*/


void executarRodadas(CellLattice& lattice, int count, int numCT, int numcC, int numPoint,
    double ncNoise_ct, double ncNoise_cc,
    const std::string& fileName, const std::vector<Obstacle>& point,
    int sr_normal, int sr_cancer)
{
std::ofstream outputFile(fileName);
if (!outputFile.is_open()) {
logError("Failed to open file: " + fileName);
std::cin.get();
return;
}
outputFile << "run,steps,escapers,seed\n";

std::mt19937 rng_run_selector(GLOBAL_SEED + numPoint + numcC + 99999);
std::uniform_int_distribution<int> dist_run(1, count);

#pragma omp parallel for schedule(dynamic)
for (int run = 1; run <= count; ++run) 
{
unsigned int seed_run = GLOBAL_SEED + 10 * sr_cancer + 10 * sr_normal + run + 1000 * numcC + 100000 * numPoint;
std::mt19937 rng_local(seed_run);
std::vector<Cell> cT_local;
std::vector<Cell> cC_local;
std::vector<Obstacle> point_local = point;

if (!lattice.placeObjects(point_local, cT_local, cC_local, numPoint, numCT, numcC, rng_local, sr_normal, sr_cancer)) {
#pragma omp critical
logError("Failed to place the objects in the execution " + std::to_string(run));
continue;
}

std::uniform_int_distribution<int> distX(0, lattice.getWidth() - 1);
std::uniform_int_distribution<int> distY(0, lattice.getHeight() - 1);

std::vector<int> presasPorPasso;
const int intervalo = 100000;
int steps = 1;
bool verificacC;

while (steps < 1000) 
{
if (!cT_local.empty()) {
for (int i = cT_local.size() - 1; i >= 0; --i) {
  verificacC = false;
  moverCelulaBoa(lattice, cT_local[i], cT_local, cC_local, point_local, rng_local, verificacC);
}
}

if (!cC_local.empty()) {
for (int i = cC_local.size() - 1; i >= 0; --i) {
  verificacC = true;
  moverCelulaRuim(lattice, cC_local[i], cT_local, cC_local, point_local, rng_local, verificacC);
}
}

// Descreve o que existe em cada posição
#pragma omp critical
{
std::cout << "[RUN " << run << " - STEP " << steps << "]\n";
for (int y = 0; y < lattice.getHeight(); ++y) {
  for (int x = 0; x < lattice.getWidth(); ++x) {
      std::string valor = lattice.getGridValue(x, y);
      if (valor != "L") {
          std::cout << "Posição (" << x << "," << y << ") tem: " << valor << "\n";
      }
  }
}
}

if (cC_local.empty()) break;

if (steps % intervalo == 0) {
presasPorPasso.push_back(static_cast<int>(cC_local.size()));
}

steps++;
std:: cin.get();
}

#pragma omp critical
{
outputFile << run << "," << steps << "," << cC_local.size() << "," << seed_run << "\n";
}
}

outputFile.close();
logInfo("Results saved to file: " + fileName);
}





int main() 
{
    int count, numCT, numcC, numPoint;

    std::vector<int> numcT_values = {5};
    std::vector<int> numcC_values = {5};
    std::vector<int> numPoint_values = {40};
    std::vector<double> numNoise_ct = {0.99};
    std::vector<double> numNoise_cc = {0.99};

    std::vector<Cell> cT;
    std::vector<Cell> cC;
    std::vector<Obstacle> point;
    std::string line;
    CellLattice lattice(WIDTH,HEIGHT);
    int sr_normal = 100;
    int sr_cancer = 100;

    count = 1;
    for (int ncT_value : numcT_values) 
    {
        numCT = ncT_value;
        for (double ncNoise_ct : numNoise_ct) 
        {
            setCTPROBABILITY(ncNoise_ct);
            for (double ncNoise_cc : numNoise_cc) 
            {
                setCCPROBABILITY(ncNoise_cc);

                for (int ncC_value : numcC_values) 
                {
                    numcC = ncC_value;

                    for (int np_value : numPoint_values) 
                    {
                        numPoint = np_value;
                        cT.clear();
                        cC.clear();
                        point.clear();

                        //if (!lattice.loadObstacles(point, numPoint, line)) 
                        //{
                         //   logError("Failed to load obstacles");
                          //  return 1;
                        //}

                        std::string fileName = gerarNomeArquivo(numCT, numcC, numPoint, ncNoise_cc, ncNoise_ct, sr_normal, sr_cancer);
                        executarRodadas(lattice, count, numCT, numcC, numPoint, ncNoise_ct, ncNoise_cc, fileName, point, sr_normal, sr_cancer);
                    }
                }
            }
        }
    }
    return 0;
}
