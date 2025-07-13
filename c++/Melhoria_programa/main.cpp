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
#include <iomanip>
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



int contarAlvosAoRedor(int x, int y,
    const std::vector<Cell>& agentes,
    const CellLattice& lattice,
    const std::vector<std::string>& tipos,
    int sigma = 2)
{
int count = 0;
for (const auto& a : agentes) {
if (std::find(tipos.begin(), tipos.end(), a.getTipo()) == tipos.end())
continue;

double dist = lattice.calculateDistance(x, y, a.getCoordenadaX(), a.getCoordenadaY());
if (dist <= sigma + 1e-6)
++count;
}
return count;
}


void moverCelulaRuim(CellLattice& lattice, Cell& cell,
    std::vector<Cell>& cT, std::vector<Cell>& cC,
    std::vector<Obstacle>& obstacles,
    std::mt19937& rng, bool checkcC, int SR)
{
    int x = cell.getCoordenadaX();
    int y = cell.getCoordenadaY();
    int newX = x, newY = y;

    const std::array<Direction, 4> direcoes = {NORTH, EAST, SOUTH, WEST};
    std::array<Direction, 4> direcoesEmbaralhadas = direcoes;
    std::shuffle(direcoesEmbaralhadas.begin(), direcoesEmbaralhadas.end(), rng);

// 1. Coleta todos os caçadores adjacentes
std::vector<std::pair<int, int>> cacadoresAdjacentes;

for (Direction dir : direcoesEmbaralhadas)
{
    int adjX = x, adjY = y;
    cell.randonWalk(adjX, adjY, dir);
    for (const auto& hunter : cT)
    {
        if (hunter.getCoordenadaX() == adjX && hunter.getCoordenadaY() == adjY)
        {
            cacadoresAdjacentes.emplace_back(adjX, adjY);
            break;
        }
    }
}

// 2. Se houver caçadores adjacentes → escolher aleatoriamente um e fugir para posição livre qualquer
if (!cacadoresAdjacentes.empty())
{
    // Não usamos diretamente o caçador escolhido, só sorteamos para satisfazer a regra
    std::shuffle(cacadoresAdjacentes.begin(), cacadoresAdjacentes.end(), rng);
    auto [cacadorX, cacadorY] = cacadoresAdjacentes.front();

    // Agora tentamos fugir para uma direção livre
    std::vector<Direction> direcoesLivres;
    for (Direction dir : direcoesEmbaralhadas)
    {
        int tempX = x, tempY = y;
        cell.randonWalk(tempX, tempY, dir);

        if (!lattice.isOccupied(tempX, tempY, cT, cC, obstacles, checkcC))
            direcoesLivres.push_back(dir);
    }

    if (!direcoesLivres.empty())
    {
        std::shuffle(direcoesLivres.begin(), direcoesLivres.end(), rng);
        Direction fuga = direcoesLivres.front();
        cell.randonWalk(newX, newY, fuga);

        lattice.setGridValue(x, y, "L");
        cell.changePosition(newX, newY);
        lattice.setGridValue(newX, newY, "O");
    }

    return; // mesmo que fique parado
}

    // 3. Se não há caçador adjacente → estratégia de densidade
    std::vector<Cell> celulas;
    celulas.insert(celulas.end(), cT.begin(), cT.end());
    celulas.insert(celulas.end(), cC.begin(), cC.end());

    int densidadeAtual = contarAlvosAoRedor(x, y, celulas, lattice, {"N"}, SR);
    int menorDensidade = densidadeAtual;
    std::vector<Direction> melhoresDirecoes;

    for (Direction dir : direcoesEmbaralhadas)
    {
        int tempX = x, tempY = y;
        cell.randonWalk(tempX, tempY, dir);
        if (lattice.isOccupied(tempX, tempY, cT, cC, obstacles, checkcC))
            continue;

        int densidadeVizinha = contarAlvosAoRedor(tempX, tempY, celulas, lattice, {"N"}, SR);
        if (densidadeVizinha < menorDensidade)
        {
            menorDensidade = densidadeVizinha;
            melhoresDirecoes.clear();
            melhoresDirecoes.push_back(dir);
        }
        else if (densidadeVizinha == menorDensidade)
        {
            melhoresDirecoes.push_back(dir);
        }
    }

    if (!melhoresDirecoes.empty())
    {
        std::shuffle(melhoresDirecoes.begin(), melhoresDirecoes.end(), rng);
        Direction melhor = melhoresDirecoes.front();
        cell.randonWalk(newX, newY, melhor);
    }
    else
    {
        std::uniform_int_distribution<int> dir(0, 3);
        cell.randonWalk(newX, newY, static_cast<Direction>(dir(rng)));
    }

    if (!lattice.isOccupied(newX, newY, cT, cC, obstacles, checkcC))
    {
        lattice.setGridValue(x, y, "L");
        cell.changePosition(newX, newY);
        lattice.setGridValue(newX, newY, "O");
    }
}





void moverCelulaBoa(CellLattice& lattice, Cell& cell,
    std::vector<Cell>& cT, std::vector<Cell>& cC,
    std::vector<Obstacle>& obstacles,
    std::mt19937& rng, bool checkcC, int SR)
{
    int x = cell.getCoordenadaX();
    int y = cell.getCoordenadaY();
    int newX = x, newY = y;
    std::bernoulli_distribution d(CTPROBABILITY);
    bool movimentoInteligente = d(rng);

    const std::array<Direction, 4> direcoes = {NORTH, EAST, SOUTH, WEST};
    std::array<Direction, 4> direcoesEmbaralhadas = direcoes;
    std::shuffle(direcoesEmbaralhadas.begin(), direcoesEmbaralhadas.end(), rng);

    if (movimentoInteligente)
    {
        // 1. Verifica se há presa adjacente
        std::vector<Direction> presasAdjacentes;
        for (Direction dir : direcoesEmbaralhadas)
        {
            int tempX = x, tempY = y;
            cell.randonWalk(tempX, tempY, dir);
            for (const auto& presa : cC)
            {
                if (presa.getCoordenadaX() == tempX && presa.getCoordenadaY() == tempY)
                {
                    presasAdjacentes.push_back(dir);
                    break;
                }
            }
        }

        // 2. Se houver presa adjacente → captura uma aleatória
        if (!presasAdjacentes.empty())
        {
            std::shuffle(presasAdjacentes.begin(), presasAdjacentes.end(), rng);
            Direction dir = presasAdjacentes.front();
            cell.randonWalk(newX, newY, dir);

            // Captura a presa naquela posição
            for (auto it = cC.begin(); it != cC.end(); )
            {
                if (it->getCoordenadaX() == newX && it->getCoordenadaY() == newY)
                {
                    lattice.setGridValue(it->getCoordenadaX(), it->getCoordenadaY(), "L");    
                    it = cC.erase(it);
                }
                else
                {
                    ++it;
                }
            }

            lattice.setGridValue(x, y, "L");
            cell.changePosition(newX, newY);
            lattice.setGridValue(newX, newY, "N");
            return;
        }

        // 3. Caso não haja presa adjacente → seguir a densidade
        std::vector<Cell> celulas;
        celulas.insert(celulas.end(), cT.begin(), cT.end());
        celulas.insert(celulas.end(), cC.begin(), cC.end());

        int densidadeAtual = contarAlvosAoRedor(x, y, celulas, lattice, {"O"}, SR);
        int maiorDensidade = densidadeAtual;
        std::vector<Direction> melhoresDirecoes;

        for (Direction dir : direcoesEmbaralhadas)
        {
            int tempX = x, tempY = y;
            cell.randonWalk(tempX, tempY, dir);
            if (lattice.isOccupied(tempX, tempY, cT, cC, obstacles, checkcC))
                continue;

            int densidadeVizinha = contarAlvosAoRedor(tempX, tempY, celulas, lattice, {"O"}, SR);
            if (densidadeVizinha > maiorDensidade)
            {
                maiorDensidade = densidadeVizinha;
                melhoresDirecoes.clear();
                melhoresDirecoes.push_back(dir);
            }
            else if (densidadeVizinha == maiorDensidade)
            {
                melhoresDirecoes.push_back(dir);
            }
        }

        if (!melhoresDirecoes.empty())
        {
            std::shuffle(melhoresDirecoes.begin(), melhoresDirecoes.end(), rng);
            Direction melhor = melhoresDirecoes.front();
            cell.randonWalk(newX, newY, melhor);
        }
        else
        {
            std::uniform_int_distribution<int> dir(0, 3);
            cell.randonWalk(newX, newY, static_cast<Direction>(dir(rng)));
        }
    }
    else
    {
        // Movimento totalmente aleatório
        std::uniform_int_distribution<int> dir(0, 3);
        cell.randonWalk(newX, newY, static_cast<Direction>(dir(rng)));
    }

    if (!lattice.isOccupied(newX, newY, cT, cC, obstacles, checkcC))
    {
        lattice.setGridValue(x, y, "L");
        cell.changePosition(newX, newY);
        lattice.setGridValue(newX, newY, "N");
    }
}

void executarRodadas(CellLattice& lattice, int count, int numCT, int numcC, int numPoint,
    double ncNoise_ct, double ncNoise_cc, const std::string& fileName, 
    const std::vector<Obstacle>& point, int sr_normal, int sr_cancer)
{
    std::ofstream outputFile(fileName);
    if (!outputFile.is_open()) {
        logError("Failed to open file: " + fileName);
        return;
    }
    outputFile << "run,steps,escapers,seed\n";

    std::mt19937 rng_run_selector(GLOBAL_SEED + numPoint + numcC + 99999);
    std::uniform_int_distribution<int> dist_run(1, count);

    const int L = lattice.getWidth(); // assume grade quadrada
    //const double delta_t = 1.0 / (L * L);
    const double intervalo = 1.0;

    #pragma omp parallel for schedule(dynamic)
    for (int run = 1; run <= count; ++run) 
    {      
        unsigned int seed_run = GLOBAL_SEED + 10 * sr_cancer + 10 * sr_normal + run + 1000 * numcC + 100000 * numPoint;
        std::mt19937 rng_local(seed_run);

        std::vector<Cell> cT_local, cC_local;
        std::vector<Obstacle> point_local = point;

        if (!lattice.placeObjects(point_local, cT_local, cC_local, numPoint, numCT, numcC, rng_local, sr_normal, sr_cancer)) 
        {
            #pragma omp critical
            logError("Failed to place objects in run " + std::to_string(run));
            continue;
        }

        std::uniform_int_distribution<int> distX(0, L - 1);
        std::uniform_int_distribution<int> distY(0, L - 1);

        double t = 0.0;
        double proximoRegistro = intervalo;
        bool verificacC;

        // Registra o tempo inicial com total de cC
        #pragma omp critical
        std::ofstream evoFile(fileName + "_run_" + std::to_string(run) + "_presas_por_passo.csv");
    	evoFile << "passo,presas_vivas\n";
        evoFile << t << "," << cC_local.size() << "\n";
    

        while (t < 1.0e4)
        {
            for (int i = 0; i < L * L; ++i)
            {
                int x_rand = distX(rng_local);
                int y_rand = distY(rng_local);
                std::string valor = lattice.getGridValue(x_rand, y_rand);
                // Verifica se há célula naquela posição
                bool encontrou = false;
                for (auto& cel : cT_local) 
                {
                    if (cel.getCoordenadaX() == x_rand && cel.getCoordenadaY() == y_rand) 
                    {
                        verificacC = false;
                        moverCelulaBoa(lattice, cel, cT_local, cC_local, point_local, rng_local, verificacC,2);
                        encontrou = true;
                        break;
                    }
                }
                if (!encontrou) 
                {
                    for (auto& cel : cC_local) 
                    {
                        if (cel.getCoordenadaX() == x_rand && cel.getCoordenadaY() == y_rand) 
                        {
                            verificacC = true;
                            moverCelulaRuim(lattice, cel, cT_local, cC_local, point_local, rng_local, verificacC,2);
                            break;
                        }
                    }
                }
                if (t >= proximoRegistro) 
                {
                    #pragma omp critical
                    evoFile << t << "," << cC_local.size() << "\n";
                    proximoRegistro += intervalo;
                }
            }
            if (cC_local.empty()) break;
            t += 1.0;
        }
        #pragma omp critical
        outputFile << run << "," << t << "," << cC_local.size() << "," << seed_run << "\n";
        evoFile.close();
    }

    outputFile.close();
    logInfo("Resultados salvos em: " + fileName);
}

int main() 
{
    int count, numCT, numcC, numPoint;


    SIZE = 128;
    WIDTH = SIZE;
    HEIGHT = SIZE;
    CAPTUREPROBABILITY = 1.0;
    //std::vector<int> numcT_values = {720};
    std::vector<int> numcC_values = {25, 50, 100, 200, 400, 800};
    std::vector<int> numPoint_values = {0};
    std::vector<double> numNoise_ct = {1.00};
    std::vector<double> numNoise_cc = {1.00};

    std::vector<Cell> cT;
    std::vector<Cell> cC;
    std::vector<Obstacle> point;
    std::string line;
    CellLattice lattice(WIDTH,HEIGHT);
    int sr_normal = 2;
    int sr_cancer = 2;

    count = 100;
        for (double ncNoise_ct : numNoise_ct) 
        {
            setCTPROBABILITY(ncNoise_ct);
            for (double ncNoise_cc : numNoise_cc) 
            {
                setCCPROBABILITY(ncNoise_cc);

                for (int ncC_value : numcC_values) 
                {
                    numcC = ncC_value;
                    numCT = numcC;
                    for (int np_value : numPoint_values) 
                    {
                        numPoint = np_value;
                        cT.clear();
                        cC.clear();
                        point.clear();
                        std::string fileName = gerarNomeArquivo(numCT, numcC, numPoint, ncNoise_cc, ncNoise_ct, sr_normal, sr_cancer);
                        executarRodadas(lattice, count, numCT, numcC, numPoint, ncNoise_ct, ncNoise_cc, fileName, point, sr_normal, sr_cancer);
                    }
                }
            }
        }
    return 0;
}
