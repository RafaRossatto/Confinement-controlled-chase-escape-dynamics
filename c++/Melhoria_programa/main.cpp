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





/*
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

    if (!lattice.isOccupied(newX, newY, cT, cC, obstacles, checkcC)) 
    {
        // Limpa a posição antiga
        lattice.setGridValue(x, y, "L");
    
        // Move a célula
        cell.changePosition(newX, newY);
    
        // Marca a nova posição
        lattice.setGridValue(newX, newY, "O"); // tipo "O" para célula ruim
    }
}
*/



void moverCelulaRuim(CellLattice& lattice, Cell& cell,
    std::vector<Cell>& cT, std::vector<Cell>& cC,
    std::vector<Obstacle>& obstacles,
    std::mt19937& rng, bool checkcC) 
{
    int x = cell.getCoordenadaX();
    int y = cell.getCoordenadaY();
    int newX = x, newY = y;

    // Junta as células em um vetor só (como você já faz)
    std::vector<Cell> celulas;
    celulas.insert(celulas.end(), cT.begin(), cT.end());
    celulas.insert(celulas.end(), cC.begin(), cC.end());

    const std::array<Direction, 4> direcoes = {NORTH, EAST, SOUTH, WEST};
    std::array<Direction, 4> direcoesEmbaralhadas = direcoes;
    std::shuffle(direcoesEmbaralhadas.begin(), direcoesEmbaralhadas.end(), rng);

    // Densidade de caçadores ao redor da posição atual
    int densidadeAtual = contarAlvosAoRedor(x, y, celulas, lattice, {"N"}, 2);

    int menorDensidade = densidadeAtual;
    std::vector<Direction> melhoresDirecoes;

    for (Direction dir : direcoesEmbaralhadas) 
    {
        int tempX = x, tempY = y;
        cell.randonWalk(tempX, tempY, dir);

        // Só considera vizinhos livres
        if (lattice.isOccupied(tempX, tempY, cT, cC, obstacles, checkcC))
        continue;

        int densidadeVizinha = contarAlvosAoRedor(tempX, tempY, celulas, lattice, {"N"}, 2);
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
        logInfo("[O] Fica parado — nenhuma direção mais segura que a atual");
        newX = x;
        newY = y;
    }

    if (!lattice.isOccupied(newX, newY, cT, cC, obstacles, checkcC)) 
    {
        lattice.setGridValue(x, y, "L");
        cell.changePosition(newX, newY);
        lattice.setGridValue(newX, newY, "O"); // tipo "O" para célula ruim
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
        std::vector<Cell> celulas;
        celulas.insert(celulas.end(), cT.begin(), cT.end());
        celulas.insert(celulas.end(), cC.begin(), cC.end());
        const std::array<Direction, 4> direcoes = {NORTH, EAST, SOUTH, WEST};
        std::array<Direction, 4> direcoesEmbaralhadas = direcoes;
        std::shuffle(direcoesEmbaralhadas.begin(), direcoesEmbaralhadas.end(), rng);
        int densidadeAtual = contarAlvosAoRedor(x, y, celulas, lattice, {"O"}, 2);
        int maiorDensidade = densidadeAtual;
        std::vector<Direction> melhoresDirecoes;
        for (Direction dir : direcoesEmbaralhadas)
        {
            int tempX = x, tempY = y;
            cell.randonWalk(tempX, tempY, dir);
            if (lattice.isOccupied(tempX, tempY, cT, cC, obstacles, checkcC))
            continue;
            int densidadeVizinha = contarAlvosAoRedor(tempX, tempY, celulas, lattice, {"O"}, 2);
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
            // Se for capturar alguém (i.e. mover para cima de uma célula "O")
            for (auto it = cC.begin(); it != cC.end(); )
            {
            if (it->getCoordenadaX() == newX && it->getCoordenadaY() == newY)
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
        else
        {
            logInfo("[N] Fica parado — nenhuma direção com mais alvos");
        }
    }
    else
    {
        std::uniform_int_distribution<int> dir(0, 3);
        cell.randonWalk(newX, newY, static_cast<Direction>(dir(rng)));
        logInfo("[N] Movimento aleatório (não inteligente)");
    }

    if (!lattice.isOccupied(newX, newY, cT, cC, obstacles, checkcC))
    {
        lattice.setGridValue(x, y, "L");
        cell.changePosition(newX, newY);
        lattice.setGridValue(newX, newY, "N");
    }
}

void executarRodadas(CellLattice& lattice, int count, int numCT, int numcC, int numPoint,
    double ncNoise_ct, double ncNoise_cc,
    const std::string& fileName, const std::vector<Obstacle>& point,
    int sr_normal, int sr_cancer)
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
    const double delta_t = 1.0 / (L * L);
    const double intervalo = 1.0;

    #pragma omp parallel for schedule(dynamic)
    for (int run = 1; run <= count; ++run) 
    {
        std::string nomeArquivoPresas = "presas_vs_tempo_run_" + std::to_string(run) + ".dat";
        std::ofstream dadosPresas(nomeArquivoPresas);
        dadosPresas << "run,tempo,presas\n";
        

        unsigned int seed_run = GLOBAL_SEED + 10 * sr_cancer + 10 * sr_normal + run + 1000 * numcC + 100000 * numPoint;
        std::mt19937 rng_local(seed_run);

        std::vector<Cell> cT_local, cC_local;
        std::vector<Obstacle> point_local = point;

        if (!lattice.placeObjects(point_local, cT_local, cC_local, numPoint, numCT, numcC, rng_local, sr_normal, sr_cancer)) {
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
        dadosPresas << t << "," << cC_local.size() << "\n";

        while (t < 50000.0)
        {
            int x_rand = distX(rng_local);
            int y_rand = distY(rng_local);
            std::string valor = lattice.getGridValue(x_rand, y_rand);

            // Verifica se há célula naquela posição
            bool encontrou = false;

            for (auto& cel : cT_local) {
                if (cel.getCoordenadaX() == x_rand && cel.getCoordenadaY() == y_rand) {
                    verificacC = false;
                    moverCelulaBoa(lattice, cel, cT_local, cC_local, point_local, rng_local, verificacC);
                    encontrou = true;
                    break;
                }
            }

            if (!encontrou) {
                for (auto& cel : cC_local) {
                    if (cel.getCoordenadaX() == x_rand && cel.getCoordenadaY() == y_rand) {
                        verificacC = true;
                        moverCelulaRuim(lattice, cel, cT_local, cC_local, point_local, rng_local, verificacC);
                        break;
                    }
                }
            }

            #pragma omp critical
            std::cout << std::fixed << std::setprecision(6)
                      << "[RUN " << run << " - STEP " << t << "] (" << x_rand << "," << y_rand << ") = "
                      << valor << " → " << (encontrou ? "AÇÃO" : "NADA") << "\n";

            if (t >= proximoRegistro) {
                #pragma omp critical
                dadosPresas << run << "," << t << "," << cC_local.size() << "\n";
                proximoRegistro += intervalo;
            }

            if (cC_local.empty()) break;
            t += delta_t;
        }

        #pragma omp critical
        outputFile << run << "," << t << "," << cC_local.size() << "," << seed_run << "\n";
        dadosPresas.close();
    }

    outputFile.close();
    //dadosPresas.close();
    logInfo("Resultados salvos em: " + fileName + " e presas_vs_tempo.dat");
}



int main() 
{
    int count, numCT, numcC, numPoint;


    SIZE = 128;
    WIDTH = SIZE;
    HEIGHT = SIZE;
    CAPTUREPROBABILITY = 1.0;


    std::vector<int> numcT_values = {20};
    std::vector<int> numcC_values = {25};
    std::vector<int> numPoint_values = {0};
    std::vector<double> numNoise_ct = {0.99};
    std::vector<double> numNoise_cc = {0.99};

    std::vector<Cell> cT;
    std::vector<Cell> cC;
    std::vector<Obstacle> point;
    std::string line;
    CellLattice lattice(WIDTH,HEIGHT);
    int sr_normal = 2;
    int sr_cancer = 2;

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

                        std::string fileName = gerarNomeArquivo(numCT, numcC, numPoint, ncNoise_cc, ncNoise_ct, sr_normal, sr_cancer);
                        executarRodadas(lattice, count, numCT, numcC, numPoint, ncNoise_ct, ncNoise_cc, fileName, point, sr_normal, sr_cancer);
                    }
                }
            }
        }
    }
    return 0;
}
