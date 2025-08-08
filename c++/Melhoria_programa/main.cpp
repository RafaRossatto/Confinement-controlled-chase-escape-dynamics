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
    if (!outputFile.is_open()) 
    {
        logError("Failed to open file: " + fileName);
        return;
    }
    outputFile << "run,steps,escapers,seed\n";
    std::mt19937 rng_run_selector(GLOBAL_SEED + numPoint + numcC + 99999);
    std::uniform_int_distribution<int> dist_run(1, count);

    const int L = lattice.getWidth(); // assume grade quadrada
    const double intervalo = 1.0;

    #pragma omp parallel for schedule(dynamic)
    for (int run = 0; run <= count-1; ++run) 
    {      
        unsigned int seed_run = GLOBAL_SEED + 10 * sr_cancer + 10 * sr_normal + run + 1000 * numcC + 100000 * numPoint;
        std::mt19937 rng_local(seed_run);

        std::ostringstream oss_nc;
        oss_nc << std::setw(2) << std::setfill('0') << numCT;
        std::string ncStr = "nC_" + oss_nc.str();  // ex: nC_05

        std::ostringstream oss_run;
        oss_run << std::setw(2) << std::setfill('0') << run;
        std::string runStr = "run_" + oss_run.str();  // ex: run_03

        std::ostringstream oss_obs;
        oss_obs << std::setw(2) << std::setfill('0') << numPoint;
        std::string ObsStr = "obs_" + oss_obs.str();  // por exemplo, o_15

        std::string caminhoArquivo = "../" + ObsStr + "/" + ncStr + "/" + runStr + "/inaccessible_preys.txt";


        std::vector<Cell> cT_local, cC_local;
        std::vector<Obstacle> point_local = point;
        
        std::ifstream arquivo(caminhoArquivo);
        if (!arquivo.is_open()) 
            {
                std::cerr << "Erro ao abrir o arquivo: " << caminhoArquivo << std::endl;
                    continue; // pula a simulação se não conseguir abrir
            }

        if (!lattice.placeObjects(point_local, cT_local, cC_local, numPoint, numCT, numcC, rng_local, sr_normal, sr_cancer,run)) 
        {
            #pragma omp critical
            logError("Failed to place objects in run " + std::to_string(run));
            continue;
        }
                    

        int countInacessiveis = 0;
        arquivo >> countInacessiveis;
        arquivo.close();    
        std::uniform_int_distribution<int> distX(0, L - 1);
        std::uniform_int_distribution<int> distY(0, L - 1);

        double t = 0.0;
        double proximoRegistro = intervalo;
        bool verificacC;

        std::ofstream evoFile(fileName + "_run_" + std::to_string(run) + "_presas_por_passo.csv");
        if (!evoFile.is_open()) 
        {
            #pragma omp critical
            std::cerr << "Erro ao criar evoFile da run " << run << std::endl;
            continue;
        }
        // Registra o tempo inicial com total de cC
        #pragma omp critical
    	evoFile << "passo,presas_vivas\n";
        evoFile << t << "," << cC_local.size() << "\n";

        while (t < 1.0e5)
        {
            if (countInacessiveis == numcC) 
            {
                #pragma omp critical
                evoFile << t << "," << cC_local.size() << "\n";
                break;
            }   

            for (int i = 0; i < L * L; ++i)
            {
                int x_rand = distX(rng_local);
                int y_rand = distY(rng_local);
                std::string valor = lattice.getGridValue(x_rand, y_rand);
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
            }

            if (t >= proximoRegistro) 
            {
                #pragma omp critical
                evoFile << t << "," << cC_local.size() << "\n";
                proximoRegistro += intervalo;
                //std:: cin.get();
            }

            if (static_cast<int>(cC_local.size()) <= countInacessiveis) break;
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
    int count = 100;
    int numCT, numcC, numPoint;

    SIZE = 128;
    WIDTH = SIZE;
    HEIGHT = SIZE;
    CAPTUREPROBABILITY = 1.0;

    //std::vector<int> numcT_values = {5,10,50,100,500,1000,1500};       // <- número de presas
    std::vector<int> numcT_values = {100,500,1000,1500};
    //std::vector<int> numPoint_values = {1638,3276,4915,6553,8192
      //                                  ,9830,11468,13107,14745};    // <- número de obstáculos
    std::vector<int> numPoint_values = {11468};    // <- número de obstáculos
    std::vector<double> numNoise_ct = {1.00};
    std::vector<double> numNoise_cc = {1.00};

    std::vector<Cell> cT;
    std::vector<Cell> cC;
    std::vector<Obstacle> point;

    CellLattice lattice(WIDTH, HEIGHT);
    int sr_normal = 2;
    int sr_cancer = 2;

    for (double ncNoise_ct : numNoise_ct) 
    {
        setCTPROBABILITY(ncNoise_ct);

        for (double ncNoise_cc : numNoise_cc) 
        {
            setCCPROBABILITY(ncNoise_cc);

            for (int numPoint : numPoint_values) 
            {
                int L = lattice.getWidth();
                int total_sites = L * L;

                int numFree = total_sites - numPoint;

                for (int numCT_value : numcT_values) 
                {
                    numCT = numCT_value;
                    numcC = (numFree)/2;  // resto do espaço livre
                    if (numcC < 0) {
                        std::cerr << "Configuração inválida: mais presas que espaço livre na grade!\n";
                        continue;
                    }

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
