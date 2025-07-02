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
std::random_device rd;
unsigned int GLOBAL_SEED = rd();

//int SEARCHRADIUS = 40; // raio de procura.
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

void moverCelulaRuim(CellLattice& lattice, Cell& celula,
                     std::vector<Cell>& cT, std::vector<Cell>& cC,
                     std::vector<Obstacle>& obstaculos,
                     std::mt19937& rng, bool verificacC)

{
    int x = celula.getCoordenadaX();
    int y = celula.getCoordenadaY();
    int newX = x, newY = y;
    std::vector<Cell> celulas;
    celulas.insert(celulas.end(), cT.begin(), cT.end());
    celulas.insert(celulas.end(), cC.begin(), cC.end());

    // Busca caçadores no raio de procura
    std::vector<std::string> tipos = {"N"};
    auto alvos = celula.findNearestTarget(celulas, lattice, tipos);

    if (!alvos.empty()) 
    {
        // Sorteia um caçador dentre os encontrados
        std::uniform_int_distribution<int> dist(0, alvos.size() - 1);
        auto [alvoX, alvoY] = alvos[dist(rng)];

        // Estratégia NOI: mover para direção que mais aumenta a distância até o caçador
        std::vector<std::pair<int, int>> direcoes = 
        {
            {0, -1}, {0, 1}, {1, 0}, {-1, 0}
        };

        std::vector<int> melhoresDirecoes;
        double melhorValor = -1e9;

        for (int i = 0; i < 4; ++i) 
        {
            int dx = direcoes[i].first;
            int dy = direcoes[i].second;
            int tempX = x + dx;
            int tempY = y + dy;

            double dist = lattice.calculateDistance(tempX, tempY, alvoX, alvoY);

            if (dist > melhorValor) 
            {
                melhorValor = dist;
                melhoresDirecoes.clear();
                melhoresDirecoes.push_back(i);
            }
            else if (dist == melhorValor) 
            {
                melhoresDirecoes.push_back(i);
            }
        }

        if (!melhoresDirecoes.empty()) 
        {
            std::uniform_int_distribution<int> distEscolha(0, melhoresDirecoes.size() - 1);
            int direcaoEscolhida = melhoresDirecoes[distEscolha(rng)];
            celula.randonWalk(newX, newY, direcaoEscolhida);
        }
    } 
    else 
    {
        // Nenhum caçador encontrado — movimento aleatório
        std::uniform_int_distribution<int> dis(0, 3);
        celula.randonWalk(newX, newY, dis(rng));
    }

    if (!lattice.isOccupied(newX, newY, cT, cC, obstaculos, verificacC)) 
    {
        celula.changePosition(newX, newY);
    }
}


void moverCelulaBoa(CellLattice& lattice, Cell& celula,
    std::vector<Cell>& cT,
    std::vector<Cell>& cC,
    std::vector<Obstacle>& obstaculos,
    std::mt19937& rng, bool verificacC)
{
    int x = celula.getCoordenadaX();
    int y = celula.getCoordenadaY();
    int newX = x, newY = y;

    std::bernoulli_distribution d(CTPROBABILITY);
    int nProbability = d(rng) ? 1 : 0;

    if (nProbability == 1) 
    {
        std::vector<Cell> celulas;
        celulas.insert(celulas.end(), cT.begin(), cT.end());
        celulas.insert(celulas.end(), cC.begin(), cC.end());

        int raio = celula.getSearchRadius();

        // Busca todos os alvos visíveis
        auto alvos = celula.findNearestTarget(celulas, lattice, {"N", "O"});

        // Inicializa variáveis para armazenar o alvo mais próximo de cada tipo
        int menorDistPresa = std::numeric_limits<int>::max();
        int menorDistCacador = std::numeric_limits<int>::max();
        std::pair<int, int> posPresa, posCacador;
        bool encontrouPresa = false, encontrouCacador = false;

        for (const auto& [ax, ay] : alvos) 
        {
            for (const auto& agente : celulas) 
            {
                if (agente.getCoordenadaX() == ax && agente.getCoordenadaY() == ay) 
                {
                    int dist = lattice.calculateDistance(x, y, ax, ay);
                    if (agente.getTipo() == "O" && dist < menorDistPresa) 
                    {
                        menorDistPresa = dist;
                        posPresa = {ax, ay};
                        encontrouPresa = true;
                    }
                    else if (agente.getTipo() == "N" && dist < menorDistCacador) 
                    {
                        menorDistCacador = dist;
                        posCacador = {ax, ay};
                        encontrouCacador = true;
                    }
                    break;
                }
            }
        }

        bool isFugindo = false;
        int alvoX, alvoY;
        bool encontrou = false;

        if (encontrouPresa && (!encontrouCacador || menorDistPresa <= menorDistCacador)) 
        {
            std::tie(alvoX, alvoY) = posPresa;
            isFugindo = false;
            encontrou = true;
        } 
        else if (encontrouCacador) 
        {
            std::tie(alvoX, alvoY) = posCacador;
            isFugindo = true;
            encontrou = true;
        }

        if (encontrou) 
        {
            std::vector<std::pair<int, int>> direcoes = {{0, -1}, {0, 1}, {1, 0}, {-1, 0}};
            std::vector<int> melhoresDirecoes;
            double melhorValor = isFugindo ? -1e9 : 1e9;

            for (int i = 0; i < 4; ++i) 
            {
                int tempX = x + direcoes[i].first;
                int tempY = y + direcoes[i].second;
                double distAlvo = lattice.calculateDistance(tempX, tempY, alvoX, alvoY);

                if ((isFugindo && distAlvo > melhorValor) ||
                    (!isFugindo && distAlvo < melhorValor)) 
                {
                    melhorValor = distAlvo;
                    melhoresDirecoes.clear();
                    melhoresDirecoes.push_back(i);
                } 
                else if (distAlvo == melhorValor) 
                {
                    melhoresDirecoes.push_back(i);
                }
            }

            if (!melhoresDirecoes.empty()) 
            {
                std::uniform_int_distribution<int> escolha(0, melhoresDirecoes.size() - 1);
                int direcaoEscolhida = melhoresDirecoes[escolha(rng)];
                celula.randonWalk(newX, newY, direcaoEscolhida);
            }
        } 
        else 
        {
            std::uniform_int_distribution<int> dis(0, 3);
            celula.randonWalk(newX, newY, dis(rng));
        }
    } 
    else 
    {
        std::uniform_int_distribution<int> dis(0, 3);
        celula.randonWalk(newX, newY, dis(rng));
    }
    if (!lattice.isOccupied(newX, newY, cT, cC, obstaculos, verificacC)) 
    {
    celula.changePosition(newX, newY);
    }

    // Verifica se capturou uma célula ruim (presa)
    for (int i = cC.size() - 1; i >= 0; --i) 
    {
    if (cC[i].getCoordenadaX() == newX && cC[i].getCoordenadaY() == newY) 
    {
        cC.erase(cC.begin() + i);
        break;
    }
    }
}




std::pair<int, int> runSimulationPaper(CellLattice& lattice,const int NUMSTEPS, std::vector<Cell>& cT, 
std::vector<Cell>& cC, std::vector<Obstacle>& point, std::mt19937& rng) 
{
    int steps = 1; // Step counter
    bool verificacC;
	
	std::vector<int> presasPorPasso;
	//const int intervalo = 50;
    while (steps < NUMSTEPS) 
    {   
        // Process cells of type T
        if (!cT.empty())
        {
            for (int i = cT.size() - 1; i >= 0; --i)
            {
                verificacC = false;
                moverCelulaBoa(lattice,cT[i], cT, cC, point, rng,verificacC);
            }
        }
        // Process cancer cells
        if (!cC.empty())
        {
            for (int i = cC.size() - 1; i >= 0; --i)
            {
                verificacC = true;
                moverCelulaRuim(lattice,cC[i], cT, cC, point, rng,verificacC);
            }
        }

        if (cC.empty()) 
        {
            break;
        }
        steps++; // Increment step count
    }
    return {steps, static_cast<int>(cC.size())};
}


void executarRodadas(CellLattice& lattice,int count, int numCT, int numcC, int numPoint,
    double ncNoise_ct, double ncNoise_cc,
    const std::string& fileName, const std::vector<Obstacle>& point,int sr_normal,int sr_cancer)
{
    //int remaining_cC;
    std::ofstream outputFile(fileName);
    if (!outputFile.is_open()) 
    {
        logError("Erro ao abrir o arquivo " + fileName);
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
		/*
        std::ofstream evoFile(fileName + "_run_" + std::to_string(run) + "_presas_por_passo.csv");
    		evoFile << "passo,presas_vivas\n";
    		for (size_t i = 0; i < presasPorPasso.size(); ++i) 
		    {
        		evoFile << (i * intervalo) << "," << presasPorPasso[i] << "\n";
    		}
    	evoFile.close();
	    */
        }
        
    }
    outputFile.close();
    logInfo("Resultados salvos no arquivo " + fileName);
}


int main() 
{
    int count, numCT, numcC, numPoint;

    //std::vector<int> numcC_values = {1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 500, 1000};
    std::vector<int> numcT_values = {5,10,25,50,100,250,500,1000,5000};
    std::vector<int> numcC_values = {10,25,50};
    //std::vector<int> numPoint_values = {0, 2, 3, 5, 6, 11, 21, 26, 51};
    std::vector<int> numPoint_values = {0};
    std::vector<double> numNoise_ct = {0.00};
    std::vector<double> numNoise_cc = {1.0};

    std::vector<Cell> cT;
    std::vector<Cell> cC;
    std::vector<Obstacle> point;
    std::string line;
    CellLattice lattice(WIDTH,HEIGHT);
    int sr_normal = 100;
    int sr_cancer = 100;

    count = 100;
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

                        if (!lattice.loadObstacles(point, numPoint, line)) {
                            logError("Failed to load obstacles");
                            std::cin.get();
                            return 1;
                        }

                        std::string fileName = gerarNomeArquivo(numCT, numcC, numPoint, ncNoise_cc, ncNoise_ct, sr_normal, sr_cancer);
                        executarRodadas(lattice, count, numCT, numcC, numPoint, ncNoise_ct, ncNoise_cc, fileName, point, sr_normal, sr_cancer);
                    }
                }
            }
        }
    }
    return 0;
}
