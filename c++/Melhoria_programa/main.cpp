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

std::random_device rd;
unsigned int GLOBAL_SEED = rd();

//int SEARCHRADIUS = 40; // raio de procura.
double CTPROBABILITY = 0.50; // probabilidade de capturar ou procurar comida
double CCPROBABILITY = 0.50; // probabilidade de escapar ou procurar comida

void setCTPROBABILITY(double newValue)
{
    CTPROBABILITY = newValue;
}

void setCCPROBABILITY(double newValue)
{
    CCPROBABILITY = newValue;
}
     
void movePosition(CellLattice& lattice,int& x, int& y, int targetX, int targetY, int nProbability,
std::vector<Cell>& celulas, std::vector<Cell>& cC, std::vector<Obstacle>& obstaculos,
int verificacC, bool isPursuing,std::mt19937& rng,Cell& celula) 
{
    int prevX = x, prevY = y;
    int newX = x, newY = y;
    int maxAttempts = 100;
    int attemptCount = 0;

    while (attemptCount < maxAttempts)
    {
        newX = x;
        newY = y;

        if (nProbability == 1)
        {
            // Lógica modificada para movimento direcionado
            // Verifica se estão alinhados (mesma linha ou coluna)
            if (x == targetX || y == targetY)
            {
                std::vector<int> direcoesPossiveis;
                if (x == targetX) // Mesma coluna: o movimento será vertical
                {
                    if (isPursuing)
                    {
                        // Perseguir: mover na direção que diminua a diferença de y
                        if (y < targetY)
                        {
                            direcoesPossiveis.push_back(NORTH); // aumenta y
                        }
                        else
                        {    
                        direcoesPossiveis.push_back(SOUTH); // diminui y
                        }
                    }
                    else
                    {
                        // Fugir: evitar o movimento que aproxima (o contrário do que reduziria a diferença)
                        if (y < targetY)
                        {
                            direcoesPossiveis = {EAST, WEST, SOUTH}; // NÃO usar NORTH
                        }
                        else
                        {
                        direcoesPossiveis = {EAST, WEST, NORTH}; // NÃO usar SOUTH
                        }
                    }
                }
                else if (y == targetY) // Mesma linha: o movimento será horizontal
                {
                    if (isPursuing)
                    {
                        if (x < targetX)
                        {
                            direcoesPossiveis.push_back(EAST); // aumenta x
                        }
                        else
                        {
                            direcoesPossiveis.push_back(WEST); // diminui x
                        }
                    }
                    else
                    {
                        if (x < targetX)
                        {
                            direcoesPossiveis = {NORTH, SOUTH, WEST}; // NÃO usar EAST
                        }
                        else
                        {
                            direcoesPossiveis = {NORTH, SOUTH, EAST}; // NÃO usar WEST
                        }
                    }
                }
                // Seleciona uma direção dentre as possíveis
                std::uniform_int_distribution<int> dist(0, direcoesPossiveis.size()-1);
                int direcaoEscolhida = direcoesPossiveis[dist(rng)];
                celula.randonWalk(newX, newY, direcaoEscolhida);
            }
            else
            {
                // Caso diagonal
                std::vector<int> direcoesPossiveis;
                int dx = targetX - x;
                int dy = targetY - y;
                if (isPursuing)
                {
                    // Permite apenas os movimentos que aproximam: direções que diminuem |dx| ou |dy|
                    if (dx > 0)
                    {
                        direcoesPossiveis.push_back(EAST);
                    }
                    else
                    {
                        direcoesPossiveis.push_back(WEST);
                    }
                    if (dy > 0)
                    {
                        direcoesPossiveis.push_back(NORTH);
                    }
                    else
                    {
                        direcoesPossiveis.push_back(SOUTH);
                    }       
                }
                else
                {
                    // Fugir: inverte os sinais para aumentar a distância
                    if (dx > 0)
                    {
                        direcoesPossiveis.push_back(WEST);
                    }
                    else 
                    {
                        direcoesPossiveis.push_back(EAST);
                    }
                    if (dy > 0)
                    {
                        direcoesPossiveis.push_back(SOUTH);
                    }
                    else
                    {
                        direcoesPossiveis.push_back(NORTH);
                    }
                }
                // Como estamos na diagonal, geralmente teremos duas direções
                std::uniform_int_distribution<int> dist(0, direcoesPossiveis.size()-1);
                int direcaoEscolhida = direcoesPossiveis[dist(rng)];
                celula.randonWalk(newX, newY, direcaoEscolhida);
            }
        }
        else
        {
            // Movimento aleatório
            std::uniform_int_distribution<int> dis(0, 3);
            celula.randonWalk(newX, newY, dis(rng));
        }

        bool capturando = isPursuing && (verificacC == 0);  // verificacC == 0 → célula boa
    
        //Essa parte é nova
    if (!lattice.isOccupied(newX, newY, celulas, cC, obstaculos, capturando ? false : verificacC))
    {
        x = newX;
        y = newY;
        return;
    }
    //Essa parte é nova
        //Essa parte é velha
    /* 
    if (!lattice.isOccupied(newX, newY, celulas, cC, obstaculos, verificacC))
        {
            x = newX;
            y = newY;
            return;
        }
        //Essa parte é velha
      */
	else
        {
            attemptCount++;
        }
    }
    x = prevX;
    y = prevY;
}

void moverCelulaRuim(CellLattice& lattice,Cell& celula, std:: vector<Cell>&cT,std::vector<Cell>& cC,
                    std::vector<Obstacle>& obstaculos, std:: mt19937& rng,bool verificacC)
{
    {
        int x = celula.getCoordenadaX();
        int y = celula.getCoordenadaY();

        bool isPursiung;
        int randomValue;
        
        auto [nearestCTIndex, distanciaAteCancer] = celula.findNearestTarget(cT, celula.getSearchRadius(), lattice);
        if (nearestCTIndex != -1 and distanciaAteCancer <= celula.getSearchRadius())
        {
            isPursiung = false;
            std:: bernoulli_distribution d(CCPROBABILITY);
            randomValue = d(rng) ? 1: 0 ;
            movePosition(lattice,x,y,
                        cT[nearestCTIndex].getCoordenadaX(),
                        cT[nearestCTIndex].getCoordenadaY(),
                        randomValue, cT, cC,
                        obstaculos, verificacC,
                        isPursiung, rng,celula);
        }
    
        else
        {
            // Não achou o alvo
            isPursiung = false;
            randomValue = 0;
            movePosition(lattice,x,y,0, 0,
                randomValue, cT, cC,
                obstaculos, verificacC,
                isPursiung, rng,celula);
        }
    
        celula.changePosition(x,y);
    }    
}

void moverCelulaBoa(CellLattice& lattice,Cell& celula, std:: vector<Cell>&cT,std::vector<Cell>& cC,
    std::vector<Obstacle>& obstaculos, std:: mt19937& rng, bool verificacC)
{
    int x = celula.getCoordenadaX();
    int y = celula.getCoordenadaY();

    bool isPursiung;
    int randomValue;
    auto [nearestCCIndex, distanciaAteCancer] = celula.findNearestTarget(cC, celula.getSearchRadius(), lattice);
    if (nearestCCIndex != -1 and distanciaAteCancer <= celula.getSearchRadius())
    {
        isPursiung = true;
        std:: bernoulli_distribution d(CTPROBABILITY);
        randomValue = d(rng) ? 1: 0 ;
	isPursiung = (randomValue == 1); // ← CORRETO: perseguição só se sorteou 1
        movePosition(lattice,x,y,
                    cC[nearestCCIndex].getCoordenadaX(),
                    cC[nearestCCIndex].getCoordenadaY(),
                    randomValue, cT, cC,
                    obstaculos, verificacC,
                    isPursiung, rng,celula);
    }
    else
    {
        // Não achou o alvo
        isPursiung = false;
        randomValue = 0;
        movePosition(lattice,x,y,0, 0,
            randomValue, cT, cC,
            obstaculos, verificacC,
            isPursiung, rng,celula);
    }

    celula.changePosition(x,y);
    
    if (indiceValido(nearestCCIndex,cC.size()) and
    x == cC[nearestCCIndex].getCoordenadaX()and
    y == cC[nearestCCIndex].getCoordenadaY())
    {
        cC.erase(cC.begin() + nearestCCIndex);
    }
}


std::pair<int, int> runSimulationPaper(CellLattice& lattice,const int NUMSTEPS, std::vector<Cell>& cT, 
std::vector<Cell>& cC, std::vector<Obstacle>& point, std::mt19937& rng) 
{
    int steps = 1; // Step counter
    bool isPursuing,verificacC;
	
	std::vector<int> presasPorPasso;
	const int intervalo = 50;
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


/*


void executarRodadas(CellLattice& lattice,int count, int numCT, int numcC, int numPoint,
    double ncNoise_ct, double ncNoise_cc,
    const std::string& fileName, const std::vector<Obstacle>& point,int sr_normal,int sr_cancer)
{
    int remaining_cC;
    std::ofstream outputFile(fileName);
    if (!outputFile.is_open()) 
    {
        logError("Erro ao abrir o arquivo " + fileName);
        std::cin.get();
        return;
    }
   	outputFile << "run,steps,escapers,seed\n";
	int runSorteada = std::uniform_int_distribution<int>(1, count)(std::mt19937(GLOBAL_SEED + numPoint + numcC));
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
                                    std::cin.get();
                                    continue;
                                }

        //int steps_local = runSimulationPaper(lattice,10000, cT_local, cC_local, point_local, rng_local);
        auto [steps_local, remaining_cC] = runSimulationPaper(lattice, 1000, cT_local, cC_local, point_local, rng_local);

        #pragma omp critical
        {
            outputFile << run << "," << steps_local << "," << remaining_cC << "," << seed_run << "\n";
        }
	if (run == runSorteada) {
            std::ofstream evoFile(fileName + "_presas_por_passo.dat");
            evoFile << "passo,presas_vivas\n";
            for (size_t i = 0; i < presasPorPasso.size(); ++i) {
                evoFile << (i * intervalo) << "," << presasPorPasso[i] << "\n";
            }
            evoFile.close();
        }
    }
    outputFile.close();
    logInfo("Resultados salvos no arquivo " + fileName);
}
*/

void executarRodadas(CellLattice& lattice,int count, int numCT, int numcC, int numPoint,
    double ncNoise_ct, double ncNoise_cc,
    const std::string& fileName, const std::vector<Obstacle>& point,int sr_normal,int sr_cancer)
{
    int remaining_cC;
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
int runSorteada = dist_run(rng_run_selector);

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
        const int intervalo = 10;

        int steps = 1;
        bool isPursuing, verificacC;
        while (steps < 2800) 
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
		std::ofstream evoFile(fileName + "_run_" + std::to_string(run) + "_presas_por_passo.csv");
    		evoFile << "passo,presas_vivas\n";
    		for (size_t i = 0; i < presasPorPasso.size(); ++i) 
		{
        		evoFile << (i * intervalo) << "," << presasPorPasso[i] << "\n";
    		}
    	evoFile.close();
	}
    }
    outputFile.close();
    logInfo("Resultados salvos no arquivo " + fileName);
}


int main() 
{
    int count, steps, numCT, numcC, numPoint, x, y;

    //std::vector<int> numcC_values = {1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 500, 1000};
    std::vector<int> numcC_values = {100};
    //std::vector<int> numPoint_values = {0, 2, 3, 5, 6, 11, 21, 26, 51};
    std::vector<int> numPoint_values = {0,51};
    std::vector<double> numNoise_ct = {0.95};
    std::vector<double> numNoise_cc = {0.95};

    std::vector<Cell> cT;
    std::vector<Cell> cC;
    std::vector<Obstacle> point;
    std::string line;
    CellLattice lattice(WIDTH,HEIGHT);

    numCT = 500;
    count = 100;

    for (int sr_normal = 1; sr_normal <= 50; ++sr_normal) 
    {
        for (int sr_cancer = 1; sr_cancer <= 50; ++sr_cancer)
        {

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
}
    return 0;
}
