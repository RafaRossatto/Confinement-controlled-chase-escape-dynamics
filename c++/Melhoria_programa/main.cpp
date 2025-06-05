#include "obstacle.h"
#include "cell.h"
#include "config.h"
#include "utils.h"
#include "cell_lattice.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <sstream>
#include<utility>

std::random_device rd;
unsigned int GLOBAL_SEED = rd();

int SEARCHRADIUS = 40; // raio de procura.
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

void setSEARCHRADIUS(int newValue)
{
    SEARCHRADIUS = newValue;
}

std::pair<int,int> encontrarAlvosMaisProximo(int x,int y,
                                            const std:: vector<Cell> alvos,
                                            int searchRadius)
{
    int nearestIndex = -1;
    int minDistance = std:: numeric_limits<int>::max();

    std:: vector<int> RAIOS_PROGRESSIVOS = {5, 10, 20, 30, 40, 50};

    for (int radius : RAIOS_PROGRESSIVOS)
    {
        if (radius > searchRadius) break;
        for (int i = 0; i < alvos.size(); i++)
        {
            int distance = calculeDistance (x,y, alvos[i].getCoordenadaX(),
                                           alvos[i].getCoordenadaY());
    
            if (distance <= radius and distance < minDistance)
            {
                minDistance = distance;
                nearestIndex = i;
            }
        }
    }    
    return{nearestIndex, minDistance};
}           

bool estaOcupado(int x, int y, const std::vector<Cell>& celulas,const std::vector<Cell>& cC, 
const std::vector<Obstacle>& obstaculos,int verificacC) 
{
    for (const auto& celula : celulas) 
    {
        if (celula.getCoordenadaX() == x && celula.getCoordenadaY() == y) 
        {
            return true;
        }
    }
    if(verificacC == 1)
    {
        for (const auto& c_celula : cC) 
        {
            if (c_celula.getCoordenadaX() == x && c_celula.getCoordenadaY() == y) 
            {
                return true;
            }
        }
    }   
    for (const auto& obstaculo : obstaculos) 
    {
        if (obstaculo.getCoordenadaX() == x && obstaculo.getCoordenadaY() == y) 
        {
            return true;
        }
    }
    return false;
}

void movePosition(int& x, int& y, int targetX, int targetY, int nProbability,
    std::vector<Cell>& celulas, std::vector<Cell>& cC, std::vector<Obstacle>& obstaculos,
    int verificacC, bool isPursuing,std::mt19937& rng ) 
{
    int prevX = x, prevY = y;
    int newX = x, newY = y;
    int maxAttempts = 100;
    int attemptCount = 0;

    while (attemptCount < maxAttempts)
    {
        // Reinicializa newX, newY a cada tentativa para evitar acumulação
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
                randonWalk(newX, newY, direcaoEscolhida);
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
                randonWalk(newX, newY, direcaoEscolhida);
            }
        }
        else
        {
            // Movimento aleatório
            std::uniform_int_distribution<int> dis(0, 3);
            randonWalk(newX, newY, dis(rng));
        }

        if (!estaOcupado(newX, newY, celulas, cC, obstaculos, verificacC))
        {
            x = newX;
            y = newY;
            return;
        }
        else
        {
            attemptCount++;
        }
    }
    x = prevX;
    y = prevY;
}

template<typename T, typename U>
bool sobrepoeComLista(const T& novo, const std:: vector<U>& lista)
{
    for(const auto& item : lista)
    {
        if (novo.getCoordenadaX() == item.getCoordenadaX() and
            novo.getCoordenadaY() == item.getCoordenadaY())
        {
            return true;
        }
    }
    return false;
}

template <typename T>
bool verificaSobreposicaoGeral (const T& novo,
                                const std::vector<Obstacle>& pontos,
                                const std::vector<Cell>& celulas,
                                const std::vector<Cell>& cancers)
{
    return  sobrepoeComLista(novo,pontos) or
            sobrepoeComLista(novo, celulas) or
            sobrepoeComLista(novo, cancers);
}

void moverCelulaRuim(Cell& celula, std:: vector<Cell>&cT,std::vector<Cell>& cC,
                    std::vector<Obstacle>& obstaculos, std:: mt19937& rng)
{
    {
        int x = celula.getCoordenadaX();
        int y = celula.getCoordenadaY();
        
        auto [nearestCTIndex, distanciaAteCancer] = encontrarAlvosMaisProximo(x, y, cC, SEARCHRADIUS);
        bool isPursiung;
        int randomValue;
        const bool verificacC = true;
        
        if (nearestCTIndex != -1 and distanciaAteCancer <= SEARCHRADIUS)
        {
            isPursiung = false;
            std:: bernoulli_distribution d(CCPROBABILITY);
            randomValue = d(rng) ? 1: 0 ;
    
            movePosition(x,y,
                        cT[nearestCTIndex].getCoordenadaX(),
                        cT[nearestCTIndex].getCoordenadaY(),
                        randomValue, cT, cC,
                        obstaculos, verificacC,
                        isPursiung, rng);
        }
    
        else
        {
            // Não achou o alvo
            isPursiung = false;
            randomValue = 0;
            movePosition(x,y,0, 0,
                randomValue, cT, cC,
                obstaculos, verificacC,
                isPursiung, rng);
        }
    
        celula.changePosition(x,y);
    }    
}

void moverCelulaBoa(Cell& celula, std:: vector<Cell>&cT,std::vector<Cell>& cC,
    std::vector<Obstacle>& obstaculos, std:: mt19937& rng)
{
    int x = celula.getCoordenadaX();
    int y = celula.getCoordenadaY();
    
    auto [nearestCCIndex, distanciaAteCancer] = encontrarAlvosMaisProximo(x, y, cC, SEARCHRADIUS);
    bool isPursiung;
    int randomValue;
    const bool verificacC = false;
    
    if (nearestCCIndex != -1 and distanciaAteCancer <= SEARCHRADIUS)
    {
        // Achou alvo próximo

        isPursiung = true;
        std:: bernoulli_distribution d(CTPROBABILITY);
        randomValue = d(rng) ? 1: 0 ;

        movePosition(x,y,
                    cC[nearestCCIndex].getCoordenadaX(),
                    cC[nearestCCIndex].getCoordenadaY(),
                    randomValue, cT, cC,
                    obstaculos, verificacC,
                    isPursiung, rng);
    }

    else
    {
        // Não achou o alvo
        isPursiung = false;
        randomValue = 0;
        movePosition(x,y,0, 0,
            randomValue, cT, cC,
            obstaculos, verificacC,
            isPursiung, rng);
    }

    celula.changePosition(x,y);
    
    if (indiceValido(nearestCCIndex,cC.size()) and
    x == cC[nearestCCIndex].getCoordenadaX()and
    y == cC[nearestCCIndex].getCoordenadaY())
    {
        cC.erase(cC.begin() + nearestCCIndex);
    }
}

bool posicionarObjetos(std::vector<Obstacle>& pontos, std::vector<Cell>& celulas, std::vector<Cell>& cancers, 
    int num_pontos, int num_celulas, int num_cancers, int largura_max, int altura_max,std::mt19937& rng) 
    {
        int max_tentativas = 10000;
        if (num_pontos != 0) 
        {
            //logInfo(" Obstaculos já carregados: " + std::to_string(pontos.size()));
            // Não fazer mais nada com obstáculos
        } else {
            // Se num_pontos for 0, criar obstáculos aleatoriamente
            int tentativas_pontos = 0;
            while (tentativas_pontos < max_tentativas && pontos.size() < num_pontos) 
            {
                std::uniform_int_distribution<int> distribX(0, largura_max - 1);
                int x = distribX(rng);
                std::uniform_int_distribution<int> distribY(0, altura_max - 1);
                int y = distribY(rng);
                int novo_id = pontos.size() + 1;
                Obstacle novo_ponto = {"C", novo_id, x, y};
            }
        }
        // Posicionar células
        int tentativas_celulas = 0;
        while (tentativas_celulas < max_tentativas && celulas.size() < num_celulas) 
        {
            std::uniform_int_distribution<int> distribX(0, largura_max - 1);
            int x = distribX(rng);
            std::uniform_int_distribution<int> distribY(0, altura_max - 1);
            int y = distribY(rng);
            int novo_id = celulas.size() + 1;
            Cell nova_celula = {"N", novo_id, x, y};
    
            if (!verificaSobreposicaoGeral(nova_celula, pontos, celulas,cancers)) 
            {
                celulas.push_back(nova_celula);
                tentativas_celulas = 0;
            } else 
            {
                tentativas_celulas++;
            }
        }
    
        // Posicionar células cancerígenas
        int tentativas_cancers = 0;
        while (tentativas_cancers < max_tentativas && cancers.size() < num_cancers) 
        {
            std::uniform_int_distribution<int> distribX(0, largura_max - 1);
            int x = distribX(rng);
            std::uniform_int_distribution<int> distribY(0, altura_max - 1);
            int y = distribY(rng);
            int novo_id = cancers.size() + 1;
            Cell nova_cancer = {"O", novo_id, x, y};
    
            if (!verificaSobreposicaoGeral(nova_cancer, pontos, celulas, cancers)) 
            {
                cancers.push_back(nova_cancer);
                tentativas_cancers = 0;
            } else 
            {
                tentativas_cancers++;
            }
        }

        // Verificação final de sucesso
        if (celulas.size() == num_celulas && cancers.size() == num_cancers) 
        {
            return true;
        } 
        else 
        {
            return false;
        }
    }


int runSimulationPaper(const int NUMSTEPS, std::vector<Cell>& cT, 
std::vector<Cell>& cC, std::vector<Obstacle>& point, std::mt19937& rng) 
{
    int steps = 1; // Step counter
    bool isPursuing;
    // Search radius to find resources or cells
    
    // Initialize random number generator and Bernoulli distribution

    // Main simulation loop
    while (steps < NUMSTEPS) 
    {   
        // Process cells of type T
        if (!cT.empty())
        {
            for (int i = cT.size() - 1; i >= 0; --i)
            {
                moverCelulaBoa(cT[i], cT, cC, point, rng);
            }
        }

        // Process cancer cells
        if (!cC.empty())
        {
            for (int i = cC.size() - 1; i >= 0; --i)
            {
            moverCelulaRuim(cC[i], cT, cC, point, rng);
            }
        }
        // Terminate if no cancer cells remain
        if (cC.empty()) 
        {

            //logInfo("No more cancer cells. Terminating progam");
            break;
        }
        steps++; // Increment step count
        //std::cout << "Passo: " << steps << "\n";
    }
    return steps;
}

std::string gerarNomeArquivo(int numCT, int numcC, int numPoint,
    double ncNoise_cc, double ncNoise_ct,
    int SR_value) 
    {   
        return "NC_" + std::to_string(numCT) +
        "_NE_" + std::to_string(numcC) +
        "_O_" + std::to_string(numPoint) +
        "_TCC_" + std::to_string(ncNoise_cc) +
        "_TCT_" + std::to_string(ncNoise_ct) +
        "_SR_" + std::to_string(SR_value) + ".dat";
    }

void executarRodadas(int count, int numCT, int numcC, int numPoint,
    double ncNoise_ct, double ncNoise_cc, int SR_value,
    const std::string& fileName, const std::vector<Obstacle>& point)
{
    std::ofstream outputFile(fileName);
    if (!outputFile.is_open()) 
    {
        logError("Erro ao abrir o arquivo " + fileName);
        std::cin.get();
        return;
    }

    outputFile << "run,steps,seed\n";

    #pragma omp parallel for schedule(dynamic)
    for (int run = 1; run <= count; ++run) 
    {
        unsigned int seed_run = GLOBAL_SEED + run + 1000 * numcC + 100000 * numPoint;
        std::mt19937 rng_local(seed_run);

        std::vector<Cell> cT_local;
        std::vector<Cell> cC_local;
        std::vector<Obstacle> point_local = point;

        if (!posicionarObjetos( point_local, cT_local, cC_local,
                                numPoint, numCT, numcC, 100, 100, rng_local)) 
                                {
                                    #pragma omp critical
                                    logError("Erro ao posicionar objetos na execução " + std::to_string(run));
                                    std::cin.get();
                                    continue;
                                }

        int steps_local = runSimulationPaper(10000, cT_local, cC_local, point_local, rng_local);

        #pragma omp critical
        {
            outputFile << run << "," << steps_local << "," << seed_run << "\n";
            
        }
    }
    outputFile.close();
    logInfo("Resultados salvos no arquivo " + fileName);
}

int main() 
{
    int count, steps, numCT, numcC, numPoint, SR_value, x, y;

    std::vector<int> numcC_values = {1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 500, 1000};
    std::vector<int> numPoint_values = {0, 2, 3, 5, 6, 11, 21, 26, 51};
    std::vector<double> numNoise_ct = {0.95};
    std::vector<double> numNoise_cc = {0.95};

    std::vector<Cell> cT;
    std::vector<Cell> cC;
    std::vector<Obstacle> point;
    std::string line;


    CellLattice lattice(WIDTH,HEIGHT);

    numCT = 500;
    count = 1e3;
    SR_value = 50;
    setSEARCHRADIUS(SR_value);

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

                    std::string fileName = gerarNomeArquivo(numCT, numcC, numPoint, ncNoise_cc, ncNoise_ct, SR_value);
                    executarRodadas(count, numCT, numcC, numPoint, ncNoise_ct, ncNoise_cc, SR_value, fileName, point);
                }
            }
        }
    }
    return 0;
}