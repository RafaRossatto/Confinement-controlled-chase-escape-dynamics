#include "obstacle.h"
#include "cell.h"
#include "recurso.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <random>
#include <tuple>
#include <set>
#include <algorithm>
#include <sstream>

const int NORTH = 0;          /**< Direction: North */
const int EAST = 1;           /**< Direction: East */
const int SOUTH = 2;          /**< Direction: South */
const int WEST = 3;           /**< Direction: West */


std::random_device rd;
unsigned int GLOBAL_SEED = rd();

/*

const int NORTHEAST = 4;      /**< Direction: Northeast 
const int SOUTHEAST = 5;      /**< Direction: Southeast 
const int SOUTHWEST = 6;      /**< Direction: Southwest 
const int NORTHWEST = 7;      /**< Direction: Northwest 
*/

const int SIZE = 1e2;    // Tamano do lado da quadrado
const int WIDTH = SIZE;          /**< Size of the grid */
const int HEIGHT = SIZE;          /**< Size of the grid */
const int NUMSTEPS = 6e7;    /**< Number of simulation steps */
const int MAXRECURSOS = 500; // Numero maximo de recursos durante a simulação
const int LIFECYCLE = 7E6; // Numero de passos para uma célula boa morrer de velha


//Parametros que são relevantes


const double APPROBABILITY = 0.00; // probabilidade de apotose
const double REPROBABILITY = 0.00; // probabilidade de ser uma célula ruim a ser recrutada
const double REPROPROBABILITY = 0.00; // probabilidade de se reproduzir
const double HURTPROBABILITY = 0.00; // probabilidade da celula sair machucada 
int SEARCHRADIUS = 40; // raio de procura.
const int REPREENERGY = 500; // energia minina para se reproduzir

const int CAPTUREPROBABILITY = 1.00; // probabilidade de capturar uma célula ruim
const int ENERGY = 1e6; // energia inical de cada elemento

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
                                const std::vector<Recurso>& recursos,
                                const std::vector<Cell>& celulas,
                                const std::vector<Cell>& cancers)
{
    return  sobrepoeComLista(novo,pontos) or
            sobrepoeComLista(novo, recursos) or
            sobrepoeComLista(novo, celulas) or
            sobrepoeComLista(novo, cancers);
}

/*
Fazer o posicionamento antes do inicio da simulação:
*/
/**
 * @brief Checks if a position is occupied by a cell or an obstacle.
 *
 * This function checks if the position specified by the coordinates (x, y) is occupied 
 * by a cell in the celulas or cC vectors, or by an obstacle in the obstaculos vector.
 *
 * @param x The X coordinate of the position to check.
 * @param y The Y coordinate of the position to check.
 * @param celulas Vector of cells that may occupy the position.
 * @param cC Vector of complementary cells that may occupy the position.
 * @param obstaculos Vector of obstacles that may occupy the position.
 * @return true if the position is occupied by a cell or an obstacle, false otherwise.
 */

bool estaOcupado(int x, int y, const std::vector<Cell>& celulas,const std::vector<Cell>& cC, 
const std::vector<Obstacle>& obstaculos,int teste) 
{
    
    // inicia a verificão se o ponto está ocupado por outra célula de proteção
    
    for (const auto& celula : celulas) 
    {
    if (celula.getCoordenadaX() == x && celula.getCoordenadaY() == y) 
        {
            return true;
        }
    }
    
    if(teste == 1)
    {
        for (const auto& c_celula : cC) 
        {
        if (c_celula.getCoordenadaX() == x && c_celula.getCoordenadaY() == y) 
            {
                return true;
            }
        }
    }   
    // inicia a verificão se o ponto está ocupado por um obstaculo
    for (const auto& obstaculo : obstaculos) 
    {
        if (obstaculo.getCoordenadaX() == x && obstaculo.getCoordenadaY() == y) 
        {
            return true;
        }
    }
        return false;
}

/**
 * @brief Positions obstacles, resources, cells, and cancer cells within specified bounds.
 *
 * This function attempts to position a specified number of obstacles (pontos), resources (recursos),
 * cells (celulas), and cancer cells (cancers) randomly within the specified maximum width and height.
 * It ensures that each object does not overlap with existing objects using the verificaSobreposicao function.
 *
 * @param pontos Vector to store obstacles.
 * @param recursos Vector to store resources.
 * @param celulas Vector to store cells.
 * @param cancers Vector to store cancer cells.
 * @param num_pontos Number of obstacles to position.
 * @param num_recursos Number of resources to position.
 * @param num_celulas Number of cells to position.
 * @param num_cancers Number of cancer cells to position.
 * @param largura_max Maximum width within which to position objects (exclusive).
 * @param altura_max Maximum height within which to position objects (exclusive).
 * @return true if all objects (obstacles, resources, cells, and cancer cells) are successfully positioned;
 *         false if there is a failure to position any type of object within the maximum attempts.
 */
 bool posicionarObjetos(std::vector<Obstacle>& pontos, std::vector<Recurso>& recursos, 
    std::vector<Cell>& celulas, std::vector<Cell>& cancers, int num_pontos, int num_recursos, 
    int num_celulas, int num_cancers, int largura_max, int altura_max,std::mt19937& rng) 
    {
        int max_tentativas = 10000;
        if (num_pontos != 0) 
        {
            logInfo(" Obstaculos já carregados: " + pontos.size());
            // Não fazer mais nada com obstáculos
        } else {
            // Se num_pontos for 0, criar obstáculos aleatoriamente
            int tentativas_pontos = 0;
            while (tentativas_pontos < max_tentativas && pontos.size() < num_pontos) {
                std::uniform_int_distribution<int> distribX(0, largura_max - 1);
                int x = distribX(rng);
                std::uniform_int_distribution<int> distribY(0, altura_max - 1);
                int y = distribY(rng);
                int novo_id = pontos.size() + 1;
                Obstacle novo_ponto = {"C", novo_id, x, y};
    
                if (!verificaSobreposicaoGeral(novo_ponto, pontos, recursos, celulas,{})) {
                    pontos.push_back(novo_ponto);
                    tentativas_pontos = 0;
                } else {
                    tentativas_pontos++;
                }
            }
        }
    
        // Posicionar recursos
        int tentativas_recursos = 0;
        while (tentativas_recursos < max_tentativas && recursos.size() < num_recursos) {
            std::uniform_int_distribution<int> distribX(0, largura_max - 1);
            int x = distribX(rng);
            std::uniform_int_distribution<int> distribY(0, altura_max - 1);
            int y = distribY(rng);
            int novo_id = recursos.size() + 1;
            Recurso novo_recurso = {"P", novo_id, x, y, ENERGY};
    
            if (!verificaSobreposicaoGeral(novo_recurso, pontos, recursos, celulas,cancers)) {
                recursos.push_back(novo_recurso);
                tentativas_recursos = 0;
            } else {
                tentativas_recursos++;
            }
        }
    
        // Posicionar células
        int tentativas_celulas = 0;
        while (tentativas_celulas < max_tentativas && celulas.size() < num_celulas) {
            std::uniform_int_distribution<int> distribX(0, largura_max - 1);
            int x = distribX(rng);
            std::uniform_int_distribution<int> distribY(0, altura_max - 1);
            int y = distribY(rng);
            int novo_id = celulas.size() + 1;
            Cell nova_celula = {"N", novo_id, x, y, ENERGY};
    
            if (!verificaSobreposicaoGeral(nova_celula, pontos, recursos, celulas,cancers)) {
                celulas.push_back(nova_celula);
                tentativas_celulas = 0;
            } else {
                tentativas_celulas++;
            }
        }
    
        // Posicionar células cancerígenas
        int tentativas_cancers = 0;
        while (tentativas_cancers < max_tentativas && cancers.size() < num_cancers) {
            std::uniform_int_distribution<int> distribX(0, largura_max - 1);
            int x = distribX(rng);
            std::uniform_int_distribution<int> distribY(0, altura_max - 1);
            int y = distribY(rng);
            int novo_id = cancers.size() + 1;
            Cell nova_cancer = {"O", novo_id, x, y, ENERGY};
    
            if (!verificaSobreposicaoGeral(nova_cancer, pontos, recursos, celulas, cancers)) {
                cancers.push_back(nova_cancer);
                tentativas_cancers = 0;
            } else {
                tentativas_cancers++;
            }
        }
    
        // Verificação final de sucesso
        if (recursos.size() == num_recursos && celulas.size() == num_celulas && cancers.size() == num_cancers) {
            return true;
        } else {
            return false;
        }
    }



bool newsCancerCell(const std::vector<Recurso>& Recurso, const std::vector<Obstacle>& pontos,
const std::vector<Cell>& celula, std::vector<Cell>& cancer,
int cancerEnergy, int nCancer,std::mt19937& rng)
{
    //srand(time(0)); // Inicializa a semente de números aleatórios uma vez
    int max_tentativas = 100000;

    int recursos_iniciais = cancer.size(); // Número de recursos já no vetor
    int total_recursos = recursos_iniciais + nCancer; // Total desejado após adição

    for (int i = recursos_iniciais; i < total_recursos; ++i) 
    {
        bool posicionado = false;
        for (int tentativa = 0; tentativa < max_tentativas && !posicionado; ++tentativa) 
        {
            std::uniform_int_distribution<int> distribX(0, WIDTH - 1);
            int x = distribX(rng);
            std::uniform_int_distribution<int> distribY(0, HEIGHT - 1);
            int y = distribY(rng);
            Cell nova_celula = {"O", i+1, x, y, cancerEnergy}; // Assume ENERGY é um valor definido

            if (!verificaSobreposicaoGeral(nova_celula, pontos, Recurso, celula, cancer)) 
            {
                cancer.push_back(nova_celula);
                posicionado = true;
            }
        }

        if (!posicionado) 
        {
            return false; // Falha ao posicionar todos os novos recursos necessários
        }
    }

    return true; // Todos os recursos foram posicionados com sucesso
}


bool newsCell(const std::vector<Recurso>& Recurso, const std::vector<Obstacle>& pontos,
std::vector<Cell>& celula,const std::vector<Cell>& cancer,
const  int cellEnergy,const int nCell,std::mt19937& rng)
{
    //srand(time(0)); // Inicializa a semente de números aleatórios uma vez
    int max_tentativas = 100;


    int recursos_iniciais = celula.size(); // Número de recursos já no vetor
    int total_recursos = recursos_iniciais + nCell; // Total desejado após adição

    for (int i = recursos_iniciais; i < total_recursos; ++i) 
    {
        bool posicionado = false;
        for (int tentativa = 0; tentativa < max_tentativas && !posicionado; ++tentativa) 
        {
            std::uniform_int_distribution<int> distribX(0, WIDTH - 1);
            int x = distribX(rng);

        
            std::uniform_int_distribution<int> distribY(0, HEIGHT - 1);
            int y = distribY(rng);
            Cell nova_celula = {"N", i+1, x, y, cellEnergy}; // Assume ENERGY é um valor definido

            if (!verificaSobreposicaoGeral(nova_celula, pontos, Recurso, celula, cancer)) 
            {
                celula.push_back(nova_celula);
                posicionado = true;       
            }
        }

        if (!posicionado) 
        {
            return false; // Falha ao posicionar todos os novos recursos necessários
        }
    }

    return true; // Todos os recursos foram posicionados com sucesso
}


/**
 * @brief Moves coordinates (x, y) towards a target position (targetX, targetY).
 *
 * This function updates the coordinates (x, y) to move towards the target position
 * (targetX, targetY) based on the following rules:
 * - Increment x if it is less than targetX, wrapping around WIDTH if necessary.
 * - Decrement x if it is greater than targetX, wrapping around WIDTH if necessary.
 * - Increment y if it is less than targetY, wrapping around HEIGHT if necessary.
 * - Decrement y if it is greater than targetY, wrapping around HEIGHT if necessary.
 *
 * @param x Reference to the x coordinate to be updated.
 * @param y Reference to the y coordinate to be updated.
 * @param targetX Target x coordinate to move towards.
 * @param targetY Target y coordinate to move towards.
 */
void moveTowardsPoint(int &x, int &y, int targetX, int targetY) 
{
    // Se a posição X ainda não está alinhada com o alvo, mover primeiro no eixo X
    if (x != targetX) 
    {
        if (x < targetX) 
        {
            x = (x + 1) % WIDTH; // Move para a direita
        } 
        else 
        {
            x = (x - 1 + WIDTH) % WIDTH; // Move para a esquerda
        }
    }
    // Se a posição X já está alinhada, mover no eixo Y
    else if (y != targetY) 
    {
        if (y < targetY) 
        {
            y = (y + 1) % HEIGHT; // Move para cima
        } 
        else 
        {
            y = (y - 1 + HEIGHT) % HEIGHT; // Move para baixo
        }
    }
}

/**
 * @brief Moves coordinates (x, y) based on a specified direction.
 *
 * This function updates the coordinates (x, y) based on the specified direction (move), 
 * which determines the direction of movement:
 * - NORTH moves y coordinate up.
 * - NORTHEAST moves x coordinate left and y coordinate up.
 * - EAST moves x coordinate right.
 * - SOUTHEAST moves x coordinate right and y coordinate down.
 * - SOUTH moves y coordinate down.
 * - SOUTHWEST moves x coordinate left and y coordinate down.
 * - WEST moves x coordinate left.
 * - NORTHWEST moves x coordinate right and y coordinate up.
 *
 * Coordinates are wrapped around the WIDTH and HEIGHT limits to simulate a toroidal grid.
 *
 * @param x Reference to the x coordinate to be updated.
 * @param y Reference to the y coordinate to be updated.
 * @param move Direction of movement (NORTH, NORTHEAST, EAST, SOUTHEAST, SOUTH, SOUTHWEST, WEST, NORTHWEST).
 */

void randonWalk(int& x, int& y,int move)
{
    switch (move) 
    {
        case NORTH:
            y = (y+1)%HEIGHT;
            break;
        
        case EAST:
            x=(x+1)% WIDTH;
            break;
        case SOUTH:
            y = (y - 1 + HEIGHT) % HEIGHT;
            break;
        case WEST:
            x = (x - 1 + WIDTH) % WIDTH;
            break;
    }
}

/**
 * @brief Opens a file in append mode.
 *
 * This function opens the specified file fileName in append mode (std::ios::app).
 * If the file cannot be opened, an error message is printed to std::cerr.
 *
 * @param fileName The name of the file to be opened.
 */
void openFile(const std::string& fileName) 
{
    // Abre o arquivo em modo de escrita e apêndice
    std::ofstream file(fileName, std::ios::app);

    // Check if the file was opened successfully
    if (!file.is_open()) 
    {
        logError("Erro ao abrir o arquivo " + fileName);
        return;
    }

    file.close();
}

/**
 * @brief Writes data from vectors of different types of cells and objects to a file.
 *
 * This function opens the specified file fileName in append mode (std::ios::app) and writes
 * information about cells and objects of types a, b, c, and d to the file. It writes the total
 * number of entries and a header indicating the timestep before writing each entry's details.
 *
 * @param fileName The name of the file to which data will be written.
 * @param a Vector containing objects of type Cell to be written to the file.
 * @param d Vector containing objects of type Cell to be written to the file.
 * @param b Vector containing objects of type Obstacle to be written to the file.
 * @param c Vector containing objects of type Recurso to be written to the file.
 * @param timeStep The current timestep number to be included in the file header.
 */
void writeToFile(const std::string& fileName, const std::vector<Cell>&a,
const std::vector<Cell>&d,const std::vector<Obstacle>&b,
const std::vector<Recurso>&c, int timeStep) 
{
    // Abre o arquivo em modo de escrita e apêndice
    std::ofstream file(fileName, std::ios::app);

    // Verifica se o arquivo foi aberto corretamente
    if (!file.is_open()) 
    {
        logError("Erro ao abrir o arquivo " + fileName);
        return;
    }
    // Write the total number of entries (sum of sizes of all vectors)
    file << a.size()+b.size()+c.size() +d.size() << '\n';
    // Write the header indicating the timestep
    file << "Atoms. Timestep: " << timeStep << '\n';

    // Write information for each object in vector 'a'
    for (const auto& teste: a) 
    {
        file << teste.getTipo() << ' '<< teste.getCoordenadaX() << ' ' 
        << teste.getCoordenadaY() <<' '<< 0.0 <<' ' << std::endl;
    }

    // Write information for each object in vector 'b'
    for (const auto& teste: b) 
    {
        file << teste.getTipo() << ' '<< teste.getCoordenadaX() << ' ' 
        << teste.getCoordenadaY() <<' '<< 0.0 <<' ' << std::endl;
    }
    
    // Write information for each object in vector 'c'
    for (const auto& teste: c) 
    {
        file << teste.getTipo() << ' '<< teste.getCoordenadaX() << ' ' 
        << teste.getCoordenadaY() <<' '<< 0.0 <<' ' << std::endl;
    }

    // Write information for each object in vector 'd'
    for (const auto& teste: d) 
    {
        file << teste.getTipo() << ' '<< teste.getCoordenadaX() << ' ' 
        << teste.getCoordenadaY() <<' '<< 0.0 <<' ' << std::endl;
    }
    
    // Close the file
    file.close();
}


void movePosition(int& x, int& y, int targetX, int targetY, int nProbability,
    std::vector<Cell>& celulas, std::vector<Cell>& cC, std::vector<Obstacle>& obstaculos,
    int teste, bool isPursuing,std::mt19937& rng ) // novo parâmetro para indicar se é aproximação ou fuga
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
                    if (isPursuing = true)
                    {
                        // Perseguir: mover na direção que diminua a diferença de y
                        if (y < targetY)
                            direcoesPossiveis.push_back(NORTH); // aumenta y
                        else
                            direcoesPossiveis.push_back(SOUTH); // diminui y
                    }
                    else
                    {
                        // Fugir: evitar o movimento que aproxima (o contrário do que reduziria a diferença)
                        if (y < targetY)
                            direcoesPossiveis = {EAST, WEST, SOUTH}; // NÃO usar NORTH
                        else
                            direcoesPossiveis = {EAST, WEST, NORTH}; // NÃO usar SOUTH
                    }
                }
                else if (y == targetY) // Mesma linha: o movimento será horizontal
                {
                    if (isPursuing)
                    {
                        if (x < targetX)
                            direcoesPossiveis.push_back(EAST); // aumenta x
                        else
                            direcoesPossiveis.push_back(WEST); // diminui x
                    }
                    else
                    {
                        if (x < targetX)
                            direcoesPossiveis = {NORTH, SOUTH, WEST}; // NÃO usar EAST
                        else
                            direcoesPossiveis = {NORTH, SOUTH, EAST}; // NÃO usar WEST
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
                        direcoesPossiveis.push_back(EAST);
                    else if (dx < 0)
                        direcoesPossiveis.push_back(WEST);
                    if (dy > 0)
                        direcoesPossiveis.push_back(NORTH);
                    else if (dy < 0)
                        direcoesPossiveis.push_back(SOUTH);
                }
                else
                {
                    // Fugir: inverte os sinais para aumentar a distância
                    if (dx > 0)
                        direcoesPossiveis.push_back(WEST);
                    else if (dx < 0)
                        direcoesPossiveis.push_back(EAST);
                    if (dy > 0)
                        direcoesPossiveis.push_back(SOUTH);
                    else if (dy < 0)
                        direcoesPossiveis.push_back(NORTH);
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

        if (!estaOcupado(newX, newY, celulas, cC, obstaculos, teste))
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

/**
 * @brief Calculates the Euclidean distance between two points (x1, y1) and (x2, y2).
 *
 * This function computes the Euclidean distance between two points in a 2D Cartesian coordinate system.
 *
 * @param x1 X-coordinate of the first point.
 * @param y1 Y-coordinate of the first point.
 * @param x2 X-coordinate of the second point.
 * @param y2 Y-coordinate of the second point.
 * @return The Euclidean distance between the points (x1, y1) and (x2, y2).
 */

double calculateDistance(int x1, int y1, int x2, int y2) {
    int dx = x2 - x1;
    if (dx < 0) dx = -dx;
    if (dx > WIDTH / 2) dx = WIDTH - dx;

    int dy = y2 - y1;
    if (dy < 0) dy = -dy;
    if (dy > HEIGHT / 2) dy = HEIGHT - dy;

    return dx + dy;
}

/*
 * @brief Runs the simulation of the system.
 * 
 * @param NUMSTEPS Maximum number of simulation steps.
 * @param cT Vector of type T cells.
 * @param cC Vector of cancer cells.
 * @param point Vector of obstacles.
 * @param food Vector of resources (food).
 * @param fileName Name of the output file where results will be written.
 */
int runSimulationPaper(const int NUMSTEPS, std::vector<Cell>& cT, 
std::vector<Cell>& cC, std::vector<Obstacle>& point, 
std::vector<Recurso>& food, const std::string& fileName, std::mt19937& rng, bool write = false) 
{
    int steps = 1; // Step counter
    bool isPursuing;
    // Search radius to find resources or cells
    
    // Initialize random number generator and Bernoulli distribution

    // Main simulation loop
    while (steps < NUMSTEPS) 
    {   
        std::bernoulli_distribution d11(REPROBABILITY);

        // Process cells of type T
        if (!cT.empty())
        {
            for (int i = cT.size() - 1; i >= 0; --i)
            {
                int x = cT[i].getCoordenadaX(); 
                int y = cT[i].getCoordenadaY();
                double minDistance = std::numeric_limits<double>::max();
                int nearestFoodIndex = -1;

                // Find nearest food resource
                for (int j = 0; j < food.size(); j++) 
                {
                    double distance = calculateDistance(x, y, food[j].getCoordenadaX(), food[j].getCoordenadaY());
                    if (distance < minDistance && distance <= SEARCHRADIUS) 
                    {
                        minDistance = distance;
                        nearestFoodIndex = j;
                    }
                }

                // Find nearest cancer cell
                int nearestCCIndex = -1;
                double minCCDistance = std::numeric_limits<double>::max();
                for (int j = cC.size() - 1; j >= 0; --j) 
                {
                    double distance = calculateDistance(x, y, cC[j].getCoordenadaX(), cC[j].getCoordenadaY());
                    if (distance < minCCDistance && distance <= SEARCHRADIUS) 
                    {
                        minCCDistance = distance;
                        nearestCCIndex = j;
                    }
                }

                // Decide whether to pursue cancer cell or food
                bool pursueCC = (cT[i].getEnergy() >= cT[i].getMinEnergy()) && (nearestCCIndex != -1);
                bool pursueFood = (cT[i].getEnergy() < cT[i].getMinEnergy()) && (nearestFoodIndex != -1);

                std::bernoulli_distribution d(CTPROBABILITY);
                if (pursueCC)
                {
                    bool isPursuing = true;
                    std::bernoulli_distribution d(CTPROBABILITY);
                    int randomValue = d(rng) ? 1 : 0;
                    movePosition(x, y, cC[nearestCCIndex].getCoordenadaX(), cC[nearestCCIndex].getCoordenadaY(), 
                    randomValue, cT, cC, point,randomValue,isPursuing,rng);
                    cT[i].changePosition(x, y);
                }
                else if (pursueFood)
                {
                    int randomValue = d(rng) ? 1 : 0;
                    movePosition(x, y, food[nearestFoodIndex].getCoordenadaX(), food[nearestFoodIndex].getCoordenadaY(), 
                    randomValue, cT, cC, point,randomValue,isPursuing,rng);
                    cT[i].changePosition(x, y);
                }
                else
                {
                    // Random walk
                    bool isPursuing = false;
                    int randomValue = 0;
                    movePosition(x, y, 0, 0, randomValue, cT, cC, point,0,isPursuing, rng);
                    cT[i].changePosition(x, y);
                }

                // Decrease energy
                cT[i].decrementEnergy();

                // Check if the cell reached the food resource
                if (nearestFoodIndex != -1 && x == food[nearestFoodIndex].getCoordenadaX() && y == food[nearestFoodIndex].getCoordenadaY())
                {
                    // Increase energy and remove food
                    cT[i].increaseEnergy(food[nearestFoodIndex].getEnergy());
                    food.erase(food.begin() + nearestFoodIndex);

                    // Cell reproduction based on energy
                    std::bernoulli_distribution d(REPROPROBABILITY);
                    int randomValue = d(rng) ? 1 : 0;
                    if (cT[i].getEnergy() >= REPREENERGY && randomValue == 1 && cT[i].getDivision() <= 50)
                    {
                        newsCell(food, point, cT, cC, cT[i].getEnergy(), 1, rng);
                        cT[i].increaseDivision();
                    }
                }

                // Check for capture of cancer cell
                if (nearestCCIndex != -1 && x == cC[nearestCCIndex].getCoordenadaX() && y == cC[nearestCCIndex].getCoordenadaY())
                {
                    
                    std::bernoulli_distribution d(CAPTUREPROBABILITY);
                    int capture = d(rng) ? 1 : 0;
                   
                    std::bernoulli_distribution d1(HURTPROBABILITY);
                    int hurt = d1(rng) ? 1 : 0;

                    if (capture == 1)
                    {
                        cC.erase(cC.begin() + nearestCCIndex);
                        if (hurt == 1)
                        {
                            cT[i].cellHurt();
                        }
                    }
                    else if (hurt == 1)
                    {
                        cT[i].cellHurt();
                    }
                }
            }
        }

        // Process cancer cells
        if (!cC.empty())
        {
            for (int i = cC.size() - 1; i >= 0; --i)
            {
                int x = cC[i].getCoordenadaX(); 
                int y = cC[i].getCoordenadaY();

                // Find nearest food
                double minDistance = std::numeric_limits<double>::max();
                int nearestFoodIndex = -1;
                for (int j = 0; j < food.size(); j++)
                {
                    double distance = calculateDistance(x, y, food[j].getCoordenadaX(), food[j].getCoordenadaY());
                    if (distance < minDistance && distance <= SEARCHRADIUS)
                    {
                        minDistance = distance;
                        nearestFoodIndex = j;
                    }
                }

                // Find nearest type T cell
                int nearestCTIndex = -1;
                double minCTDistance = std::numeric_limits<double>::max();
                for (int j = cT.size() - 1; j >= 0; --j)
                {
                    double distance = calculateDistance(x, y, cT[j].getCoordenadaX(), cT[j].getCoordenadaY());
                    if (distance < minCTDistance && distance <= SEARCHRADIUS)
                    {
                        minCTDistance = distance;
                        nearestCTIndex = j;
                    }
                }

                // Decide whether to escape or pursue food
                bool escapeCT = (cC[i].getEnergy() >= cC[i].getMinEnergy()) && (nearestCTIndex != -1);
                bool pursueFood = (cC[i].getEnergy() < cC[i].getMinEnergy()) && (nearestFoodIndex != -1);

                std::bernoulli_distribution d(CCPROBABILITY);
                int randomValue = d(rng) ? 1 : 0;
                
                if (escapeCT)
                {
                    bool isPursuing = false;
                    int randomValue = d(rng) ? 1 : 0;
                    movePosition(x, y, -cT[nearestCTIndex].getCoordenadaX(), -cT[nearestCTIndex].getCoordenadaY(), randomValue, cT, cC, point,randomValue,isPursuing,rng);
                    cC[i].changePosition(x, y);
                }
                else if (pursueFood)
                {
                    int randomValue = d(rng) ? 1 : 0;
                    movePosition(x, y, food[nearestFoodIndex].getCoordenadaX(), food[nearestFoodIndex].getCoordenadaY(), randomValue, cT, cC, point,randomValue,isPursuing,rng);
                    cC[i].changePosition(x, y);
                }
                else
                {
                    // Random walk
                    int randomValue = 0;
                    movePosition(x, y, 0, 0, randomValue, cT, cC, point,1,isPursuing,rng);
                    cC[i].changePosition(x, y);
                }

                // Decrease energy of cancer cell
                cC[i].decrementEnergy();

                // Check if cancer cell reached food
                if (nearestFoodIndex != -1 && x == food[nearestFoodIndex].getCoordenadaX() && y == food[nearestFoodIndex].getCoordenadaY())
                {
                    cC[i].increaseEnergy(food[nearestFoodIndex].getEnergy());
                    food.erase(food.begin() + nearestFoodIndex);
                    
                    std::bernoulli_distribution d(REPROPROBABILITY);
                    int randomValue = d(rng) ? 1 : 0;
                    if (cC[i].getEnergy() >= REPREENERGY && randomValue == 1)
                    {
                        newsCancerCell(food, point, cT, cC, cC[i].getEnergy(), 1, rng);
                    }
                }

                // Check for cell death
                if (cC[i].getEnergy() <= 0 || cC[i].getLifeCycle() >= 2e5)
                {
                    cC.erase(cC.begin() + i);
                }
            }
        }

        if (write == true)
        {
        writeToFile(fileName, cT,cC, point, food, steps);
        }
        // Terminate if no cancer cells remain
        if (cC.empty()) 
        {
            logInfo("No more cancer cells. Terminating progam");
            break;
        }

        steps++; // Increment step count
        //std::cout << "Passo: " << steps << std::endl;
    }

    return steps;
}

bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good(); // Verifica se o arquivo pode ser aberto
}

bool carregarObstaculos(std:: vector<Obstacle>& point, int numPoint, std:: string& line)
{
    if (numPoint == 0)
    {
        return true; // Faz nada
    }
    
    std:: string filename = " obstacules_" + std::to_string(numPoint)+ ".txt";
    if (!fileExists(filename))
    {
        std::cerr<< "Erro Arquivo " << filename << " não existe.\n";
        return false; 
    }

    std:: ifstream inputFile(filename);
    if (!inputFile.is_open())
    {
        std:: cerr << "Erro: Não foi possível abrir o arquivo" << filename << "\n";
        return false; 
    }
    int id_counter = 1;
    int x,y;
    
    while (std::getline(inputFile,line))
    {
        std:: istringstream lineStream(line);
        if (lineStream >> x >> y)
        {
            Obstacle new_obstacle = {"C", id_counter ++, x,y};
            point.push_back(new_obstacle);
        }
        
    }
    inputFile.close();
    return true;
    
    
    
}



std::string gerarNomeArquivo(int numCT, int numcC, int numPoint,
    double ncNoise_cc, double ncNoise_ct,
    int SR_value) {
return "NC_" + std::to_string(numCT) +
"_NE_" + std::to_string(numcC) +
"_O_" + std::to_string(numPoint) +
"_TCC_" + std::to_string(ncNoise_cc) +
"_TCT_" + std::to_string(ncNoise_ct) +
"_SR_" + std::to_string(SR_value) + ".dat";
}



void executarRodadas(int count, int numCT, int numcC, int numPoint,
    double ncNoise_ct, double ncNoise_cc, int SR_value,
    const std::string& fileName, const std::vector<Obstacle>& point, bool write)
{
    std::ofstream outputFile(fileName);
    if (!outputFile.is_open()) 
    {
    logError("Erro ao abrir o arquivo " + fileName);
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
        std::vector<Recurso> recursos_local;

        if (!posicionarObjetos( point_local, recursos_local, cT_local, cC_local,
                                numPoint, 0, numCT, numcC, 100, 100, rng_local)) 
                                {
                                    #pragma omp critical
                                    logError("Erro ao posicionar objetos na execução " + std::to_string(run));
                                    continue;
                                }

        std::string trajectoryFileName;
        if (write) 
        {
            trajectoryFileName= "Trajectory_NH_500_NE_" + std::to_string(numcC) + "_O_" +
                                std::to_string(numPoint) + "_TCC_" +
                                std::to_string(ncNoise_cc) + "_TCT_" +
                                std::to_string(ncNoise_ct) + "_SR_" +
                                std::to_string(SR_value) + "_Run_" + std::to_string(run) + ".xyz";
        }

        int steps_local = runSimulationPaper(10000, cT_local, cC_local, point_local,
                                            recursos_local, trajectoryFileName, rng_local, write);

        #pragma omp critical
        {
            outputFile << run << "," << steps_local << "," << seed_run << "\n";
            std::cout << "Execução " << run << ", Número de passos: " << steps_local << ", Número da seed: " 
            << seed_run << std::endl;
            logInfo("Execução " + std::to_string(run)+
                    ", Número de pasos: " + std::to_string(steps_local)+
                    ", Número da seed " + std::to_string(seed_run));
        }
    }
    
    outputFile.close();
    logInfo("Resultados salvos no arquivo " + fileName);
}

void logInfo(const std::string& mensagem) 
{
    std::cout << "[INFO] " << mensagem << std::endl;
}

void logWarning(const std::string& mensagem) 
{
    std::cout << "[WARNING] " << mensagem << std::endl;
}

void logError(const std::string& mensagem) 
{
    std::cerr << "[ERROR] " << mensagem << std::endl;
}


int main() 
{
    int count, steps, numCT, numcC, numPoint, numRecursos, SR_value, x, y;
    bool write;

    //std::vector<int> numcC_values = {1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 500, 1000};
    std::vector<int> numcC_values = {1};
    std::vector<int> numPoint_values = {0, 2, 3, 5, 6, 11, 21, 26, 51};
    //std::vector<int> SR_values = {5, 10, 20, 30, 40, 50};
    std::vector<double> numNoise_ct = {0.05};
    std::vector<double> numNoise_cc = {0.05};

    write = false;
    std::vector<Cell> cT;
    std::vector<Cell> cC;
    std::vector<Obstacle> point;
    std::vector<Recurso> recursos;
    std::string line;

    numCT = 500;
    numRecursos = 0;
    count = 1e3;
    SR_value = 5;
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
                    recursos.clear();

                    if (!carregarObstaculos(point, numPoint, line)) 
                    {
                        continue; // Se não carregar, pula para o próximo
                    }
                    std::string fileName = gerarNomeArquivo(numCT, numcC, numPoint, ncNoise_cc, ncNoise_ct, SR_value);
                    executarRodadas(count, numCT, numcC, numPoint, ncNoise_ct, ncNoise_cc, SR_value, fileName, point, write);
                }
            }
        }
    }
    return 0;
}