#include "obstacle.h"
#include "cell.h"
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
const int NORTH = 0;          /**< Direction: North */
const int EAST = 1;           /**< Direction: East */
const int SOUTH = 2;          /**< Direction: South */
const int WEST = 3;           /**< Direction: West */

const int SIZE = 100;
const int WIDTH = SIZE;
const int HEIGHT = SIZE;

const double CAPTUREPROBABILITY = 1.00;

// Raio progressivo utilizado na busca
const std::vector<int> RAIOS_PROGRESSIVOS = {5, 10, 20, 30, 40, 50};




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


void logInfo(const std::string& mensagem) 
{
    std::cout << "[INFO] " << mensagem << "\n";
}

void logWarning(const std::string& mensagem) 
{
    std::cout << "[WARNING] " << mensagem << "\n";
}

void logError(const std::string& mensagem) 
{
    std::cerr << "[ERROR] " << mensagem << "\n";
}

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










// Função para calcular distância considerando condições periódicas
double calculeDistance(int x1, int y1, int x2, int y2) {
    int dx = x2 - x1;
    if (dx < 0) dx = -dx;
    if (dx > WIDTH / 2) dx = WIDTH - dx;

    int dy = y2 - y1;
    if (dy < 0) dy = -dy;
    if (dy > HEIGHT / 2) dy = HEIGHT - dy;

    return dx + dy;
}

// Função para verificar se um índice é válido
bool indiceValido(int indice, int tamanho)
{
    return (indice >= 0) && (indice < tamanho);
}

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


bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good(); // Verifica se o arquivo pode ser aberto
}


std::pair<int,int> encontrarAlvosMaisProximo(int x,int y,
                                            const std:: vector<Cell>& alvos,
                                            int searchRadius)
{
    int nearestIndex = -1;
    int minDistance = std:: numeric_limits<int>::max();

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
            // Achou alvo próximo
    
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
        int max_tentativas = 100;
        if (num_pontos != 0) 
        {
            logInfo(" Obstaculos já carregados: " + std:: to_string(pontos.size()));
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
            Cell nova_celula = {"N", novo_id, x, y};
    
            if (!verificaSobreposicaoGeral(nova_celula, pontos, celulas,cancers)) {
                celulas.push_back(nova_celula);
                tentativas_celulas = 0;
            } else {
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
    
            if (!verificaSobreposicaoGeral(nova_cancer, pontos, celulas, cancers)) {
                cancers.push_back(nova_cancer);
                tentativas_cancers = 0;
            } else {
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


void writeToFile(const std::string& fileName, const std::vector<Cell>&a,
const std::vector<Cell>&d,const std::vector<Obstacle>&b, int timeStep) 
{
    // Abre o arquivo em modo de escrita e apêndice
    // Se eu deixar assim ele pega o nome, se eu tiro as linhas cout e cin ele não pega o nome, verificar por que isso 
    // está ocorrendo
    
    std::ofstream file(fileName, std::ios::app);
    
    //std:: cout << " esta aqui"<< fileName ;
    //std:: cin.get();
    
    // Verifica se o arquivo foi aberto corretamente
    if (!file.is_open()) 
    {
        std:: cout << " esta aqui erro"<< fileName ;
        std:: cin.get();
        logError(" Erro1111 ao abrir o arquivo " + fileName);
        //break;
    }
    // Write the total number of entries (sum of sizes of all vectors)
    file << a.size()+b.size() +d.size() << '\n';
    // Write the header indicating the timestep
    file << "Atoms. Timestep: " << timeStep << '\n';

    // Write information for each object in vector 'a'
    for (const auto& teste: a) 
    {
        file << teste.getTipo() << ' '<< teste.getCoordenadaX() << ' ' 
        << teste.getCoordenadaY() <<' '<< 0.0 <<' ' << "\n";
    }

    // Write information for each object in vector 'b'
    for (const auto& teste: b) 
    {
        file << teste.getTipo() << ' '<< teste.getCoordenadaX() << ' ' 
        << teste.getCoordenadaY() <<' '<< 0.0 <<' ' << "\n";
    }
    
    // Write information for each object in vector 'd'
    for (const auto& teste: d) 
    {
        file << teste.getTipo() << ' '<< teste.getCoordenadaX() << ' ' 
        << teste.getCoordenadaY() <<' '<< 0.0 <<' ' << "\n";
    }
    
    // Close the file
    file.close();
}

int runSimulationPaper(const int NUMSTEPS, std::vector<Cell>& cT, 
std::vector<Cell>& cC, std::vector<Obstacle>& point, const std::string& fileName, std::mt19937& rng, bool write = true) 
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
        if (write == true)
        {
            writeToFile(fileName, cT,cC, point, steps);
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


//19,46,4132088286
//82,56,4132088349
//15,35,1113190909
int main() 
{
    // Parâmetros do caso específico
    int run = 82;
    int numCT = 500;
    int numcC = 500; // ajuste conforme o caso do .dat
    int numPoint = 21; // idem
    double ncNoise_ct = 0.05;
    double ncNoise_cc = 0.95;
    int SR_value = 50;
    unsigned int seed_run =1053522277; // do .dat

    setCTPROBABILITY(ncNoise_ct);
    setCCPROBABILITY(ncNoise_cc);
    setSEARCHRADIUS(SR_value);

    bool write = true;
    std::mt19937 rng_local(seed_run);


    std::vector<Cell> cT;
    std::vector<Cell> cC;
    std::vector<Obstacle> point;
    std::string line;
    int x, y;
         // Se houver obstáculos
        if (numPoint != 0) {
            std::string filename = "obstacules_" + std::to_string(numPoint) + ".txt";
            if (!fileExists(filename)) {
                std::cerr << "Erro: Arquivo " << filename << " não existe.\n";
                return 1;
            }
            std::ifstream inputFile(filename);
            int id_counter = 1;
            while (std::getline(inputFile, line)) {
                std::istringstream lineStream(line);
                if (lineStream >> x >> y) {
                    Obstacle new_obstacle = {"C", id_counter++, x, y};
                    point.push_back(new_obstacle);
                }
            }
            inputFile.close();
        }
    // Posiciona todos os elementos
    if (!posicionarObjetos(point, cT, cC, numPoint, numCT,
         numcC, 100, 100,rng_local)) 
    {
        std::cerr << "Erro ao posicionar objetos para run " << run << "\n";
        return 1;
    }


    std::string trajectoryFileName = "Run_Trajectory_NH_500_NE_" + std::to_string(numcC) + "_O_" +
                                     std::to_string(numPoint) + "_TCC_" +
                                     std::to_string(ncNoise_cc) + "_TCT_" +
                                     std::to_string(ncNoise_ct) + "_SR_" +
                                     std::to_string(SR_value) + "_Run_" + std::to_string(run) + ".xyz";

    int steps_local = runSimulationPaper(10000, cT, cC, point, trajectoryFileName,rng_local, true);

    std::cout << "Re-execução completa com " << steps_local << " passos." << "\n";

    return 0;
}