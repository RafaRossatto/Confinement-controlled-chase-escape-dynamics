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
#include <cmath> // necessário para std::round
std::random_device rd;
unsigned int GLOBAL_SEED = rd();

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

int runSimulationPaper(CellLattice& lattice,const int NUMSTEPS, std::vector<Cell>& cT, 
std::vector<Cell>& cC, std::vector<Obstacle>& point, const std::string& fileName, std::mt19937& rng,
bool write,int sr_normal, int sr_cancer) 
{
    int steps = 1; // Step counter
    bool verificacC;

    // Main simulation loop
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


//35,43,0,4117423416
//13,3873,0,4117463394
//82,22180,0,3613368428
//26,17159,0,3101174506
int main() 
{
    SIZE = 128;
    WIDTH = SIZE;
    HEIGHT = SIZE;
    CAPTUREPROBABILITY = 1.0;
    // Parâmetros do caso específico
    int run = 26;
    int numCT = 5;
    int numcC = 10; // ajuste conforme o caso do .dat
    int numPoint = 0; // idem
    double ncNoise_ct = 0.90;
    double ncNoise_cc = 0.05;
    unsigned int seed_run = 1608041491; // do .dat
    CellLattice lattice(WIDTH, HEIGHT);

    setCTPROBABILITY(ncNoise_ct);
    setCCPROBABILITY(ncNoise_cc);
    int sr_cancer = 100;
    int sr_normal = 100;

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
    if (!lattice.placeObjects(point, cT, cC, numPoint, 
        numCT, numcC, rng_local,sr_normal,sr_cancer)) 
        {
            std::cerr << "Erro ao posicionar objetos para run " << run << "\n";
            return 1;
        }



// Verificação de posições iniciais
std::cout << "\n=== VERIFICAÇÃO DE POSIÇÕES INICIAIS ===\n";
bool hasInvalidPositions = false;

// Verificar células normais (cT)
for (const auto& cell : cT) {
    if (cell.getCoordenadaX() < 0 || cell.getCoordenadaX() >= 100 ||
        cell.getCoordenadaY() < 0 || cell.getCoordenadaY() >= 100) {
        std::cerr << "ERRO: Célula NORMAL " << cell.getNumero() 
                  << " em posição inválida (" 
                  << cell.getCoordenadaX() << "," 
                  << cell.getCoordenadaY() << ")\n";
        hasInvalidPositions = true;
    }
}

// Verificar células cancerosas (cC)
for (const auto& cell : cC) {
    if (cell.getCoordenadaX() < 0 || cell.getCoordenadaX() >= 100 ||
        cell.getCoordenadaY() < 0 || cell.getCoordenadaY() >= 100) {
        std::cerr << "ERRO: Célula CANCEROSA " << cell.getNumero() 
                  << " em posição inválida (" 
                  << cell.getCoordenadaX() << "," 
                  << cell.getCoordenadaY() << ")\n";
        hasInvalidPositions = true;
    }
}

// Verificar obstáculos
for (const auto& obs : point) {
    if (obs.getCoordenadaX() < 0 || obs.getCoordenadaX() >= 100 ||
        obs.getCoordenadaY() < 0 || obs.getCoordenadaY() >= 100) {
        std::cerr << "ERRO: Obstáculo " << obs.getNumero() 
                  << " em posição inválida (" 
                  << obs.getCoordenadaX() << "," 
                  << obs.getCoordenadaY() << ")\n";
        hasInvalidPositions = true;
    }
}

if (!hasInvalidPositions) {
    std::cout << "Todas as posições iniciais são válidas!\n";
} else {
    std::cerr << "\nATENÇÃO: Foram encontradas posições iniciais inválidas!\n";
    // Adicione uma pausa para visualizar o erro
    std::cin.get();
    return 1; // Encerra o programa com erro
}

std::cout << "=======================\n\n";
std::cin.get();



    std::string trajectoryFileName = "Run_Trajectory_NH_"+std::to_string(numCT)+"_NE_" + std::to_string(numcC) +
                                    "_O_" + std::to_string(numPoint) +
                                    "_TCC_" + std::to_string(ncNoise_cc) +
                                    "_SR_" + std::to_string(sr_cancer)+
                                    "_TCT_" + std::to_string(ncNoise_ct) +
                                    "_SR_" + std::to_string(sr_normal)+"Run_"+ std:: to_string(run) + ".xyz";

    int steps_local = runSimulationPaper(lattice,50000, cT, cC, point, trajectoryFileName,rng_local, write,sr_normal,sr_cancer);

    std::cout << "Re-execução completa com " << steps_local << " passos." << "\n";

    return 0;
}