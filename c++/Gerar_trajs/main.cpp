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
#include <array>
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

void moverCelulaRuim(CellLattice& lattice, Cell& celula,
    std::vector<Cell>& cT, std::vector<Cell>& cC,
    std::vector<Obstacle>& obstaculos,
    std::mt19937& rng, bool verificacC)
{
const int WIDTH = 100;
const int HEIGHT = 100;

int x = celula.getCoordenadaX();
int y = celula.getCoordenadaY();
int newX = x, newY = y;

// Busca caçadores no raio de procura
std::vector<Cell> celulas(cT.begin(), cT.end());
celulas.insert(celulas.end(), cC.begin(), cC.end());

std::vector<std::string> tipos_alvo = {"N"};
auto alvos = celula.findNearestTarget(celulas, lattice, tipos_alvo);

if (!alvos.empty()) {
// Escolhe aleatoriamente um caçador para fugir
std::uniform_int_distribution<int> dist_alvo(0, alvos.size() - 1);
auto [alvoX, alvoY] = alvos[dist_alvo(rng)];

// Direções possíveis usando o enum Direction
const std::array<Direction, 4> todas_direcoes = {NORTH, EAST, SOUTH, WEST};

// Encontra a direção que maximiza a distância do caçador
Direction melhor_direcao = NORTH;
double maior_distancia = -1.0;




for (Direction dir : todas_direcoes) {
int tempX = x, tempY = y;
celula.randonWalk(tempX, tempY, dir);

double dist = lattice.calculateDistance(tempX, tempY, alvoX, alvoY);
if (dist > maior_distancia) {
maior_distancia = dist;
melhor_direcao = dir;
}
}

// Aplica o movimento de fuga
celula.randonWalk(newX, newY, melhor_direcao);
} 
else {
// Movimento aleatório se não encontrar caçadores
std::uniform_int_distribution<int> dis(0, 3);
Direction dir_aleatoria = static_cast<Direction>(dis(rng));
celula.randonWalk(newX, newY, dir_aleatoria);
std::cerr << "⚠️ Célula " << celula.getNumero() << " fazendo movimento aleatório\n";
}

// Verifica e aplica o movimento se a posição estiver livre
if (!lattice.isOccupied(newX, newY, cT, cC, obstaculos, verificacC)) {
celula.changePosition(newX, newY);
} 
else {
//std::cerr << "⛔ Célula " << celula.getNumero() 
 // << " não pode mover para (" << newX << "," << newY << ")\n";
}
}

void moverCelulaBoa(CellLattice& lattice, Cell& celula,
    std::vector<Cell>& cT,
    std::vector<Cell>& cC,
    std::vector<Obstacle>& obstaculos,
    std::mt19937& rng, bool verificacC)
{
const int WIDTH = 100;
const int HEIGHT = 100;

std::cout << "[INICIO] Célula número " << celula.getNumero()
<< " posição atual: (" << celula.getCoordenadaX() << "," << celula.getCoordenadaY() << ")\n";

int x = celula.getCoordenadaX();
int y = celula.getCoordenadaY();
int newX = x, newY = y;

std::bernoulli_distribution d(CTPROBABILITY);
bool movimentoInteligente = d(rng);

if (movimentoInteligente) 
{
std::vector<Cell> celulas;
celulas.insert(celulas.end(), cT.begin(), cT.end());
celulas.insert(celulas.end(), cC.begin(), cC.end());

// Busca todos os alvos visíveis
auto alvos = celula.findNearestTarget(celulas, lattice, {"O","N"});

// Encontra alvo mais próximo
int menorDist = std::numeric_limits<int>::max();
int alvoX = 0, alvoY = 0;
bool isFugindo = false;
bool encontrou = false;

for (const auto& [ax, ay] : alvos) 
{
for (const auto& agente : celulas) 
{
 if (agente.getCoordenadaX() == ax && agente.getCoordenadaY() == ay) 
 {
     int dist = lattice.calculateDistance(x, y, ax, ay);
     if (agente.getTipo() == "O" && dist < menorDist) 
     {
         menorDist = dist;
         alvoX = ax;
         alvoY = ay;
         isFugindo = false;
         encontrou = true;
     }
     else if (agente.getTipo() == "N" && dist < menorDist) 
     {
         menorDist = dist;
         alvoX = ax;
         alvoY = ay;
         isFugindo = true;
         encontrou = true;
     }
     break;
 }
}
}

if (encontrou) 
{
if (!isFugindo) 
{
 // Persegue presa
 int dx = (alvoX - x + WIDTH) % WIDTH;
 int dy = (alvoY - y + HEIGHT) % HEIGHT;
 
 if (dx > WIDTH/2) dx -= WIDTH;
 if (dy > HEIGHT/2) dy -= HEIGHT;

 if (std::abs(dx) > std::abs(dy)) 
 {
     newX = (x + (dx > 0 ? 1 : -1) + WIDTH) % WIDTH;
     newY = y;
 } 
 else 
 {
     newY = (y + (dy > 0 ? 1 : -1) + HEIGHT) % HEIGHT;
     newX = x;
 }
} 
else 
{
 // Foge de caçador
 const std::array<std::pair<int, int>, 4> direcoes = {{
     {0, 1},   // NORTH
     {1, 0},   // EAST
     {0, -1},  // SOUTH
     {-1, 0}   // WEST
 }};

 double maiorDistancia = -1e9;
 std::pair<int, int> melhorDirecao = {0, 0};

 for (const auto& dir : direcoes) 
 {
     int tempX = (x + dir.first + WIDTH) % WIDTH;
     int tempY = (y + dir.second + HEIGHT) % HEIGHT;
     double dist = lattice.calculateDistance(tempX, tempY, alvoX, alvoY);

     if (dist > maiorDistancia) 
     {
         maiorDistancia = dist;
         melhorDirecao = dir;
     }
 }

 newX = (x + melhorDirecao.first + WIDTH) % WIDTH;
 newY = (y + melhorDirecao.second + HEIGHT) % HEIGHT;
}
}
else 
{
// Movimento aleatório se não encontrar alvos
std::uniform_int_distribution<int> dis(0, 3);
celula.randonWalk(newX, newY, static_cast<Direction>(dis(rng)));
std::cerr << "⚠️ Fazendo movimento aleatório (nenhum alvo encontrado)\n";
}
} 
else 
{
// Movimento aleatório por probabilidade
std::uniform_int_distribution<int> dis(0, 3);
celula.randonWalk(newX, newY, static_cast<Direction>(dis(rng)));
std::cerr << "⚠️ Fazendo movimento aleatório (decisão probabilística)\n";
}

// Verifica e aplica movimento
if (!lattice.isOccupied(newX, newY, cT, cC, obstaculos, verificacC)) 
{
std::cout << "✅ Posição livre, movendo para (" << newX << "," << newY << ")\n";
celula.changePosition(newX, newY);

// Verifica captura
for (auto it = cC.begin(); it != cC.end(); ) 
{
if (it->getCoordenadaX() == newX && it->getCoordenadaY() == newY) 
{
 it = cC.erase(it);
 std::cout << "✔️ Célula " << celula.getNumero() << " capturou presa!\n";
} 
else 
{
 ++it;
}
}
} 
else 
{
std::cout << "⛔ Posição ocupada! Movimento cancelado.\n";
}
}


/*


void moverCelulaBoa(CellLattice& lattice, Cell& celula,
    std::vector<Cell>& cT, std::vector<Cell>& cC,
    std::vector<Obstacle>& obstaculos, std::mt19937& rng, bool verificacC) 
{
std::cout << "[INICIO] Célula número " << celula.getNumero()
<< " posição atual: (" << celula.getCoordenadaX() << "," << celula.getCoordenadaY() << ")\n";

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

// Busca todos os alvos visíveis (presas "O" e caçadores "N")
auto alvos = celula.findNearestTarget(celulas, lattice, {"O", "N"});

// Variáveis para armazenar alvos
int menorDistPresa = std::numeric_limits<int>::max();
int menorDistCacador = std::numeric_limits<int>::max();
std::pair<int, int> posPresa, posCacador;
bool encontrouPresa = false, encontrouCacador = false;

// Identifica presa e caçador mais próximos
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

// DECISÃO: Perseguir ou Fugir (regra modificada)
bool isFugindo = false;
int alvoX, alvoY;
bool encontrou = false;

// Prioridade 1: Persegue presa se ela estiver mais perto OU igual distância ao caçador
if (encontrouPresa && (!encontrouCacador || menorDistPresa <= menorDistCacador)) 
{
std::tie(alvoX, alvoY) = posPresa;
isFugindo = false;
encontrou = true;
std::cout << "🎯 Perseguindo presa em (" << alvoX << "," << alvoY << ")\n";
} 
// Prioridade 2: Só foge se o caçador estiver MAIS PERTO que qualquer presa
else if (encontrouCacador && menorDistCacador < menorDistPresa) 
{
std::tie(alvoX, alvoY) = posCacador;
isFugindo = true;
encontrou = true;
std::cout << "🚫 Fugindo de caçador em (" << alvoX << "," << alvoY << ")\n";
}

// ===== LÓGICA DE MOVIMENTO =====
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

 if ((!isFugindo && distAlvo < melhorValor) || (isFugindo && distAlvo > melhorValor)) 
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
// Movimento aleatório se não encontrar alvos relevantes
std::uniform_int_distribution<int> dis(0, 3);
celula.randonWalk(newX, newY, dis(rng));
std::cout << "⚠️ Nenhum alvo prioritário. Movimento aleatório.\n";
}
} 
else 
{
// Movimento aleatório se nProbability == 0
std::uniform_int_distribution<int> dis(0, 3);
celula.randonWalk(newX, newY, dis(rng));
std::cout << "⚠️ Movimento aleatório (probabilidade).\n";
}

// ===== VERIFICA SE PODE MOVER =====
std::cout << "➡️ Tentando mover de (" << x << "," << y << ") para (" << newX << "," << newY << ")\n";

if (!lattice.isOccupied(newX, newY, cT, cC, obstaculos, verificacC)) 
{
std::cout << "✅ Posição livre, movendo!\n";
celula.changePosition(newX, newY);

// Verifica se capturou uma presa
for (int i = cC.size() - 1; i >= 0; --i) 
{
if (cC[i].getCoordenadaX() == newX && cC[i].getCoordenadaY() == newY && cC[i].getTipo() == "O") 
{
 cC.erase(cC.begin() + i);
 std::cout << "� Capturou uma presa!\n";
 break;
}
}
} 
else 
{
std::cout << "⛔ Posição ocupada! Movimento cancelado.\n";
}
}

*/



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
    // Parâmetros do caso específico
    int run = 26;
    int numCT = 5;
    int numcC = 10; // ajuste conforme o caso do .dat
    int numPoint = 0; // idem
    double ncNoise_ct = 0.99;
    double ncNoise_cc = 0.99;
    unsigned int seed_run = 2279780005; // do .dat
    CellLattice lattice(100, 100);

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