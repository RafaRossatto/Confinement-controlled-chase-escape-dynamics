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
    int x = celula.getCoordenadaX();
    int y = celula.getCoordenadaY();
    int newX = x, newY = y;
    std::vector<Cell> celulas;
    celulas.insert(celulas.end(), cT.begin(), cT.end());
    celulas.insert(celulas.end(), cC.begin(), cC.end());

    // Busca caçadores no raio de procura
    std::vector<std::string> tipos = {"C"};
    auto alvos = celula.findNearestTarget(celulas, celula.getSearchRadius(), lattice, tipos);

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
        auto alvos = celula.findNearestTarget(celulas, raio, lattice, {"N", "C"});

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
                    if (agente.getTipo() == "N" && dist < menorDistPresa) 
                    {
                        menorDistPresa = dist;
                        posPresa = {ax, ay};
                        encontrouPresa = true;
                    }
                    else if (agente.getTipo() == "C" && dist < menorDistCacador) 
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
        const int intervalo = 1;

        int steps = 1;
        bool verificacC;
        while (steps < 10000) 
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

















//velha
/*
int runSimulationPaper(const int NUMSTEPS, std::vector<Cell>& cT, 
std::vector<Cell>& cC, std::vector<Obstacle>& point, const std::string& fileName, std::mt19937& rng,
bool write,int sr_normal, int sr_cancer) 
{
    int steps = 1; // Step counter
    bool isPursuing,verificacC;

    // Main simulation loop
    while (steps < NUMSTEPS) 
    {   
        // Process cells of type T
        if (!cT.empty())
        {
            for (int i = cT.size() - 1; i >= 0; --i)
            {
                verificacC = false;
                moverCelulaBoa(cT[i], cT, cC, point, rng,verificacC);
            }
        }

        // Process cancer cells
        if (!cC.empty())
        {
            for (int i = cC.size() - 1; i >= 0; --i)
            {
            verificacC = true;
            moverCelulaRuim(cC[i], cT, cC, point, rng,verificacC);
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

*/
//8,41,0,2241289420
//12,35,0,2241289174
//12,280,0,2709020504
int main() 
{
    // Parâmetros do caso específico
    int run = 12;
    int numCT = 500;
    int numcC = 100; // ajuste conforme o caso do .dat
    int numPoint = 0; // idem
    double ncNoise_ct = 0.05;
    double ncNoise_cc = 0.05;
    unsigned int seed_run = 2709020504; // do .dat

    setCTPROBABILITY(ncNoise_ct);
    setCCPROBABILITY(ncNoise_cc);
    int sr_cancer = 50;
    int sr_normal = 5;

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
         numcC, 100, 100,rng_local,sr_normal,sr_cancer)) 
    {
        std::cerr << "Erro ao posicionar objetos para run " << run << "\n";
        return 1;
    }


    std::string trajectoryFileName = "Run_Trajectory_NH_500_NE_" + std::to_string(numcC) +
                                    "_O_" + std::to_string(numPoint) +
                                    "_TCC_" + std::to_string(ncNoise_cc) +
                                    "_SR_" + std::to_string(sr_cancer)+
                                    "_TCT_" + std::to_string(ncNoise_ct) +
                                    "_SR_" + std::to_string(sr_normal)+"Run_"+ std:: to_string(run) + ".xyz";

    int steps_local = runSimulationPaper(10000, cT, cC, point, trajectoryFileName,rng_local, true,sr_normal,sr_cancer);

    std::cout << "Re-execução completa com " << steps_local << " passos." << "\n";

    return 0;
}