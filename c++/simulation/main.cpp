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
                logError("Erro ao abrir o arquivo: " + (caminhoArquivo));
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
            logError("Erro ao criar evoFile da run " + std::to_string(run));
            
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
                        lattice.moverCelulaBoa(cel, cT_local, cC_local, point_local, rng_local, verificacC,2);
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
                        lattice.moverCelulaRuim(cel, cT_local, cC_local, point_local, rng_local, verificacC,2);
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
    std::vector<int> numcT_values = {1024};
    //std::vector<int> numPoint_values = {1638,3276,4915,6553,8192
      //                                  ,9830,11468,13107,14745};    // <- número de obstáculos
    std::vector<int> numPoint_values = {8192};    // <- número de obstáculos
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
                    numcC = (numFree)/4;  // resto do espaço livre
                    if (numcC < 0) {
                        logError("Configuração inválida: mais presas que espaço livre na grade!");
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
