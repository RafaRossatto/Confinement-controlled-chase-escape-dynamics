#include "simulation.h"
#include "obstacle.h"
#include "config.h"
#include "utils.h"
#include "cell.h"
#include "cell_lattice.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <omp.h>

std::random_device rd;
unsigned int GLOBAL_SEED = rd();

int main() 
{
    // Configurações básicas
    int count = 100;
    SIZE = 128;
    WIDTH = SIZE;
    HEIGHT = SIZE;
    CAPTUREPROBABILITY = 1.0;

    // Parâmetros
    std::vector<int> numcT_values = {1024};
    std::vector<int> numPoint_values = {8192};
    std::vector<double> numNoise_ct = {1.00};
    std::vector<double> numNoise_cc = {1.00};
    int sr_normal = 2;
    int sr_cancer = 2;

    // Inicializar
    CellLattice lattice(WIDTH, HEIGHT);
    std::vector<Obstacle> point;

    // Calcular numcC
    int L = lattice.getWidth();
    int total_sites = L * L;
    int numPoint = numPoint_values[0];
    int numFree = total_sites - numPoint;
    int numcC = numFree / 4;
    int numCT = numcT_values[0];

    // Criar simulação
    Simulation sim(lattice, numCT, numcC, numPoint,
                  numNoise_ct[0], numNoise_cc[0],
                  point, sr_normal, sr_cancer,
                  GLOBAL_SEED);

    logInfo("Iniciando " + std::to_string(count) + " runs");

    // Abrir arquivo de saída
    std::ofstream outputFile(sim.getFileName());
    if (!outputFile.is_open()) {
        logError("Failed to open file: " + sim.getFileName());
        return 1;
    }
    outputFile << "run,steps,escapers,seed\n";

    // Executar runs em paralelo
    #pragma omp parallel for schedule(dynamic)
    for (int run = 0; run < count; run++) 
    {
        unsigned int seed_run = GLOBAL_SEED + 10 * sr_cancer + 10 * sr_normal + 
                              run + 1000 * numcC + 100000 * numPoint;
        std::mt19937 rng_local(seed_run);
        
        Simulation sim_local = sim;
        
        #pragma omp critical
        logInfo("Thread " + std::to_string(omp_get_thread_num()) + " processando run " + std::to_string(run));
        
        // Executar run - a escrita serializada está DENTRO do runSingle
        sim_local.runSingle(run, rng_local, outputFile);
        
        #pragma omp critical
        logInfo("Thread " + std::to_string(omp_get_thread_num()) + " concluiu run " + std::to_string(run));
    }
    
    outputFile.close();
    logInfo("Todas as runs concluídas!");

    return 0;
}