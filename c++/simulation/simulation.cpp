#include "simulation.h"
#include "utils.h"
#include <iomanip>
#include <omp.h>

// ⚡ IMPLEMENTAÇÃO DO CONSTRUTOR ⚡
Simulation::Simulation(CellLattice& lattice_, int numCT_, int numcC_, int numPoint_,
    double ncNoise_ct_, double ncNoise_cc_, 
    const std::vector<Obstacle>& obstacles_, int sr_normal_, int sr_cancer_,
    unsigned int seed_)
: lattice(lattice_), numCT(numCT_), numcC(numcC_), numPoint(numPoint_),
ncNoise_ct(ncNoise_ct_), ncNoise_cc(ncNoise_cc_), 
obstacles(obstacles_), sr_normal(sr_normal_), sr_cancer(sr_cancer_), seed(seed_)
{
fileName = gerarNomeArquivo(numCT, numcC, numPoint, ncNoise_cc, ncNoise_ct, sr_normal, sr_cancer);
}

void Simulation::runSingle(int run, std::mt19937& rng, std::ofstream& outputFile) 
{
    const int L = lattice.getWidth();
    std::ostringstream oss_nc;
    oss_nc << std::setw(2) << std::setfill('0') << numCT;
    std::string ncStr = "nC_" + oss_nc.str();

    std::ostringstream oss_run;
    oss_run << std::setw(2) << std::setfill('0') << run;
    std::string runStr = "run_" + oss_run.str();

    std::ostringstream oss_obs;
    oss_obs << std::setw(2) << std::setfill('0') << numPoint;
    std::string ObsStr = "obs_" + oss_obs.str();

    std::string caminhoArquivo = "../" + ObsStr + "/" + ncStr + "/" + runStr + "/inaccessible_preys.txt";
    std::vector<Cell> cT_local, cC_local;
    std::vector<Obstacle> point_local = obstacles;
    
    // Abrir arquivo de presas inacessíveis
    std::ifstream arquivo(caminhoArquivo);
    if (!arquivo.is_open()) {
        #pragma omp critical
        logError("Erro ao abrir o arquivo: " + caminhoArquivo);
        return;
    }

    // Colocar objetos na grade
    if (!lattice.placeObjects(point_local, cT_local, cC_local, numPoint, numCT, numcC, rng, sr_normal, sr_cancer, run)) {
        #pragma omp critical
        logError("Failed to place objects in run " + std::to_string(run));
        return;
    }

    int countInacessiveis = 0;
    arquivo >> countInacessiveis;
    arquivo.close();
    
    std::uniform_int_distribution<int> distX(0, L - 1);
    std::uniform_int_distribution<int> distY(0, L - 1);

    double t = 0.0;
    const double intervalo = 1.0;
    double proximoRegistro = intervalo;
    bool verificacC;

    // Arquivo de evolução das presas (local para cada run)
    std::ofstream evoFile(fileName + "_run_" + std::to_string(run) + "_presas_por_passo.csv");
    if (!evoFile.is_open()) {
        #pragma omp critical
        logError("Erro ao criar evoFile da run " + std::to_string(run));
        return;
    }

    evoFile << "passo,presas_vivas\n";
    evoFile << t << "," << cC_local.size() << "\n";

    // Loop principal da simulação
    while (t < 1.0e5) {
        // Condição de parada: todas as presas foram capturadas ou são inacessíveis
        if (countInacessiveis == static_cast<int>(cC_local.size())) {
            evoFile << t << "," << cC_local.size() << "\n";
            break;
        }

        // Processar L*L movimentos (um passo de tempo)
        for (int i = 0; i < L * L; ++i) {
            int x_rand = distX(rng);
            int y_rand = distY(rng);
            std::string valor = lattice.getGridValue(x_rand, y_rand);
            bool encontrou = false;

            // Tentar mover célula normal (caçadora)
            for (auto& cel : cT_local) {
                if (cel.getCoordenadaX() == x_rand && cel.getCoordenadaY() == y_rand) {
                    verificacC = false;
                    lattice.moverCelulaBoa(cel, cT_local, cC_local, point_local, rng, verificacC, 2);
                    encontrou = true;
                    break;
                }
            }

            // Se não encontrou célula normal, tentar mover célula cancerosa (presa)
            if (!encontrou) {
                for (auto& cel : cC_local) {
                    if (cel.getCoordenadaX() == x_rand && cel.getCoordenadaY() == y_rand) {
                        verificacC = true;
                        lattice.moverCelulaRuim(cel, cT_local, cC_local, point_local, rng, verificacC, 2);
                        break;
                    }
                }
            }
        }

        // Registrar estado a cada intervalo
        if (t >= proximoRegistro) {
            evoFile << t << "," << cC_local.size() << "\n";
            proximoRegistro += intervalo;
        }

        // Condição de parada: número de presas atingiu o mínimo
        if (static_cast<int>(cC_local.size()) <= countInacessiveis) {
            break;
        }

        t += 1.0;
    }

    // ⚡⚡⚡ ESCRITA SERIALIZADA - APENAS AQUI ⚡⚡⚡
    #pragma omp critical
    {
        outputFile << run << "," << t << "," << cC_local.size() << "," << rng << "\n";
        outputFile.flush(); // Garantir escrita imediata
    }

    evoFile.close();
}