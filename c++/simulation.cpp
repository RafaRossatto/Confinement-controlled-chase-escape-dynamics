#include "simulation.h"
#include "utils.h"
#include <iomanip>
#include <fstream>

/**
 * @brief Constructs a new Simulation object
 */
Simulation::Simulation(CellLattice& lattice, int numHunters, int numPrey, int numObstacles,
    double hunterNoise, double preyNoise, 
    const std::vector<Obstacle>& obstacles, int hunterSearchRadius, int preySearchRadius,
    unsigned int seed)
: m_lattice(lattice), m_numHunters(numHunters), m_numPrey(numPrey), m_numObstacles(numObstacles),
  m_hunterNoise(hunterNoise), m_preyNoise(preyNoise), 
  m_obstacles(obstacles), m_hunterSearchRadius(hunterSearchRadius), 
  m_preySearchRadius(preySearchRadius), m_seed(seed)
{
    m_fileName = generateFileName(m_numHunters, m_numPrey, m_numObstacles, 
                                 m_preyNoise, m_hunterNoise, 
                                 m_hunterSearchRadius, m_preySearchRadius);
    clearTrajectoryData();
}

/**
 * @brief Saves current positions to trajectory data
 */
void Simulation::saveCurrentPositions(double time, const std::vector<Cell>& prey, const std::vector<Cell>& hunters) 
{
    // Save timestep
    m_trajectoryData.timesteps.push_back(time);
    
    // Save prey positions
    std::vector<std::pair<int, int>> currentPreyPositions;
    for (const auto& p : prey) {
        currentPreyPositions.emplace_back(p.getPositionX(), p.getPositionY());
    }
    m_trajectoryData.preyPositions.push_back(currentPreyPositions);
    
    // Save hunter positions
    std::vector<std::pair<int, int>> currentHunterPositions;
    for (const auto& h : hunters) {
        currentHunterPositions.emplace_back(h.getPositionX(), h.getPositionY());
    }
    m_trajectoryData.hunterPositions.push_back(currentHunterPositions);
}

/**
 * @brief Saves trajectory data to TrajPy format files
 */
void Simulation::saveTrajectoryData(int run) const 
{
    // Save prey trajectories
    std::ofstream preyFile(m_fileName + "_run_" + std::to_string(run) + "_prey_trajectories.csv");
    if (preyFile.is_open()) {
        preyFile << "timestep,cell_id,x,y\n";
        for (size_t t = 0; t < m_trajectoryData.timesteps.size(); ++t) {
            for (size_t c = 0; c < m_trajectoryData.preyPositions[t].size(); ++c) {
                preyFile << m_trajectoryData.timesteps[t] << ","
                         << c << ","
                         << m_trajectoryData.preyPositions[t][c].first << ","
                         << m_trajectoryData.preyPositions[t][c].second << "\n";
            }
        }
        preyFile.close();
    }
    
    // Save hunter trajectories
    std::ofstream hunterFile(m_fileName + "_run_" + std::to_string(run) + "_hunter_trajectories.csv");
    if (hunterFile.is_open()) {
        hunterFile << "timestep,cell_id,x,y\n";
        for (size_t t = 0; t < m_trajectoryData.timesteps.size(); ++t) {
            for (size_t c = 0; c < m_trajectoryData.hunterPositions[t].size(); ++c) {
                hunterFile << m_trajectoryData.timesteps[t] << ","
                           << c << ","
                           << m_trajectoryData.hunterPositions[t][c].first << ","
                           << m_trajectoryData.hunterPositions[t][c].second << "\n";
            }
        }
        hunterFile.close();
    }
}

/**
 * @brief Runs a single simulation run
 */
SimulationResult Simulation::runSingle(int run, std::mt19937& rng) 
{
    // Clear previous trajectory data
    clearTrajectoryData();
    
    const int gridSize = m_lattice.getWidth();
    std::ostringstream ossHunters;
    ossHunters << std::setw(2) << std::setfill('0') << m_numHunters;
    std::string huntersStr = "nC_" + ossHunters.str();

    std::ostringstream ossRun;
    ossRun << std::setw(2) << std::setfill('0') << run;
    std::string runStr = "run_" + ossRun.str();

    std::ostringstream ossObs;
    ossObs << std::setw(2) << std::setfill('0') << m_numObstacles;
    std::string obsStr = "obs_" + ossObs.str();

    std::string filePath = "../" + obsStr + "/" + huntersStr + "/" + runStr + "/inaccessible_preys.txt";
    std::vector<Cell> localHunters, localPrey;
    std::vector<Obstacle> localObstacles = m_obstacles;
    
    // Open inaccessible prey file
    std::ifstream file(filePath);
    if (!file.is_open()) {
        logError("Error opening file: " + filePath);
        return {0.0, 0};
    }

    // Place objects on the grid
    if (!m_lattice.placeObjects(localObstacles, localHunters, localPrey, 
                               m_numObstacles, m_numHunters, m_numPrey, 
                               rng, m_hunterSearchRadius, m_preySearchRadius, run)) {
        logError("Failed to place objects in run " + std::to_string(run));
        return {0.0, 0};
    }

    int inaccessibleCount = 0;
    file >> inaccessibleCount;
    file.close();
    
    std::uniform_int_distribution<int> distX(0, gridSize - 1);
    std::uniform_int_distribution<int> distY(0, gridSize - 1);

    double time = 0.0;
    const double recordingInterval = 1.0;
    double nextRecordingTime = recordingInterval;
    bool checkPrey;

    saveCurrentPositions(time, localPrey, localHunters);

    // Evolution file for prey count
    std::ofstream evolutionFile(m_fileName + "_run_" + std::to_string(run) + "_prey_per_step.csv");
    if (!evolutionFile.is_open()) {
        logError("Error creating evolution file for run " + std::to_string(run));
        return {0.0, 0};
    }

    evolutionFile << "step,living_prey\n";
    evolutionFile << time << "," << localPrey.size() << "\n";

    std::ofstream captureFile(m_fileName + "_run_" + std::to_string(run) + "_captures.csv");
    if (!captureFile.is_open()) 
    {
        logError("Error creating capture log file for run " + std::to_string(run));
        return {0.0, 0};
    }
captureFile << "timestep,hunter_id,prey_id\n";


    // Main simulation loop
    while (time < 1.0e5) {
        // Stop condition: all prey are captured or inaccessible
        if (inaccessibleCount == static_cast<int>(localPrey.size())) {
            evolutionFile << time << "," << localPrey.size() << "\n";
            break;
        }

        // Process gridSize*gridSize movements
        for (int i = 0; i < gridSize * gridSize; ++i) {
            int randomX = distX(rng);
            int randomY = distY(rng);
            std::string gridValue = m_lattice.getGridValue(randomX, randomY);
            bool found = false;

            // Try to move hunter cell
            for (auto& cell : localHunters) 
            {
                if (cell.getPositionX() == randomX && cell.getPositionY() == randomY) {
                    checkPrey = false;
                    int capturedPreyId = m_lattice.moveNormalCell(cell, localHunters, localPrey,
                        localObstacles, rng, checkPrey,
                        m_hunterSearchRadius);

                    if (capturedPreyId >= 0) 
                    {
                        int preyX = cell.getPositionX();  // posição do hunter após mover = posição da presa
                        int preyY = cell.getPositionY();
                        logCapture(captureFile, time, cell.getId(), capturedPreyId);
                    }
                    found = true;
                    break;
                }
            }

            // If no hunter found, try to move prey cell
            if (!found) {
                for (auto& cell : localPrey) {
                    if (cell.getPositionX() == randomX && cell.getPositionY() == randomY) {
                        checkPrey = true;
                        m_lattice.moveCancerCell(cell, localHunters, localPrey, localObstacles, 
                                               rng, checkPrey, m_preySearchRadius);
                        break;
                    }
                }
            }
        }

        // Record state at each interval
        if (time >= nextRecordingTime) {
            evolutionFile << time << "," << localPrey.size() << "\n";
            //  Save positions at recording interval
            saveCurrentPositions(time, localPrey, localHunters);
            nextRecordingTime += recordingInterval;
        }

        // Stop condition: prey count reached minimum
        if (static_cast<int>(localPrey.size()) <= inaccessibleCount) {
            break;
        }

        time += 1.0;
    }

    // Save final positions
    saveCurrentPositions(time, localPrey, localHunters);
    
    evolutionFile.close();
    
    // Save trajectory data to files
    saveTrajectoryData(run);
    captureFile.close();
    
    return {time, static_cast<int>(localPrey.size())};
}

void Simulation::logCapture(std::ofstream& captureFile, double timestep,
    int hunterId, int preyId) const
{
captureFile << timestep << ","
<< hunterId << ","
<< preyId <<"\n";
}