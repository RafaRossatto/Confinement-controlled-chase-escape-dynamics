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

/**
 * @file main.cpp
 * @brief Main simulation driver program
 * 
 * This program runs multiple simulation runs sequentially and collects
 * statistical results for analysis.
 */

std::random_device rd;
unsigned int GLOBAL_SEED = rd(); /**< Global random seed for reproducibility */

/**
 * @struct RunResult
 * @brief Stores the results of a single simulation run
 */
struct RunResult {
    int run;                /**< Run identifier number */
    double steps;           /**< Number of steps executed */
    int remainingPrey;      /**< Number of remaining prey cells */
    unsigned int seed;      /**< Random seed used for this run */
};

/**
 * @brief Main function that drives the simulation
 * 
 * @return int Exit status (0 for success)
 */

int main() 
{
    // Basic configuration
    const int NUM_RUNS = 200;      /**< Number of simulation runs to execute */
    SIZE = 128;                    /**< Grid size (will be square grid) */
    WIDTH = SIZE;                  /**< Grid width */
    HEIGHT = SIZE;                 /**< Grid height */

    // Simulation parameters
    std::vector<int> numHuntersValues = {1433};        /**< Number of hunter cells */
    std::vector<int> numObstaclesValues = {4915};      /**< Number of obstacles */
    std::vector<double> hunterNoiseValues = {1.00};    /**< Hunter movement noise */
    std::vector<double> preyNoiseValues = {1.00};      /**< Prey movement noise */
    const int HUNTER_SEARCH_RADIUS = 2;               /**< Search radius for hunters */
    const int PREY_SEARCH_RADIUS = 2;                 /**< Search radius for prey */

    // Initialize lattice and obstacles
    CellLattice lattice(WIDTH, HEIGHT);
    std::vector<Obstacle> obstacles;

    // Calculate number of prey cells based on available space
    const int gridSize = lattice.getWidth();
    const int totalSites = gridSize * gridSize;
    const int numObstacles = numObstaclesValues[0];
    const int numFreeSites = totalSites - numObstacles;
    const int numPrey = numFreeSites / 4;             /**< Prey occupy 1/4 of free space */
    const int numHunters = numHuntersValues[0];

    // Vector to store results from all runs
    std::vector<RunResult> results(NUM_RUNS);

    logInfo("Starting " + std::to_string(NUM_RUNS) + " simulation runs in parallel");

    // Execute runs in parallel with OpenMP
    #pragma omp parallel for
    for (int run = 0; run < NUM_RUNS; run++) 
    {
        // Generate unique seed for each run based on parameters
        unsigned int runSeed = 1801929750 + run;
        std::mt19937 rng(runSeed);
        
        // Create simulation instance for this thread
        Simulation sim(lattice, numHunters, numPrey, numObstacles,
                      hunterNoiseValues[0], preyNoiseValues[0],
                      obstacles, HUNTER_SEARCH_RADIUS, PREY_SEARCH_RADIUS,
                      runSeed);
        
        logInfo("Processing run " + std::to_string(run));
        
        // Execute simulation run and collect results
        SimulationResult result = sim.runSingle(run, rng);
        
        results[run] = {run, result.steps, result.remainingPrey, runSeed};
        logInfo("Completed run " + std::to_string(run));
    }
    
    // Write results to file (serial section)
    std::ofstream outputFile("simulation_results.csv");
    if (!outputFile.is_open()) {
        logError("Failed to open output file: simulation_results.csv");
        return 1;
    }
    
    // Write CSV header
    outputFile << "run,steps,remaining_prey,seed\n";
    
    // Write results for all runs
    for (const auto& result : results) {
        outputFile << result.run << "," 
                   << result.steps << "," 
                   << result.remainingPrey << "," 
                   << result.seed << "\n";
    }
    outputFile.close();
    
    logInfo("All simulation runs completed! Results saved to: simulation_results.csv");

    return 0;
}
