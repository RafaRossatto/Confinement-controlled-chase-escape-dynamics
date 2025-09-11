#include "simulation.h"
#include "obstacle.h"
#include "config.h"
#include "utils.h"
#include "cell.h"
#include "cell_lattice.h"
#include <iostream>
#include <fstream>
#include <vector>

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
    const int NUM_RUNS = 100;      /**< Number of simulation runs to execute */
    SIZE = 128;                    /**< Grid size (will be square grid) */
    WIDTH = SIZE;                  /**< Grid width */
    HEIGHT = SIZE;                 /**< Grid height */
    //CAPTURE_PROBABILITY = 1.0;     /**< Probability of successful capture */

    // Simulation parameters
    std::vector<int> numHuntersValues = {1024};        /**< Number of hunter cells */
    std::vector<int> numObstaclesValues = {8192};      /**< Number of obstacles */
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

    // Create simulation instance
    Simulation sim(lattice, numHunters, numPrey, numObstacles,
                  hunterNoiseValues[0], preyNoiseValues[0],
                  obstacles, HUNTER_SEARCH_RADIUS, PREY_SEARCH_RADIUS,
                  GLOBAL_SEED);

    logInfo("Starting " + std::to_string(NUM_RUNS) + " simulation runs");

    // Vector to store results from all runs
    std::vector<RunResult> results(NUM_RUNS);

    // Execute runs sequentially
    for (int run = 0; run < NUM_RUNS; run++) 
    {
        // Generate unique seed for each run based on parameters
        unsigned int runSeed = GLOBAL_SEED + 10 * PREY_SEARCH_RADIUS + 10 * HUNTER_SEARCH_RADIUS + 
                              run + 1000 * numPrey + 100000 * numObstacles;
        std::mt19937 rng(runSeed);
        
        logInfo("Processing run " + std::to_string(run));
        
        // Execute simulation run and collect results
        SimulationResult result = sim.runSingle(run, rng);
        
        results[run] = {run, result.steps, result.remainingPrey, runSeed};
        logInfo("Completed run " + std::to_string(run));
    }
    
    // Write results to file
    std::ofstream outputFile(sim.getFileName());
    if (!outputFile.is_open()) {
        logError("Failed to open output file: " + sim.getFileName());
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
    
    logInfo("All simulation runs completed! Results saved to: " + sim.getFileName());

    return 0;
}