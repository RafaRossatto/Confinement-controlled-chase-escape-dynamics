#pragma once
#ifndef SIMULATION_H
#define SIMULATION_H

#include "cell_lattice.h"
#include "obstacle.h"
#include "config.h"
#include <string>
#include <vector>
#include <random>
#include <fstream>
#include <utility> // para std::pair

/**
 * @struct TrajectoryData
 * @brief Stores trajectory data for all cells across timesteps
 */
struct TrajectoryData 
{
    std::vector<double> timesteps;
    std::vector<std::vector<std::pair<int, int>>> preyPositions;   // [timestep][cell](x,y)
    std::vector<std::vector<std::pair<int, int>>> hunterPositions; // [timestep][cell](x,y)
};

/**
 * @struct SimulationResult
 * @brief Stores the results of a simulation run
 */
struct SimulationResult 
{
    double steps;        /**< Number of steps executed */
    int remainingPrey;   /**< Number of remaining prey cells at the end */
};

/**
 * @class Simulation
 * @brief Manages and executes the cell simulation
 * 
 * This class handles the main simulation loop, movement of cells,
 * and collection of results for statistical analysis.
 */
class Simulation 
{
private:
    CellLattice& m_lattice;          /**< Reference to the cell lattice environment */
    int m_numHunters;                /**< Number of hunter (normal) cells */
    int m_numPrey;                   /**< Number of prey (cancer) cells */
    int m_numObstacles;              /**< Number of obstacles */
    double m_hunterNoise;            /**< Noise parameter for hunter movement */
    double m_preyNoise;              /**< Noise parameter for prey movement */
    std::string m_fileName;          /**< Base filename for output files */
    std::vector<Obstacle> m_obstacles; /**< Vector of obstacles */
    int m_hunterSearchRadius;        /**< Search radius for hunter cells */
    int m_preySearchRadius;          /**< Search radius for prey cells */
    unsigned int m_seed;             /**< Random seed for reproducibility */
    TrajectoryData m_trajectoryData; /**< Trajectory data storage */

    /**
     * @brief Saves current positions to trajectory data
     * @param time Current simulation time
     * @param prey Vector of prey cells
     * @param hunters Vector of hunter cells
     */
    void saveCurrentPositions(double time, const std::vector<Cell>& prey, const std::vector<Cell>& hunters);

public:
    /**
     * @brief Constructs a new Simulation object
     * 
     * @param lattice Reference to the cell lattice
     * @param numHunters Number of hunter cells
     * @param numPrey Number of prey cells
     * @param numObstacles Number of obstacles
     * @param hunterNoise Noise parameter for hunters
     * @param preyNoise Noise parameter for prey
     * @param obstacles Vector of obstacles
     * @param hunterSearchRadius Search radius for hunters
     * @param preySearchRadius Search radius for prey
     * @param seed Random seed (0 for random)
     */
    Simulation(CellLattice& lattice, int numHunters, int numPrey, int numObstacles,
               double hunterNoise, double preyNoise, 
               const std::vector<Obstacle>& obstacles, int hunterSearchRadius, int preySearchRadius,
               unsigned int seed = 0);
    
    /**
     * @brief Runs a single simulation run
     * 
     * @param run Run identifier number
     * @param rng Random number generator
     * @return SimulationResult Results of the simulation run
     */
    SimulationResult runSingle(int run, std::mt19937& rng);
    
    // Getters
    int getNumHunters() const { return m_numHunters; }
    int getNumPrey() const { return m_numPrey; }
    int getNumObstacles() const { return m_numObstacles; }
    std::string getFileName() const { return m_fileName; }
    unsigned int getSeed() const { return m_seed; }
    
    /**
     * @brief Gets the trajectory data
     * @return const TrajectoryData& Reference to trajectory data
     */
    const TrajectoryData& getTrajectoryData() const { return m_trajectoryData; }
    
    /**
     * @brief Clears the trajectory data
     */
    void clearTrajectoryData() { 
        m_trajectoryData.timesteps.clear();
        m_trajectoryData.preyPositions.clear();
        m_trajectoryData.hunterPositions.clear();
    }
    
    /**
     * @brief Saves trajectory data to TrajPy format files
     * @param run Run identifier number
     */
    void saveTrajectoryData(int run) const;

    /**
     * @brief Salva evento de captura (quem capturou quem, quando e onde).
     * 
     * @param captureFile Arquivo CSV já aberto para escrita
     * @param timestep Momento da captura
     * @param hunterId ID do caçador
     * @param preyId ID da presa
     * @param preyX Posição X da presa
     * @param preyY Posição Y da presa
     */
    void logCapture(std::ofstream& captureFile, double timestep,
        int hunterId, int preyId) const;


};

#endif // SIMULATION_H