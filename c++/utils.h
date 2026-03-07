#pragma once
#ifndef UTILS_H
#define UTILS_H

#include <cmath>        // for std::abs
#include "config.h"     // For WIDTH and HEIGHT if needed
#include <iostream>
#include <fstream>
#include <string>

/**
 * @brief Logs an informational message to standard output
 * 
 * @param message The message to log
 */
inline void logInfo(const std::string& message) 
{
    std::cout << "[INFO] " << message << std::endl;
}


/**
 * @brief Logs a warning message to standard output
 * 
 * @param message The message to log
 */
inline void logWarning(const std::string& message) 
{
    std::cout << "[WARNING] " << message << std::endl;
}

/**
 * @brief Logs an error message to standard error
 * 
 * @param message The message to log
 */
inline void logError(const std::string& message) 
{
    std::cerr << "[ERROR] " << message << std::endl;
}

/**
 * @brief Checks if an index is valid for a given size
 * 
 * @param index The index to check
 * @param size The size of the container
 * @return true if index is valid
 * @return false if index is invalid
 */
inline bool isValidIndex(int index, int size)
{
    return (index >= 0) && (index < size);
}

/**
 * @brief Opens a file in append mode (creates if doesn't exist)
 * 
 * @param fileName The name of the file to open
 */
inline void openFile(const std::string& fileName) 
{
    // Open file in write and append mode
    std::ofstream file(fileName, std::ios::app);

    // Check if the file was opened successfully
    if (!file.is_open()) 
    {
        logError("Error opening file " + fileName);
        return;
    }

    file.close();
}


/**
 * @brief Checks if a file exists
 * 
 * @param filename The name of the file to check
 * @return true if file exists
 * @return false if file doesn't exist
 */
inline bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good(); // Checks if the file can be opened
}

/**
 * @brief Generates a standardized filename for simulation output
 * 
 * @param numHunters Number of hunter cells
 * @param numPrey Number of prey cells
 * @param numObstacles Number of obstacles
 * @param preyNoise Noise parameter for prey movement
 * @param hunterNoise Noise parameter for hunter movement
 * @param hunterSearchRadius Search radius for hunter cells
 * @param preySearchRadius Search radius for prey cells
 * @return std::string Generated filename
 */
inline std::string generateFileName(int numHunters, int numPrey, int numObstacles,
                                   double preyNoise, double hunterNoise,
                                   int hunterSearchRadius, int preySearchRadius) 
{   
    return "NC_" + std::to_string(numHunters) +
           "_NE_" + std::to_string(numPrey) +
           "_O_" + std::to_string(numObstacles) +
           "_TCC_" + std::to_string(preyNoise) +
           "_SR_" + std::to_string(preySearchRadius) +
           "_TCT_" + std::to_string(hunterNoise) +
           "_SR_" + std::to_string(hunterSearchRadius) + ".dat";
}

#endif // UTILS_H
