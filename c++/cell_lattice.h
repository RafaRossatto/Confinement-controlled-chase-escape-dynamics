#pragma once
#ifndef CELL_LATTICE_H
#define CELL_LATTICE_H

#include "cell_lattice.h"
#include "utils.h" // for fileExists, logError
#include "cell.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <algorithm> 
#include <random>            // for std::mt19937
#include "obstacle.h"        // for Obstacle class

class Cell;

/**
 * @class CellLattice
 * @brief Represents a 2D grid environment for cell simulation
 * 
 * This class manages a grid-based environment where cells and obstacles
 * are placed and interact. It handles movement, collision detection,
 * and spatial queries.
 */
class CellLattice 
{
    private:
    int m_width;                                /**< Width of the grid */
    int m_height;                               /**< Height of the grid */
    std::vector<std::vector<std::string>> m_grid; /**< 2D grid storing cell types */

    
    public:
    /**
     * @brief Constructs a new CellLattice object
     * 
     * @param width Width of the grid
     * @param height Height of the grid
     */
    CellLattice(int width, int height);
        /**
     * @brief Loads obstacles from file
     * 
     * @param obstacles Vector to store loaded obstacles
     * @param numObstacles Number of obstacles to load
     * @param line Reference to store read lines
     * @return true if loading successful
     * @return false if loading failed
     */
    bool loadObstacles(std::vector<Obstacle>& obstacles, int numObstacles, std::string& line) const;
    
    /**
     * @brief Calculates Manhattan distance with toroidal wrapping
     * 
     * @param x1 First point X coordinate
     * @param y1 First point Y coordinate
     * @param x2 Second point X coordinate
     * @param y2 Second point Y coordinate
     * @return double Manhattan distance
     */
    double calculateDistance(int x1, int y1, int x2, int y2) const;
    
    /**
     * @brief Prints the grid to console
     */
    void printGrid() const;

    //GETTERS
    /**
     * @brief Sets a value in the grid
     * 
     * @param x X coordinate
     * @param y Y coordinate
     * @param value Value to set
     */

    /**
     * @brief Gets a value from the grid
     * 
     * @param x X coordinate
     * @param y Y coordinate
     * @return std::string Value at position
     */
    std::string getGridValue(int x, int y) const;

    /**
     * @brief Gets the grid width
     * @return int Grid width
     */
    int getWidth() const { return m_width; }
    
    /**
     * @brief Gets the grid height
     * @return int Grid height
     */
    int getHeight() const { return m_height; }
    
    //SETTERS
    /**
     * @brief Sets a value in the grid
     * 
     * @param x X coordinate
     * @param y Y coordinate
     * @param value Value to set
     */
    void setGridValue(int x, int y, const std::string& value);

    
        /**
     * @brief Places objects (obstacles, normal cells, cancer cells) on the grid
     * 
     * @param obstacles Vector of obstacles
     * @param normalCells Vector of normal cells
     * @param cancerCells Vector of cancer cells
     * @param numObstacles Number of obstacles
     * @param numNormal Number of normal cells
     * @param numCancer Number of cancer cells
     * @param rng Random number generator
     * @param searchRadiusNormal Search radius for normal cells
     * @param searchRadiusCancer Search radius for cancer cells
     * @param run Current run identifier
     * @return true if placement successful
     * @return false if placement failed
     */
    bool placeObjects(std::vector<Obstacle>& obstacles,
        std::vector<Cell>& normalCells,
        std::vector<Cell>& cancerCells,
        int numObstacles, int numNormal, int numCancer,
        std::mt19937& rng,
        int searchRadiusNormal, int searchRadiusCancer, int run);
    
    
    
    /**
     * @brief Checks if an object overlaps with objects in a list
     * 
     * @tparam T Type of the object to check
     * @tparam U Type of objects in the list
     * @param obj Object to check
     * @param list List of objects to check against
     * @return true if overlap exists
     * @return false if no overlap
     */
    template<typename T, typename U>
    bool overlapsWithList(const T& obj, const std::vector<U>& list) const;

    /**
     * @brief Checks general overlap with all object types
     * 
     * @tparam T Type of the object to check
     * @param obj Object to check
     * @param obstacles Vector of obstacles
     * @param normalCells Vector of normal cells
     * @param cancerCells Vector of cancer cells
     * @return true if overlap exists
     * @return false if no overlap
     */
    template<typename T>
    bool generalOverlap(const T& obj,
                        const std::vector<Obstacle>& obstacles,
                        const std::vector<Cell>& normalCells,
                        const std::vector<Cell>& cancerCells) const;
       

    /**
     * @brief Checks if a position is occupied
     * 
     * @param x X coordinate
     * @param y Y coordinate
     * @param normalCells Vector of normal cells
     * @param cancerCells Vector of cancer cells
     * @param obstacles Vector of obstacles
     * @param checkCancer Whether to check cancer cells
     * @return true if position is occupied
     * @return false if position is free
     */
    bool isOccupied(int x, int y,
        const std::vector<Cell>& normalCells,
        const std::vector<Cell>& cancerCells,
        const std::vector<Obstacle>& obstacles,
        bool checkCancer) const;
;

    /**
     * @brief Counts targets around a position
     * 
     * @param x X coordinate
     * @param y Y coordinate
     * @param agents Vector of agents to check
     * @param types Types of agents to count
     * @param searchRadius Search radius
     * @return int Number of targets found
     */
    int countTargetsAround(int x, int y,
        const std::vector<Cell>& agents,
        const std::vector<std::string>& types,
        int searchRadius = 2) const;
                    
    /**
     * @brief Moves a cancer cell (bad cell) according to its behavior
     * 
     * @param cell Cell to move
     * @param normalCells Vector of normal cells
     * @param cancerCells Vector of cancer cells
     * @param obstacles Vector of obstacles
     * @param rng Random number generator
     * @param checkCancer Whether to check cancer cells
     * @param searchRadius Search radius
     */
    void moveCancerCell(Cell& cell,
        std::vector<Cell>& normalCells, std::vector<Cell>& cancerCells,
        std::vector<Obstacle>& obstacles,
        std::mt19937& rng, bool checkCancer, int searchRadius);
                    
    /**
     * @brief Moves a normal cell (good cell) according to its behavior
     * 
     * @param cell Cell to move
     * @param normalCells Vector of normal cells
     * @param cancerCells Vector of cancer cells
     * @param obstacles Vector of obstacles
     * @param rng Random number generator
     * @param checkCancer Whether to check cancer cells
     * @param searchRadius Search radius
     */
    int moveNormalCell(Cell& cell,
        std::vector<Cell>& normalCells, std::vector<Cell>& cancerCells,
        std::vector<Obstacle>& obstacles,
        std::mt19937& rng, bool checkCancer, int searchRadius);
};

// Template implementations
template<typename T, typename U>
bool CellLattice::overlapsWithList(const T& obj, const std::vector<U>& list) const {
    for (const auto& item : list) {
        if (obj.getPositionX() == item.getPositionX() &&
            obj.getPositionY() == item.getPositionY()) {
            return true;
        }
    }
    return false;
}

template<typename T>
bool CellLattice::generalOverlap(const T& obj,
                                 const std::vector<Obstacle>& obstacles,
                                 const std::vector<Cell>& normalCells,
                                 const std::vector<Cell>& cancerCells) const {
    return overlapsWithList(obj, obstacles) ||
           overlapsWithList(obj, normalCells) ||
           overlapsWithList(obj, cancerCells);
}

#endif // CELL_LATTICE_H