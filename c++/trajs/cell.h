#pragma once

#include <string>
#include "cell_lattice.h"
#include <limits>
class CellLattice;


/**
 * @enum Direction
 * @brief Represents the possible movement directions
 */
enum Direction 
{ 
    NORTH = 1,  /**< North direction */
    EAST = 0,   /**< East direction */
    SOUTH = 2,  /**< South direction */
    WEST = 3    /**< West direction */
};

/**
 * @class Cell
 * @brief Represents a cell in the simulation environment
 * 
 * This class manages cell properties, position, and movement behavior
 * within a bounded lattice environment.
 */
class Cell 
{
    private:
        std::string m_type;           /**< Type of the cell */
        int m_id;                     /**< Unique identifier of the cell */
        int m_positionX;              /**< X coordinate position */
        int m_positionY;              /**< Y coordinate position */
        int m_searchRadius;           /**< Search radius for cell operations */

    public:
        /**
         * @brief Constructs a new Cell object
         * 
         * @param type Type of the cell
         * @param id Unique identifier
         * @param positionX Initial X coordinate
         * @param positionY Initial Y coordinate
         */
        Cell(const std::string& type, int id, int positionX, int positionY);

        /**
         * @brief Performs a random walk movement in the specified direction
         * 
         * @param x Reference to X coordinate (will be modified)
         * @param y Reference to Y coordinate (will be modified)
         * @param move Direction of movement
         */
        void randomWalk(int& x, int& y, Direction move);

        /**
         * @brief Changes the cell's position with boundary validation
         * 
         * @param newX New X coordinate
         * @param newY New Y coordinate
         */
        void changePosition(int newX, int newY);
        
        // Getters
        /**
         * @brief Gets the cell type
         * @return std::string Type of the cell
         */
        std::string getType() const;

        /**
         * @brief Gets the cell unique identifier
         * @return int Unique ID
         */
        int getId() const;

        /**
         * @brief Gets the X coordinate position
         * @return int X coordinate
         */
        int getPositionX() const;
        
        /**
         * @brief Gets the Y coordinate position
         * @return int Y coordinate
         */
        int getPositionY() const; 

        /**
         * @brief Gets the current search radius
         * @return int Search radius value
         */
        int getSearchRadius() const;

        // Setters
        /**
         * @brief Sets the search radius for the cell
         * @param searchRadius New search radius value
         */
        void setSearchRadius(int searchRadius);
};