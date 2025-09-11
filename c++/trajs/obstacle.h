#pragma once

#include <string>

/**
 * @class Obstacle
 * @brief Represents an obstacle in the simulation environment
 * 
 * This class stores information about obstacles, including their type,
 * unique identifier, and position in the 2D environment.
 */
class Obstacle 
{
    private:
        std::string m_type;        /**< Type of the obstacle  */
        int m_id;                  /**< Unique identifier of the obstacle */
        int m_positionX;           /**< X coordinate of the obstacle's position */
        int m_positionY;           /**< Y coordinate of the obstacle's position */
    
    public:
    /**
     * @brief Constructs a new Obstacle object
     * 
     * @param type Type of the obstacle
     * @param id Unique numeric identifier
     * @param positionX X coordinate position
     * @param positionY Y coordinate position
     */

     Obstacle(const std::string& type, int id, int positionX, int positionY);
    

    // Getters
    /**
     * @brief Gets the obstacle type
     * @return std::string Type of the obstacle
     */
    std::string getType() const;
    
    /**
     * @brief Gets the obstacle unique identifier
     * @return int Unique ID of the obstacle
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
};