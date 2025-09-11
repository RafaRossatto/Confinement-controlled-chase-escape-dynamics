#include "obstacle.h"

/**
 * @brief Constructs a new Obstacle object with initialization list
 * 
 * @param type Type of the obstacle
 * @param id Unique numeric identifier
 * @param positionX X coordinate position
 * @param positionY Y coordinate position
 */

 Obstacle::Obstacle(const std::string& type, int id, int positionX, int positionY)
 : m_type(type), m_id(id), m_positionX(positionX), m_positionY(positionY)
{}


/**
 * @brief Gets the obstacle type
 * @return std::string Type of the obstacle
 */
std::string Obstacle::getType() const
{
    return m_type;
}

/**
 * @brief Gets the obstacle unique identifier
 * @return int Unique ID of the obstacle
 */
int Obstacle::getId() const
{
    return m_id;
}

/**
 * @brief Gets the X coordinate position
 * @return int X coordinate
 */
int Obstacle::getPositionX() const
{
    return m_positionX;
}

/**
 * @brief Gets the Y coordinate position
 * @return int Y coordinate
 */
int Obstacle::getPositionY() const
{
    return m_positionY;
}