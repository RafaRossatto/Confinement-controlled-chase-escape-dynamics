#pragma once

#include <string>
#include "cell_lattice.h"
#include <limits>
class CellLattice;

enum Direction { NORTH=1, EAST=0, SOUTH=2, WEST=3 };
class Cell 
{
    private:
    std:: string m_tipo;   /**< The type of the agent. */
    int m_numero;          /**< The number of the agent. */ 
    int m_coordenadaX; /**< The x-coordinate of the agent's position. */
    int m_coordenadaY; /**< The y-coordinate of the agent's position. */
    int search_radius;

    public:
    
        Cell(const std::string& tipo, int numero, int coordenadaX, int coordenadaY);

        void randonWalk(int& x, int& y,Direction move);
        
    // Getters
    std::string getTipo() const; 
    int getNumero() const; 
    int getCoordenadaX() const; 
    int getCoordenadaY() const;
    void setSearchRadius(int sr);
    int getSearchRadius() const;

    // Método para alterar a posição
    void changePosition(int newX,int newY);
};