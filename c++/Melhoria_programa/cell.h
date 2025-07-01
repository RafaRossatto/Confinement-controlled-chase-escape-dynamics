#pragma once

#include <string>
#include "cell_lattice.h"
#include <limits>
class CellLattice;

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


        std::vector<std::pair<int, int>> findNearestTarget(
            const std::vector<Cell>& targets,
            int searchRadius,
            const CellLattice& lattice,
            const std::vector<std::string>& tipos_alvo) const;
        

        void randonWalk(int& x, int& y,int move);
        
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