#include "cell.h"

// Construtor sem lista de inicialização
Cell::Cell(const std::string& tipo, int numero, int coordenadaX, int coordenadaY) 
{
    m_tipo = tipo;
    m_numero = numero;
    m_coordenadaX = coordenadaX;
    m_coordenadaY = coordenadaY;
}

// Getters
std::string Cell::getTipo() const 
{
    return m_tipo;
}

int Cell::getNumero() const 
{
    return m_numero;
}

int Cell::getCoordenadaX() const 
{
    return m_coordenadaX;
}

int Cell::getCoordenadaY() const 
{
    return m_coordenadaY;
}

// Método para alterar a posição
void Cell::changePosition(int newX, int newY) 
{
    m_coordenadaX = newX;
    m_coordenadaY = newY;
}

std::pair<int, int> Cell::findNearestTarget(const std::vector<Cell>& targets,
    int searchRadius,
    const CellLattice& lattice) const
{
int nearestIndex = -1;
int minDistance = std::numeric_limits<int>::max();

for (int radius : RAIOS_PROGRESSIVOS) {
if (radius > searchRadius) break;

for (int i = 0; i < targets.size(); ++i) {
int distance = lattice.calculateDistance(this->getCoordenadaX(),
             this->getCoordenadaY(),
             targets[i].getCoordenadaX(),
             targets[i].getCoordenadaY());

if (distance <= radius && distance < minDistance) {
minDistance = distance;
nearestIndex = i;
}
}
}

return {nearestIndex, minDistance};
}

void Cell::randonWalk(int& x, int& y,int move)
{
    switch (move) 
    {
        case NORTH:
            y = (y+1)%HEIGHT;
            break;
        
        case EAST:
            x=(x+1)% WIDTH;
            break;
        case SOUTH:
            y = (y - 1 + HEIGHT) % HEIGHT;
            break;
        case WEST:
            x = (x - 1 + WIDTH) % WIDTH;
            break;
    }
}

void Cell::setSearchRadius(int sr) {
    search_radius = sr;
}

int Cell::getSearchRadius() const {
    return search_radius;
}