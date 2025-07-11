#include "cell.h"
#include "cell_lattice.h"
#include <algorithm>
#include <vector>

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
    if (newX < 0 || newX >= HEIGHT || newY < 0 || newY >= WIDTH) 
    {
        logError("CRITICAL ERROR: Attempted to move cell " + std::to_string(m_numero) +
         " to an invalid position (" + std::to_string(newX) + "," + std::to_string(newY) + ")\n" +
         "Current position: (" + std::to_string(m_coordenadaX) + "," + std::to_string(m_coordenadaY) + ")\n" +
         "Stack trace:");
        newX = (newX % 100 + 100) % 100; // Força correção
        newY = (newY % 100 + 100) % 100;
    std::cin.get();
    }
    m_coordenadaX = newX;
    m_coordenadaY = newY;
}

/*
std::vector<std::pair<int, int>> Cell::findNearestTarget(
    const std::vector<Cell>& targets,
    const CellLattice& lattice,
    const std::vector<std::string>& tipos_alvo) const
{
    int searchRadius = 100; // 🔥 DEFINIDO FIXO PARA TESTES
    int x = m_coordenadaX;
    int y = m_coordenadaY;

    std::vector<std::pair<int, int>> encontrados;

    for (const auto& alvo : targets) {
        if (alvo.getCoordenadaX() == this->getCoordenadaX() &&
        alvo.getCoordenadaY() == this->getCoordenadaY())
        continue;
       // std::cout << "[DEBUG] Testando alvo tipo: " << alvo.getTipo() 
         //     << " em (" << alvo.getCoordenadaX() << "," << alvo.getCoordenadaY() << ") ";
        if (std::find(tipos_alvo.begin(), tipos_alvo.end(), alvo.getTipo()) == tipos_alvo.end())
        {
        //std::cout << "❌ Ignorado (tipo não incluso)\n";    
        continue;
        }

        
        double dist = lattice.calculateDistance(x, y, alvo.getCoordenadaX(), alvo.getCoordenadaY());

        //td::cout << "[DISTANCIA] De (" << x << "," << y << ") até (" << alvo.getCoordenadaX() << "," << alvo.getCoordenadaY()
         // << ") = " << dist << "\n";

        if (dist <= static_cast<double>(searchRadius) + 1e-6) 
        {
            encontrados.emplace_back(alvo.getCoordenadaX(), alvo.getCoordenadaY());
        }
    }
    
    std::cout << "[DEBUG] Célula em (" << m_coordenadaX << ", " << m_coordenadaY << ") encontrou "
          << encontrados.size() << " alvos:\n";

    for (const auto& par : encontrados) {
    std::cout << "    → Alvo em (" << par.first << ", " << par.second << ")\n";
}   
    
   //std:: cin.get();
    
   return encontrados;
}


*/

/*
std::vector<std::pair<int, int>> Cell::findNearestTarget(
    const std::vector<Cell>& targets,
    const CellLattice& lattice,
    const std::vector<std::string>& tipos_alvo) const
{
    int x = m_coordenadaX;
    int y = m_coordenadaY;
    int sigma = 2;
    std::vector<std::pair<int, int>> encontrados;

    for (const auto& alvo : targets) {
        if (alvo.getCoordenadaX() == x && alvo.getCoordenadaY() == y)
            continue;

        if (std::find(tipos_alvo.begin(), tipos_alvo.end(), alvo.getTipo()) == tipos_alvo.end())
            continue;

        double dist = lattice.calculateDistance(x, y, alvo.getCoordenadaX(), alvo.getCoordenadaY());
        
        if (dist <= static_cast<double>(sigma) + 1e-6) {
            encontrados.emplace_back(alvo.getCoordenadaX(), alvo.getCoordenadaY());
        }
    }

    return encontrados;
}

*/

void Cell::randonWalk(int& x, int& y, Direction move) 
{
    int oldX = x, oldY = y; // Keep the original position.

    switch (move) {
        case SOUTH:
            y = (y - 1 + HEIGHT) % HEIGHT;
            break;
        case NORTH:
            y = (y + 1) % HEIGHT;
            break;
        case EAST:
            x = (x + 1) % WIDTH;
            break;
        
        case WEST:
            x = (x - 1 + WIDTH) % WIDTH;
            break;
    }

    // Verificação de movimento inválido
    if (x < 0 || y < 0) 
    {
        logError("Failed to place the objects in the execution at position: " + std::to_string(oldX) + "," + std::to_string(oldY));
        logError("With direction: " + std::to_string(move));
        logError("Trying to move to position: " + std::to_string(x) + "," + std::to_string(y));

        x = (x + WIDTH) % WIDTH;  // Corrige imediatamente
        y = (y + HEIGHT) % HEIGHT;
        std ::cin.get();
    }
}
void Cell::setSearchRadius(int sr) 
{
    search_radius = sr;
}

int Cell::getSearchRadius() const 
{
    return search_radius;
}