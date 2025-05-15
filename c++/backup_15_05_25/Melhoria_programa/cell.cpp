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