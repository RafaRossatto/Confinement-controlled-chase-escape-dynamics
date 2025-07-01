#include "obstacle.h"

// Construtor sem lista de inicialização
Obstacle::Obstacle(const std::string& tipo, int numero, int coordenadaX, int coordenadaY) 
{
    m_tipo = tipo;
    m_numero = numero;
    m_coordenadaX = coordenadaX;
    m_coordenadaY = coordenadaY;
}

// Getters
std::string Obstacle::getTipo() const {
    return m_tipo;
}

int Obstacle::getNumero() const {
    return m_numero;
}

int Obstacle::getCoordenadaX() const {
    return m_coordenadaX;
}

int Obstacle::getCoordenadaY() const {
    return m_coordenadaY;
}