#include "recurso.h"

// Construtor sem lista de inicialização
Recurso::Recurso(const std::string& tipo, int numero, int coordenadaX, int coordenadaY,int energy) 
{
    m_tipo = tipo;
    m_numero = numero;
    m_coordenadaX = coordenadaX;
    m_coordenadaY = coordenadaY;
    m_energy = energy;
}

// Getters
std::string Recurso::getTipo() const {
    return m_tipo;
}

int Recurso::getNumero() const {
    return m_numero;
}

int Recurso::getCoordenadaX() const {
    return m_coordenadaX;
}

int Recurso::getCoordenadaY() const {
    return m_coordenadaY;
}
int Recurso::getEnergy() const {
    return m_energy;
}

