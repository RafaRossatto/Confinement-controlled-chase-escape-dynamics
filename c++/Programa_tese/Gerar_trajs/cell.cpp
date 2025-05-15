#include "cell.h"

// Construtor sem lista de inicialização
Cell::Cell(const std::string& tipo, int numero, int coordenadaX, int coordenadaY) 
{
    m_tipo = tipo;
    m_numero = numero;
    m_coordenadaX = coordenadaX;
    m_coordenadaY = coordenadaY;
    m_lifeCycle = 0;
    m_division = 0;
    m_hurt = false;
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

bool Cell::getHurt() const
{
    return m_hurt;
}

int Cell::getCoordenadaX() const 
{
    return m_coordenadaX;
}

int Cell::getCoordenadaY() const 
{
    return m_coordenadaY;
}

int Cell::getEnergy() const 
{
    return m_energy;
}

int Cell:: getMinEnergy() const
{
    return m_minEnergy;
}

int Cell:: getLifeCycle() const
{
    return m_lifeCycle;
}

int Cell:: getDivision() const
{
    return m_division;
}

// Método para alterar a posição
void Cell::changePosition(int newX, int newY) 
{
    m_coordenadaX = newX;
    m_coordenadaY = newY;
}

void Cell:: decrementEnergy() 
    {
        m_energy--;
    }

void Cell:: increaseEnergy(int amount)     
{
        m_energy += amount;
}
void Cell:: increaseLifeCycle()
{
    m_lifeCycle++;

}

void Cell:: chanceEnergy(int amount)
{
    m_energy = m_energy/amount;

}

void  Cell:: increaseDivision()
{
    m_division++;
}

void Cell:: cellHurt()
{
    m_hurt = true;
}