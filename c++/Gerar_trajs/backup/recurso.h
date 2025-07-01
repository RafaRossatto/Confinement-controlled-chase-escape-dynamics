#include <string>
class Recurso 
{
    private:
    std:: string m_tipo;   /**< The type of the agent. */
    int m_numero;          /**< The number of the agent. */ 
    int m_coordenadaX; /**< The x-coordinate of the agent's position. */
    int m_coordenadaY; /**< The y-coordinate of the agent's position. */
    int m_energy;
    

    public:
    
        Recurso(const std::string& tipo, int numero, int coordenadaX, int coordenadaY,int energy);
    

    // Getters
    std::string getTipo() const; 
    int getNumero() const; 
    int getCoordenadaX() const; 
    int getCoordenadaY() const;
    int getEnergy() const;
};