#include <string>
class Cell 
{
    private:
    std:: string m_tipo;   /**< The type of the agent. */
    int m_numero;          /**< The number of the agent. */ 
    int m_coordenadaX; /**< The x-coordinate of the agent's position. */
    int m_coordenadaY; /**< The y-coordinate of the agent's position. */
    int m_energy;
    int m_minEnergy;
    int m_lifeCycle;
    int m_division;
    bool m_hurt;
    
    public:
    
        Cell(const std::string& tipo, int numero, int coordenadaX, int coordenadaY);
    

    // Getters
    std::string getTipo() const; 
    int getNumero() const; 
    int getCoordenadaX() const; 
    int getCoordenadaY() const;
    int getEnergy() const;
    int getMinEnergy() const;
    int getLifeCycle() const; 
    int getDivision() const;
    bool getHurt() const; 
    
    // Método para alterar a posição
    void changePosition(int newX,int newY);
    // Método para decrementar energia
    void decrementEnergy();
    // Método para incrementar energia
    void increaseEnergy(int amount);
    
    void increaseLifeCycle();

    void increaseDivision();

    void chanceEnergy(int m_energy);

    void cellHurt();
};