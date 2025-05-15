#include <string>
class Cell 
{
    private:
    std:: string m_tipo;   /**< The type of the agent. */
    int m_numero;          /**< The number of the agent. */ 
    int m_coordenadaX; /**< The x-coordinate of the agent's position. */
    int m_coordenadaY; /**< The y-coordinate of the agent's position. */

    public:
    
        Cell(const std::string& tipo, int numero, int coordenadaX, int coordenadaY);
    

    // Getters
    std::string getTipo() const; 
    int getNumero() const; 
    int getCoordenadaX() const; 
    int getCoordenadaY() const;
    
    // Método para alterar a posição
    void changePosition(int newX,int newY);
};