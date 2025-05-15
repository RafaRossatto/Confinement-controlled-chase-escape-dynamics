#ifndef UTILS_H
#define UTILS_H

#include <cmath> // para std::abs
#include "config.h" // Para usar WIDTH e HEIGHT, se necessário
#include <iostream>
#include <fstream>

// Função para calcular distância considerando condições periódicas
inline double calculeDistance(int x1, int y1, int x2, int y2) {
    int dx = x2 - x1;
    if (dx < 0) dx = -dx;
    if (dx > WIDTH / 2) dx = WIDTH - dx;

    int dy = y2 - y1;
    if (dy < 0) dy = -dy;
    if (dy > HEIGHT / 2) dy = HEIGHT - dy;

    return dx + dy;
}

inline void logInfo(const std::string& mensagem) 
{
    std::cout << "[INFO] " << mensagem << std::endl;
}

inline void logWarning(const std::string& mensagem) 
{
    std::cout << "[WARNING] " << mensagem << std::endl;
}

inline void logError(const std::string& mensagem) 
{
    std::cerr << "[ERROR] " << mensagem << std::endl;
}

// Função para verificar se um índice é válido
inline bool indiceValido(int indice, int tamanho)
{
    return (indice >= 0) && (indice < tamanho);
}


inline void moveTowardsPoint(int &x, int &y, int targetX, int targetY) 
{
    // Se a posição X ainda não está alinhada com o alvo, mover primeiro no eixo X
    if (x != targetX) 
    {
        if (x < targetX) 
        {
            x = (x + 1) % WIDTH; // Move para a direita
        } 
        else 
        {
            x = (x - 1 + WIDTH) % WIDTH; // Move para a esquerda
        }
    }
    // Se a posição X já está alinhada, mover no eixo Y
    else if (y != targetY) 
    {
        if (y < targetY) 
        {
            y = (y + 1) % HEIGHT; // Move para cima
        } 
        else 
        {
            y = (y - 1 + HEIGHT) % HEIGHT; // Move para baixo
        }
    }
}

inline void randonWalk(int& x, int& y,int move)
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

inline void openFile(const std::string& fileName) 
{
    // Abre o arquivo em modo de escrita e apêndice
    std::ofstream file(fileName, std::ios::app);

    // Check if the file was opened successfully
    if (!file.is_open()) 
    {
        logError("Erro ao abrir o arquivo " + fileName);
        return;
    }

    file.close();
}

inline bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good(); // Verifica se o arquivo pode ser aberto
}


#endif // UTILS_H
