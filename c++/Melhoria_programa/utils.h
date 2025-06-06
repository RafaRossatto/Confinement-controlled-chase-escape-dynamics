#pragma once
#ifndef UTILS_H
#define UTILS_H
#include <cmath> // para std::abs
#include "config.h" // Para usar WIDTH e HEIGHT, se necessário
#include <iostream>
#include <fstream>

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

inline std::string gerarNomeArquivo(int numCT, int numcC, int numPoint,
    double ncNoise_cc, double ncNoise_ct,
    int SR_value) 
{   
return "NC_" + std::to_string(numCT) +
"_NE_" + std::to_string(numcC) +
"_O_" + std::to_string(numPoint) +
"_TCC_" + std::to_string(ncNoise_cc) +
"_TCT_" + std::to_string(ncNoise_ct) +
"_SR_" + std::to_string(SR_value) + ".dat";
}

#endif // UTILS_H
