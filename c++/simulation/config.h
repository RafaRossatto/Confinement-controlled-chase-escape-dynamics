// config.h
#ifndef CONFIG_H
#define CONFIG_H

extern int WIDTH;
extern int HEIGHT;
extern int SIZE;

extern double CAPTUREPROBABILITY;
// --- Probabilidades globais ---
extern double CTPROBABILITY;  // probabilidade de capturar ou procurar comida
extern double CCPROBABILITY;  // probabilidade de escapar ou procurar comida

// --- Funções de configuração ---
void setCTPROBABILITY(double newValue);
void setCCPROBABILITY(double newValue);


#endif // CONFIG_H