#ifndef CONFIG_H
#define CONFIG_H
#include<vector>

// Tamanho do grid
const int NORTH = 0;          /**< Direction: North */
const int EAST = 1;           /**< Direction: East */
const int SOUTH = 2;          /**< Direction: South */
const int WEST = 3;           /**< Direction: West */

const int SIZE = 100;
const int WIDTH = SIZE;
const int HEIGHT = SIZE;

const double CAPTUREPROBABILITY = 1.00;

// Raio progressivo utilizado na busca
const std::vector<int> RAIOS_PROGRESSIVOS = {5, 10, 20, 30, 40, 50};

#endif // CONFIG_H
