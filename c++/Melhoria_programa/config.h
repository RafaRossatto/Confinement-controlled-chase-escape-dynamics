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
//const std::vector<int> RAIOS_PROGRESSIVOS = {5, 10, 20, 30, 40, 50};

const std::vector<int> RAIOS_PROGRESSIVOS = [](){
    std::vector<int> r;
    for (int i = 1; i <= 71; ++i)
        r.push_back(i);
    return r;
}();
#endif // CONFIG_H
