#include "cell_lattice.h"
#include <iostream>

int main() {
    int largura = 100;
    int altura = 100;

    CellLattice lattice(largura, altura);

    // Adiciona obstáculos
    lattice.addObstacle(10, 20);
    lattice.addObstacle(15, 25);

    // Verifica se posição está bloqueada
    if (lattice.isObstacle(10, 20)) {
        std::cout << "Tem obstáculo em (10, 20)\n";
    }

    return 0;
}
