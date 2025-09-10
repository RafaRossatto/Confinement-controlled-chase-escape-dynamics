#pragma once
#ifndef SIMULATION_H
#define SIMULATION_H

#include "cell_lattice.h"
#include "obstacle.h"
#include "config.h"
#include <string>
#include <vector>
#include <random>
#include <fstream>

class Simulation {
private:
    CellLattice& lattice;
    int numCT;
    int numcC;
    int numPoint;
    double ncNoise_ct;
    double ncNoise_cc;
    std::string fileName;
    std::vector<Obstacle> obstacles;
    int sr_normal;
    int sr_cancer;
    unsigned int seed;

public:
    Simulation(CellLattice& lattice_, int numCT_, int numcC_, int numPoint_,
               double ncNoise_ct_, double ncNoise_cc_, 
               const std::vector<Obstacle>& obstacles_, int sr_normal_, int sr_cancer_,
               unsigned int seed_ = 0);
    
    // Mantém a mesma assinatura - recebe o arquivo para escrita
    void runSingle(int run, std::mt19937& rng, std::ofstream& outputFile);
    
    // Getters
    int getNumCT() const { return numCT; }
    int getNumcC() const { return numcC; }
    int getNumPoint() const { return numPoint; }
    std::string getFileName() const { return fileName; }
    unsigned int getSeed() const { return seed; }
};

#endif // SIMULATION_H