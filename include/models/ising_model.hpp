#ifndef ISING_MODEL_HPP
#define ISING_MODEL_HPP

#include <vector>
#include <random>

class IsingModel {
public:
    std::vector<std::vector<double>> J;
    std::vector<double> H;
    int n_sites;
    double temperature;

    // Constructor
    IsingModel(int n_sites, double temperature);

};

#endif  // ISING_MODEL_HPP
