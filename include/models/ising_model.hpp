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

    // Method to initialize J and H with random values
    void random_initialize(double j_min = -1.0, double j_max = 1.0, double h_min = -1.0, double h_max = 1.0);

};

#endif  // ISING_MODEL_HPP
