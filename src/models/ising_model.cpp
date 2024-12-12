#include "models/ising_model.hpp"
#include <random>

// Constructor
IsingModel::IsingModel(int n_sites, double temperature) {
    this->n_sites = n_sites;
    J = std::vector<std::vector<double>>(n_sites, std::vector<double>(n_sites, 0.0));
    H = std::vector<double>(n_sites, 0.0);
    this->temperature = temperature;
}
