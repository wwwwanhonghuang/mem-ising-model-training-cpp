#include "models/ising_model.hpp"
#include <random>

// Constructor
IsingModel::IsingModel(int n_sites, double temperature) {
    this->n_sites = n_sites;
    J = std::vector<std::vector<double>>(n_sites, std::vector<double>(n_sites, 0.0));
    H = std::vector<double>(n_sites, 0.0);
    this->temperature = temperature;
}

// Method to initialize J and H with random values
void IsingModel::random_initialize(double j_min, double j_max, double h_min, double h_max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> j_dist(j_min, j_max);
    std::uniform_real_distribution<double> h_dist(h_min, h_max);

    // Randomly initialize J (symmetric matrix with zero diagonal)
    for (int i = 0; i < n_sites; ++i) {
        for (int j = 0; j < n_sites; ++j) {
            double random_value = j_dist(gen);
            J[i][j] = random_value;
            J[j][i] = random_value;  // Ensure symmetry
        }
        J[i][i] = 0.0;  // Zero diagonal
    }

    // Randomly initialize H
    for (int i = 0; i < n_sites; ++i) {
        H[i] = h_dist(gen);
    }
}
