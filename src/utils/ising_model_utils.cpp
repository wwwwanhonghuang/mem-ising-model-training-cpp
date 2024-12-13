#include <vector>
#include "utils/ising_model_utils.hpp"


std::vector<char> to_binary_representation(int n_bits, int configuration){
    std::vector<char> binary_representation(n_bits, 0);
    int i = 0;
    while(configuration){
        binary_representation[i++] = configuration & 1;
        configuration >>= 1;
    }
    return binary_representation;
};


// Method to initialize J and H with random values
void random_initialize_ising_model(std::shared_ptr<IsingModel> ising_model, double j_min, double j_max, double h_min, double h_max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> j_dist(j_min, j_max);
    std::normal_distribution<double> h_dist(h_min, h_max);

    for (int i = 0; i < ising_model->n_sites; ++i) {
        for (int j = 0; j < ising_model->n_sites; ++j) {
            double random_value = j_dist(gen);
            ising_model->J[i][j] = random_value;
        }
        ising_model->J[i][i] = 0.0;  // Zero diagonal
    }

    // Randomly initialize H
    for (int i = 0; i < ising_model->n_sites; ++i) {
        ising_model->H[i] = h_dist(gen);
    }
}


