#ifndef ISING_MODEL_UTILS_HPP
#define ISING_MODEL_UTILS_HPP
#include <vector>
#include <memory>
#include "models/ising_model.hpp"

std::vector<char> to_binary_representation(int n_bits, int configuration);
void random_initialize_ising_model(std::shared_ptr<IsingModel> ising_model, double j_min, double j_max, double h_min, double h_max);
#endif