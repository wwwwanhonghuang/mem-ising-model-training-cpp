#ifndef METROPOLIS_ALGORITHM_SAMPLING_HPP
#define METROPOLIS_ALGORITHM_SAMPLING_HPP

std::vector<char> initialize_configuration(int n_sites);

std::vector<char> flip_configuration(const std::vector<char>& configuration);
std::vector<std::vector<char>> 
metropolis_mcmc_sampling(std::shared_ptr<IsingModel> ising_model, 
                         std::shared_ptr<IsingInferencer> inferencer,
                         int n_samples, double beta, int equilibrium_steps);
#endif