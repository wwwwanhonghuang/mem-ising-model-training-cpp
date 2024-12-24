#include "utils/metropolis_algorithm_sampling.hpp"
#include <iostream>
#include <vector>
#include <memory>
#include "models/ising_model.hpp"
#include "inferencer/inferencer.hpp"
#include "utils/ising_model_utils.hpp"

#include <cstdlib>  // For rand()
#include <ctime>    // For srand()

// Function to initialize the configuration randomly
std::vector<char> initialize_configuration(int n_sites) {
    std::vector<char> configuration(n_sites);

    // Initialize random seed (this can be done once globally in main)
    srand(time(0));

    // Assign random values of -1 or 1 to each spin
    for (int i = 0; i < n_sites; ++i) {
        configuration[i] = (rand() % 2 == 0) ? 1 : 0;  // Randomly assign +1 or 0
    }

    return configuration;
}


std::vector<char> flip_configuration(const std::vector<char>& configuration) {
    // Create a copy of the original configuration
    std::vector<char> new_configuration = configuration;

    // Randomly select an index to flip
    int n_sites = configuration.size();
    int index_to_flip = rand() % n_sites;  // Randomly choose an index between 0 and n_sites - 1

    // Flip the selected spin
    new_configuration[index_to_flip] = (new_configuration[index_to_flip] == 1) ? -1 : 1;

    // Return the new configuration with the flipped spin
    return new_configuration;
}

std::vector<std::vector<char>> 
metropolis_mcmc_sampling(std::shared_ptr<IsingModel> ising_model, 
                         std::shared_ptr<IsingInferencer> inferencer,
                         int n_samples, double beta, int equilibrium_steps) {
    std::vector<std::vector<char>> samples;
    int n_sites = ising_model->n_sites;

    std::vector<char> configuration_t = to_binary_representation(n_sites, initialize_configuration());
    std::vector<char> configuration_next_t = to_binary_representation(n_sites, initialize_configuration());

    // Equilibrium Phase
    for (int i = 0; i < equilibrium_steps; ++i) {
        // Flip a single spin or a subset of spins (Metropolis step)
        std::vector<char> new_configuration_t = flip_configuration(configuration_t);
        std::vector<char> new_configuration_next_t = flip_configuration(configuration_next_t);

        // Calculate energy before and after the flip
        double energy_before = inferencer->time_dependent_ising_energy(ising_model, configuration_t, configuration_next_t);
        double energy_after = inferencer->time_dependent_ising_energy(ising_model, new_configuration_t, new_configuration_next_t);

        // Accept or reject based on energy difference
        double delta_E = energy_after - energy_before;
        if (delta_E < 0 || std::exp(-beta * delta_E) > rand() / double(RAND_MAX)) {
            // Accept the new configuration
            configuration_t = new_configuration_t;
            configuration_next_t = new_configuration_next_t;
        }
    }

    std::cout << "Equilibrium Completed." << std::endl;

    // Sampling Phase
    while (samples.size() < n_samples) {
        // Propose new configurations by flipping some spins
        std::vector<char> new_configuration_t = flip_configuration(configuration_t);
        std::vector<char> new_configuration_next_t = flip_configuration(configuration_next_t);

        // Calculate energy before and after the flip
        double energy_before = inferencer->time_dependent_ising_energy(ising_model, configuration_t, configuration_next_t);
        double energy_after = inferencer->time_dependent_ising_energy(ising_model, new_configuration_t, new_configuration_next_t);

        // Metropolis-Hastings accept/reject step
        double delta_E = energy_after - energy_before;
        if (delta_E < 0 || std::exp(-beta * delta_E) > rand() / double(RAND_MAX)) {
            // Accept the new configuration
            samples.push_back(new_configuration_t); // Store accepted configuration
            configuration_t = new_configuration_t;
            configuration_next_t = new_configuration_next_t;
        }
    }

    return samples;
}
