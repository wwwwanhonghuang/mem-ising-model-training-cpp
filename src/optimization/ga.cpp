// #include <vector>
// #include <algorithm> // For std::sort
// #include <random>    // For random generation
// #include <iostream>  // For debugging
// #include <numeric>

// std::vector<int> generate_initial_population(int population_size) {
//     std::vector<int> population;
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<> dis(0, 100); // Example: Random integers between 0 and 100

//     for (int i = 0; i < population_size; ++i) {
//         population.emplace_back(dis(gen));
//     }
//     return population;
// }
// int fitness(int individual) {
//     return individual; // For example, higher values are better
// }

// int produce_one_individual(int parent_1, int parent_2) {
//     // Simple crossover: average the parents
//     int child = (parent_1 + parent_2) / 2;

//     // Mutation: Add a small random value
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<> mutation_dis(-5, 5); // Mutation range
//     child += mutation_dis(gen);

//     return child;
// }

// std::vector<int> select_parent(const std::vector<int>& current_solutions, int population_capacity) {
//     std::vector<int> selected_parents;

//     // Compute fitnesses
//     std::vector<int> fitnesses;
//     for (int individual : current_solutions) {
//         fitnesses.emplace_back(fitness(individual));
//     }

//     // Normalize fitness
//     int total_fitness = std::accumulate(fitnesses.begin(), fitnesses.end(), 0);
//     std::vector<double> probabilities;
//     for (int f : fitnesses) {
//         probabilities.emplace_back(static_cast<double>(f) / total_fitness);
//     }

//     // Randomly select two parents based on probabilities
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());

//     selected_parents.emplace_back(current_solutions[dist(gen)]);
//     selected_parents.emplace_back(current_solutions[dist(gen)]);

//     return selected_parents;
// }

// // Produce one individual (crossover + mutation)
// int produce_one_individual(int parent_1, int parent_2) {
//     // Simple crossover: average the parents
//     int child = (parent_1 + parent_2) / 2;

//     // Mutation: Add a small random value
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<> mutation_dis(-5, 5); // Mutation range
//     child += mutation_dis(gen);

//     return child;
// }

// // Genetic Algorithm
// void do_ga(const std::vector<int>& initial_population, int epoch, int max_individual_per_epoch, 
//             int population_capacity) {
//     std::vector<int> current_solutions(initial_population);

//     for (int epoch_id = 0; epoch_id < epoch; ++epoch_id) {
//         // Generate offspring until max individuals per epoch are reached
//         while (current_solutions.size() < max_individual_per_epoch) {
//             std::vector<int> parents = select_parent(current_solutions, population_capacity);
//             int individual = produce_one_individual(parents[0], parents[1]);
//             current_solutions.emplace_back(individual);
//         }

//         // Sort current solutions based on fitness
//         std::sort(current_solutions.begin(), current_solutions.end(), [](int a, int b) {
//             return fitness(a) > fitness(b); // Higher fitness first
//         });

//         // Trim population to the allowed capacity
//         if (current_solutions.size() > population_capacity) {
//             current_solutions.resize(population_capacity);
//         }

//         // Debugging output for current epoch
//         std::cout << "Epoch " << epoch_id + 1 << ":\n";
//         for (int individual : current_solutions) {
//             std::cout << individual << " (fitness: " << fitness(individual) << ")\n";
//         }
//         std::cout << "\n";
//     }
// }