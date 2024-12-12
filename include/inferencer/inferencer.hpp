#ifndef INFERENCE_HPP
#define INFERENCE_HPP

#include <memory>
#include <vector>
#include <cmath>
#include "models/ising_model.hpp"

class IsingInferencer {
public:
    // Function prototypes
    void update_partition_function(std::shared_ptr<IsingModel> ising_model, std::vector<int> configurations, bool update_order_1_partition_function = false);
    double energy(std::shared_ptr<IsingModel> ising_model, const std::vector<char>& configuration, int order = 2);
    double calculate_configuration_possibility(std::shared_ptr<IsingModel> ising_model, const std::vector<char>& configuration, int order = 2);

private:
    double Z1 = 0.0;
    double Z2 = 0.0;
};

#endif
