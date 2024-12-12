#ifndef MEM_TRAINER_HPP
#define MEM_TRAINER_HPP
#include <vector>
#include <memory>
#include <iostream>
#include <unordered_map>
#include "models/ising_model.hpp"
#include "utils/ising_model_utils.hpp"
#include "inferencer/inferencer.hpp"


std::vector<double> calculate_observation_essembly_average_si(const std::vector<int>& observation_configurations, std::shared_ptr<IsingModel> ising_model);
std::vector<std::vector<double>> calculate_observation_essembly_average_si_sj(const std::vector<int>& observation_configurations, 
        std::shared_ptr<IsingModel> ising_model);
std::vector<double> calculate_model_proposed_essembly_average_si(const std::vector<int>& configurations, std::shared_ptr<IsingModel> ising_model, std::shared_ptr<IsingInferencer> ising_inferencer);
std::vector<std::vector<double>> calculate_model_proposed_essembly_average_si_sj(const std::vector<int>& configurations, std::shared_ptr<IsingModel> ising_model, std::shared_ptr<IsingInferencer> ising_inferencer);

struct IsingMEMTrainer{
    private:
    std::shared_ptr<IsingModel> ising_model;
    std::shared_ptr<IsingInferencer> ising_model_inferencer;
    std::vector<double> buffer_beta_H;
    std::vector<std::vector<double>> buffer_beta_J;
    bool require_evaluation;
    const std::vector<int>& train_configurations;
    const std::vector<int>& observation_configurations;
    double alpha;
    std::unordered_map<int, double> observation_configuration_possibility_map;
    
    public:
    IsingMEMTrainer(std::shared_ptr<IsingModel> ising_model, 
                    std::shared_ptr<IsingInferencer> inferencer, 
                    const std::vector<int>& train_configurations, 
                    const std::vector<int>& observation_configurations,
                    double alpha, bool require_evaluation);

    void prepare_training();
    void update_model_parameters();

    void update_model_partition_functions();

    void evaluation();

    void gradient_descending_step();
};
#endif