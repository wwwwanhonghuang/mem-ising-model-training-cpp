#include <iostream>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
#include <random>
#include <cassert>
#include "macros.def"
#include "models/ising_model.hpp"

#include "utils/ising_model_utils.hpp"
#include "inferencer/inferencer.hpp"
#include "mem_training/mem_trainer.hpp"
#include "utils/ising_io.hpp"



int main(){   
    // 1. Read configurations.
    YAML::Node config;
    try{
        config = YAML::LoadFile("config.yaml");        
    }catch(const YAML::Exception& e){
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    int temp_n = config["mem-trainer"]["n"].as<int>();
    if (temp_n < 0) {
        std::cerr << "Error: n cannot be negative." << std::endl;
        return -1;
    }
    size_t n = static_cast<size_t>(temp_n);    
    bool require_evaluation = config["mem-trainer"]["evaluation"].as<bool>();
    double alpha = config["mem-trainer"]["training"]["alpha"].as<double>();
    int iterations = config["mem-trainer"]["iterations"].as<int>();
    std::string spin_configuration_training_data_file_path = config["mem-trainer"]["ising_training_samples_spin_configurations"].as<std::string>();
    std::string spin_configuration_observation_data_file_path = config["mem-trainer"]["ising_observation_spin_configurations"].as<std::string>();
    if (spin_configuration_training_data_file_path.empty()) {
        std::cerr << "Error: Training data file path is empty." << std::endl;
        return -1;
    }

    if (spin_configuration_observation_data_file_path.empty()) {
        std::cerr << "Error: Observation data file path is empty." << std::endl;
        return -1;
    }
    std::cout << "Read spin configurations..." << std::endl;


    // 2. Necessary Instances
    std::shared_ptr<IsingModel> ising_model = std::make_shared<IsingModel>(n, 1.0);
    std::shared_ptr<IsingInferencer> ising_model_inferencer = std::make_shared<IsingInferencer>();
    
    // 3. Read ising spin configuration data for training. 
    std::cout << " - Load samples for training from file " << spin_configuration_training_data_file_path << std::endl;

    std::vector<int> training_configurations = 
        ISINGIO::read_spin_configurations(spin_configuration_training_data_file_path);

    std::cout << " - Loaded samples for training from file  configuration count = " << training_configurations.size() << std::endl;
    
    // 4. Read observation_configurations
    std::cout << " - Load observation samples from file " << spin_configuration_observation_data_file_path << std::endl;

    std::vector<int> observation_configurations = 
        ISINGIO::read_spin_configurations(spin_configuration_observation_data_file_path);
    std::cout << " - Read spin configurations finished. configuration count = " << observation_configurations.size() << std::endl;    
    
    // 5. Training Loop
    std::shared_ptr<IsingMEMTrainer> ising_model_mem_trainer = 
        std::make_shared<IsingMEMTrainer>(ising_model, ising_model_inferencer, training_configurations, observation_configurations, alpha, require_evaluation);

    std::cout << "Enter training loop..." << std::endl;

    for(int iteration_id = 0; iteration_id < iterations; iteration_id++){
        std::cout << "Iteration " << iteration_id + 1  << "/" << iterations << ":" << std::endl;
        std::cout << "  - " << "Calculate gradients..." << std::endl;
        ising_model_mem_trainer->gradient_descending_step();
        std::cout << "  - " << "Update model parameters..." << std::endl;
        ising_model_mem_trainer->update_model_parameters();
        std::cout << "  - " << "Update model partition functions..." << std::endl;
        ising_model_mem_trainer->update_model_partition_functions();

        //　Evaluation
        if(require_evaluation){
            std::cout << "  - " << "Evaluate reliability..." << std::endl;
            ising_model_mem_trainer->evaluation();
        }
        std::cout << std::endl;
        
    }
    return 0;
}