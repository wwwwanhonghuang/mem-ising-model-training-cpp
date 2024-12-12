#include <iostream>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
#include <random>
#include <cassert>
std::vector<char> to_binary_representation(int n_bits, int configuration);


class IsingInferencer{
public:
    inline void update_partition_function(std::shared_ptr<IsingModel> ising_model, std::vector<int> configurations, bool update_order_1_partition_function = false){
        Z2 = 0.0;
        for(int configuration : configurations){
            std::vector<char> v = to_binary_representation(ising_model->n_sites, configuration);
            Z2 += std::exp(-energy(ising_model, v) / ising_model->temperature);
            if(update_order_1_partition_function){
                Z1 += std::exp(-energy(ising_model, v, 1) / ising_model->temperature);

            }
        }       
    }
    inline double energy(std::shared_ptr<IsingModel> ising_model, const std::vector<char>& configuration, int order = 2){
        double energy = 0.0;
        if(order != 1 && order != 2){
            std::cout << "Error: Order shold equal to 1 or 2." << std::endl;
            assert(false);
        }

        if(order == 2){
            for (int i = 0; i < ising_model->n_sites; i++) {
                for (int j = i + 1; j < ising_model->n_sites; j++) {
                    energy += -ising_model->J[i][j] * configuration[i] * configuration[j];
                }
            }
        }
        for (int i = 0; i < ising_model->n_sites; i++) {
            energy += ising_model->H[i] * configuration[i];
        }

        return energy;
    }
    double calculate_configuration_possibility(std::shared_ptr<IsingModel> ising_model, const std::vector<char>& configuration, int order = 2){
        if(order == 2){
            return std::exp(-energy(ising_model, configuration) / ising_model->temperature) / Z2;
        }
        if(order == 1){
            return std::exp(-energy(ising_model, configuration, 1) / ising_model->temperature) / Z1;
        }
        std::cout << "model inference order should equal to 1 or 2" << std::endl;
        assert(false);
    }
private:
    double Z1 = 0.0;
    double Z2 = 0.0;
};

std::vector<char> to_binary_representation(int n_bits, int configuration){
    std::vector<char> binary_representation(n_bits, 0);
    int i = 0;
    while(configuration){
        binary_representation[i++] = configuration & 1;
        configuration >>= 1;
    }
    return binary_representation;
};

std::vector<double> calculate_observation_essembly_average_si(const std::vector<int>& observation_configurations, std::shared_ptr<IsingModel> ising_model){
    std::vector<double> essembly_average(ising_model->n_sites, 0);
    
    for(int configuration : observation_configurations){
        std::vector<char> v = to_binary_representation(ising_model->n_sites, configuration);
        for(int i = 0; i < v.size(); i++){
            essembly_average[i] += v[i];
        }
    }

    for(int i = 0; i < ising_model->n_sites; i++){
        essembly_average[i] /= observation_configurations.size();
    }
    return essembly_average;
};

std::vector<std::vector<double>> calculate_observation_essembly_average_si_sj(const std::vector<int>& observation_configurations, 
        std::shared_ptr<IsingModel> ising_model){
    std::vector<std::vector<double>> essembly_average(ising_model->n_sites, std::vector<double>(ising_model->n_sites, 0));
    for(int configuration : observation_configurations){
        std::vector<char> v = to_binary_representation(ising_model->n_sites, configuration);
        for(int i = 0; i < ising_model->n_sites; i++){
            for(int j = 0; j < ising_model->n_sites; j++){
                if(v[i] == 1 && v[j] == 1) essembly_average[i][j] += 1;
            }
        }
    }
    for(int i = 0; i < ising_model->n_sites; i++){
        for(int j = 0; j < ising_model->n_sites; j++){
            essembly_average[i][j] /= observation_configurations.size();
        }
    }
    return essembly_average;
};

std::vector<double> calculate_model_proposed_essembly_average_si(const std::vector<int>& configurations, std::shared_ptr<IsingModel> ising_model, std::shared_ptr<IsingInferencer> ising_inferencer){
    std::vector<double> essembly_average(ising_model->n_sites, 0);
    for(int configuration : configurations){
        std::vector<char> v = to_binary_representation(ising_model->n_sites, configuration);
        double possibility = ising_inferencer->calculate_configuration_possibility(ising_model, v);

        for(int i = 0; i < v.size(); i++){
            essembly_average[i] += v[i] == 0 ? 0 : possibility;
        }
    }

    for(int i = 0; i < ising_model->n_sites; i++){
        essembly_average[i] /= configurations.size();
    }
    return essembly_average;
};

std::vector<std::vector<double>> calculate_model_proposed_essembly_average_si_sj(const std::vector<int>& configurations, std::shared_ptr<IsingModel> ising_model, std::shared_ptr<IsingInferencer> ising_inferencer){
    std::vector<std::vector<double>> essembly_average(ising_model->n_sites, std::vector<double>(ising_model->n_sites, 0));
    for(int configuration : configurations){
        std::vector<char> v = to_binary_representation(ising_model->n_sites, configuration);
        double possibility = ising_inferencer->calculate_configuration_possibility(ising_model, v);
        for(int i = 0; i < ising_model->n_sites; i++){
            for(int j = 0; j < ising_model->n_sites; j++){
                if(v[i] == 1 && v[j] == 1) 
                    essembly_average[i][j] += 1 * possibility;
            }
        }
    }
    
    return essembly_average;
};

void reading_observation_configurations(std::vector<std::string>& configurations){

}

int main(){
    // 1. Reading configurations.
    YAML::Node config;
    try{
        config = YAML::LoadFile("config.yaml");        
    }catch(const YAML::Exception& e){
        std::cerr << "Error: cannot open configuration file." << std::endl;
        return -1;
    }
    size_t n = config["mem-trainer"]["n"].as<int>();
    bool require_evaluation = config["mem-trainer"]["evaluation"].as<bool>();
    int iterations = config["mem-trainer"]["iterations"].as<int>();
    std::string training_data_file_path = config["mem-trainer"]["ising_training_configurations"].as<std::string>();


    std::ifstream input_training_data(training_data_file_path, std::ios::binary);
    std::vector<int> configurations_for_training;

    if(!input_training_data){
        std::cerr << "Error: cannot open ising model configuration data file " << training_data_file_path << std::endl;
        return -1;
    }
    
    while (input_training_data) {
        std::vector<char> buffer(4);        
        input_training_data.read(buffer.data(), 4);
        std::streamsize bytes_read = input_training_data.gcount();
        
        if (bytes_read > 0) {
            int value = *reinterpret_cast<int*>(buffer.data());
            configurations_for_training.push_back(value);
        }
    }

    input_training_data.close();

    std::shared_ptr<IsingModel> ising_model = std::make_shared<IsingModel>(n);
    std::shared_ptr<IsingInferencer> ising_model_inferencer = std::make_shared<IsingInferencer>();


    std::vector<int> observation_configurations; 
    reading_observation_configurations(observation_configurations);

    std::unordered_map<int, double> observation_configuration_possibility_map;
    for(int configuration : observation_configurations){
        observation_configuration_possibility_map[configuration]++;
    }
    for(auto& configuration : observation_configuration_possibility_map){
        observation_configuration_possibility_map[configuration.first] /= observation_configurations.size();
    }
    
    for(int iteration_id = 0; iteration_id < iterations; iteration_id++){
        std::cout << "Iter " << iteration_id << ":" << std::endl;

        std::vector<double> obs_essembly_avgerage_si = 
            calculate_observation_essembly_average_si(observation_configurations, ising_model);

        std::vector<std::vector<double>> obs_essembly_avgerage_si_sj = 
            calculate_observation_essembly_average_si_sj(observation_configurations, ising_model);

        std::vector<double> model_essembly_avgerage_si = 
            calculate_model_proposed_essembly_average_si(configurations_for_training, 
                ising_model, ising_model_inferencer);
        std::vector<std::vector<double>> model_essembly_avgerage_si_sj = 
            calculate_model_proposed_essembly_average_si_sj(configurations_for_training, 
                ising_model, ising_model_inferencer);

        std::vector<double> delta_H(n, 0.0);
        std::vector<std::vector<double>> delta_J(n, std::vector<double>(n, 0.0));
        
        const double alpha = 0.7;


        
        // calculate delta H, and update H
        for(int i = 0; i < n; i++){
            double delta = alpha * std::log(obs_essembly_avgerage_si[i] / model_essembly_avgerage_si[i]); 
            ising_model->H[i] += delta;
        }

        // calculate delta J, and update J
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                double delta = alpha * 
                    std::log(obs_essembly_avgerage_si_sj[i][j] / model_essembly_avgerage_si_sj[i][j]); 
                ising_model->J[i][j] += delta;
            }
        }

        ising_model_inferencer->update_partition_function(ising_model, configurations_for_training, require_evaluation);

        // evaluation
        double S1 = 0.0; // order-1 entropy
        double S2 = 0.0; // order-2 entropy
        double SN = 0.0; // empirical entropy
        for(int configuration : configurations_for_training){
            double p2 = ising_model_inferencer->calculate_configuration_possibility(ising_model, to_binary_representation(n, configuration), 2);
            double p1 = ising_model_inferencer->calculate_configuration_possibility(ising_model, to_binary_representation(n, configuration), 1);
            S2 += -p2 * std::log(p2);
            S1 += -p1 * std::log(p1);
            SN += observation_configuration_possibility_map.find(configuration) != observation_configuration_possibility_map.end() 
                ? -observation_configuration_possibility_map[configuration] * std::log(observation_configuration_possibility_map[configuration]): 0;
        }

        double r_s = (S1 - S2) / (S1 - SN);
        
        double D1 = 0.0;
        double D2 = 0.0;
        for(auto& configuration_record : observation_configuration_possibility_map){
            D1 += configuration_record.second * std::log(configuration_record.second / ising_model_inferencer->calculate_configuration_possibility(ising_model, to_binary_representation(n, configuration_record.first), 1));
            D2 += configuration_record.second * std::log(configuration_record.second / ising_model_inferencer->calculate_configuration_possibility(ising_model, to_binary_representation(n, configuration_record.first), 2));
        }

        double r_d = (D1 - D2) / D1;
        std::cout << "r_d = " << r_d << std::endl;
        std::cout << "Reliablity: r_s / r_d = " << r_s / r_d << std::endl;
    }
    return 0;
}