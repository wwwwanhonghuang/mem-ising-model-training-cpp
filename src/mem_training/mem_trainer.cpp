#include "mem_training/mem_trainer.hpp"
#include <cmath>

std::vector<double> calculate_observation_essembly_average_si(const std::vector<int>& observation_configurations, 
        std::shared_ptr<IsingModel> ising_model){
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

void IsingMEMTrainer::update_model_parameters(){
    for (int i = 0; i < ising_model->n_sites; i++) {
        ising_model->H[i] += buffer_beta_H[i];
    }

    for (int i = 0; i < ising_model->n_sites; i++) {
        for (int j = 0; j < ising_model->n_sites; j++) {
            ising_model->J[i][j] += buffer_beta_J[i][j];
        }
    }
};

void IsingMEMTrainer::update_model_partition_functions(){
    ising_model_inferencer->update_partition_function(ising_model, train_configurations, require_evaluation);
};


void IsingMEMTrainer::evaluation(){
    double S1 = 0.0; // order-1 entropy
    double S2 = 0.0; // order-2 entropy
    double SN = 0.0; // empirical entropy
    for(int configuration : train_configurations){
        double p2 = ising_model_inferencer->calculate_configuration_possibility(ising_model, to_binary_representation(ising_model->n_sites, configuration), 2);
        double p1 = ising_model_inferencer->calculate_configuration_possibility(ising_model, to_binary_representation(ising_model->n_sites, configuration), 1);
        S2 += -p2 * std::log(p2);
        S1 += -p1 * std::log(p1);
        SN += observation_configuration_possibility_map.find(configuration) != observation_configuration_possibility_map.end() 
            ? -observation_configuration_possibility_map[configuration] * std::log(observation_configuration_possibility_map[configuration]): 0;
    }

    double r_s = (S1 - S2) / (S1 - SN);
    
    double D1 = 0.0;
    double D2 = 0.0;
    for(auto& configuration_record : observation_configuration_possibility_map){
        D1 += configuration_record.second * 
            std::log(configuration_record.second / 
                ising_model_inferencer->calculate_configuration_possibility(ising_model, to_binary_representation(ising_model->n_sites, configuration_record.first), 1));
        D2 += configuration_record.second * 
            std::log(configuration_record.second / 
                ising_model_inferencer->calculate_configuration_possibility(ising_model, to_binary_representation(ising_model->n_sites, configuration_record.first), 2));
    }

    double r_d = (D1 - D2) / D1;
    std::cout << "r_d = " << r_d << std::endl;
    std::cout << "Reliablity: r_s / r_d = " << r_s / r_d << std::endl;
};



void IsingMEMTrainer::gradient_descending_step(){
    
    std::vector<double> obs_essembly_avgerage_si = 
        calculate_observation_essembly_average_si(observation_configurations, ising_model);

    std::vector<std::vector<double>> obs_essembly_avgerage_si_sj = 
        calculate_observation_essembly_average_si_sj(observation_configurations, ising_model);

    std::vector<double> model_essembly_avgerage_si = 
        calculate_model_proposed_essembly_average_si(train_configurations, 
            ising_model, ising_model_inferencer);

    std::vector<std::vector<double>> model_essembly_avgerage_si_sj = 
        calculate_model_proposed_essembly_average_si_sj(train_configurations, 
            ising_model, ising_model_inferencer);


    // calculate delta H, and update H
    for(int i = 0; i < ising_model->n_sites; i++){
        double delta = alpha * std::log(obs_essembly_avgerage_si[i] / model_essembly_avgerage_si[i]); 
        buffer_beta_H[i] = delta;
    }

    // calculate delta J, and update J
    for(int i = 0; i < ising_model->n_sites; i++){
        for(int j = 0; j < ising_model->n_sites; j++){
            double delta = alpha * 
                std::log(obs_essembly_avgerage_si_sj[i][j] / model_essembly_avgerage_si_sj[i][j]); 
            buffer_beta_J[i][j] = delta;
        }
    }
}

IsingMEMTrainer::IsingMEMTrainer(std::shared_ptr<IsingModel> ising_model, 
                    std::shared_ptr<IsingInferencer> inferencer, 
                    const std::vector<int>& train_configurations, 
                    const std::vector<int>& observation_configurations,
                    double alpha, bool require_evaluation)
        : ising_model(ising_model), 
          ising_model_inferencer(inferencer), 
          train_configurations(train_configurations), 
          observation_configurations(observation_configurations),
          alpha(alpha), 
          require_evaluation(require_evaluation) {
        
    buffer_beta_H.resize(ising_model->n_sites, 0.0);
    buffer_beta_J.resize(ising_model->n_sites, std::vector<double>(ising_model->n_sites, 0.0));
    // Build observation frequency map.
    
    for(int configuration : observation_configurations){
        observation_configuration_possibility_map[configuration]++;
    }
    for(auto& configuration : observation_configuration_possibility_map){
        observation_configuration_possibility_map[configuration.first] /= observation_configurations.size();
    }
}