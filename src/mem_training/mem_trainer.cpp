#include "mem_training/mem_trainer.hpp"
#include "utils/ising_model_utils.hpp"
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
            essembly_average[i] += (v[i] == 0 ? 0 : possibility);
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



inline double laplace_smoothing(double p, int n_configurations){
    const double e = 1e-39;
    return (p + e) / (e * n_configurations + 1);
}
double IsingMEMTrainer::evaluation(){
    double S1 = 0.0; // order-1 entropy
    double S2 = 0.0; // order-2 entropy
    double SN = 0.0; // empirical entropy
    for(int configuration : train_configurations){
        double p2 = 
            ising_model_inferencer->calculate_configuration_possibility(ising_model, to_binary_representation(ising_model->n_sites, configuration), 2);
        double p1 = 
            ising_model_inferencer->calculate_configuration_possibility(ising_model, to_binary_representation(ising_model->n_sites, configuration), 1);
        p2 = laplace_smoothing(p2, n_configurations);
        p1 = laplace_smoothing(p1, n_configurations);
        S2 += -p2 * std::log(p2);
        S1 += -p1 * std::log(p1);

        double p_observation = 0.0;
        if(observation_configuration_possibility_map.find(configuration) != observation_configuration_possibility_map.end()){
            p_observation = observation_configuration_possibility_map[configuration];
        }
        p_observation = laplace_smoothing(p_observation, n_configurations);
        SN += -p_observation * std::log(p_observation);
    }

    double r_s = (S1 - S2) / (S1 - SN);
    
    double D1 = 0.0;
    double D2 = 0.0;
    for(int configuration : train_configurations){
        double obs_possibility = 0.0;
        if(observation_configuration_possibility_map.find(configuration) != observation_configuration_possibility_map.end()){
            obs_possibility = observation_configuration_possibility_map[configuration];
        }
        obs_possibility = laplace_smoothing(obs_possibility, n_configurations);

        double model_possibility_order_1 = ising_model_inferencer->calculate_configuration_possibility(ising_model, to_binary_representation(ising_model->n_sites, configuration), 1);
        double model_possibility_order_2 = ising_model_inferencer->calculate_configuration_possibility(ising_model, to_binary_representation(ising_model->n_sites, configuration), 2);
        model_possibility_order_1 = laplace_smoothing(model_possibility_order_1, n_configurations);
        model_possibility_order_2 = laplace_smoothing(model_possibility_order_2, n_configurations);

        D1 += obs_possibility * std::log(obs_possibility / model_possibility_order_1);
        D2 += obs_possibility * std::log(obs_possibility / model_possibility_order_2);
    }

    double r_d = (D1 - D2) / D1;
    std::cout << "r_d = " << r_d << std::endl;
    std::cout << "Reliablity: r_s / r_d = " << r_s / r_d << std::endl;
    return r_s / r_d;
};


void print_si(const std::vector<double>& si){
    for(int i = 0; i < si.size(); i++){
        std::cout << si[i] << " ";
    }
    std::cout << std::endl;
}

void print_sisj(const std::vector<std::vector<double>>& sisj){
    for(int i = 0; i < sisj.size(); i++){
        const std::vector<double>& row = sisj[i];
        for(int j = 0; j < row.size(); j++){
            std::cout << sisj[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void IsingMEMTrainer::gradient_descending_step(){
    std::cout << "      - begin gradient_descending_step()" << std::endl;

    std::cout << "        - calculate obs_essembly_avgerage_si" << std::endl;
    std::vector<double> obs_essembly_avgerage_si = 
        calculate_observation_essembly_average_si(observation_configurations, ising_model);
    print_si(obs_essembly_avgerage_si);

    std::cout << "        - calculate obs_essembly_avgerage_si_sj" << std::endl;
    std::vector<std::vector<double>> obs_essembly_avgerage_si_sj = 
        calculate_observation_essembly_average_si_sj(observation_configurations, ising_model);
    print_sisj(obs_essembly_avgerage_si_sj);


    std::cout << "        - calculate model_essembly_avgerage_si" << std::endl;
    std::vector<double> model_essembly_avgerage_si = 
        calculate_model_proposed_essembly_average_si(train_configurations, 
            ising_model, ising_model_inferencer);
    print_si(model_essembly_avgerage_si);

    std::cout << "        - calculate model_essembly_avgerage_si_sj" << std::endl;
    std::vector<std::vector<double>> model_essembly_avgerage_si_sj = 
        calculate_model_proposed_essembly_average_si_sj(train_configurations, 
            ising_model, ising_model_inferencer);
    print_sisj(model_essembly_avgerage_si_sj);


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
    n_configurations = 1 << ising_model->n_sites;
}

void IsingMEMTrainer::prepare_training(){
    std::cout << "prepare trainning.." << std::endl;
    std::cout << "  - prepare partition functions.." << std::endl;
    ising_model_inferencer->update_partition_function(ising_model, train_configurations, require_evaluation);
    std::cout << "  - partition functions Z1 = " << ising_model_inferencer->get_Z(1) << std::endl;
    std::cout << "  - partition functions Z2 = " << ising_model_inferencer->get_Z(2) << std::endl;
}