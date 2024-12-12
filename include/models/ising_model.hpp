#ifndef ISING_MODEL_HPP
#define ISING_MODEL_HPP
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

class IsingModel{
public:
    std::vector<std::vector<double>> J;
    std::vector<double> H;
    int n_sites;
    double temperature;

    IsingModel(int n_sites){
        this->n_sites = n_sites;
        J = std::vector<std::vector<double>>(n_sites, std::vector<double>(n_sites, 0.0));
        H = std::vector<double>(n_sites, 0.0);
        temperature = 1.0;
    }

    IsingModel(const std::string& model_filepath){
        load_from_file(model_filepath);
    }

    void serialize_to_file(const std::string& model_filepath){
        std::ofstream out_file(model_filepath);
        if (!out_file.is_open()) {
            std::cerr << "Error: Could not open file for writing.\n";
            return;
        }

        // Save the number of sites and temperature
        out_file << n_sites << "\n";
        out_file << temperature << "\n";

        for (const auto& row : J) {
            for (double value : row) {
                out_file << value << " ";
            }
            out_file << "\n";
        }

        for (double h : H) {
            out_file << h << " ";
        }
        out_file << "\n";

        out_file.close();
    }
    private:
    // Method to load the model from a file
    void load_from_file(const std::string& model_filepath) {
        std::ifstream in_file(model_filepath);
        if (!in_file.is_open()) {
            std::cerr << "Error: Could not open file for reading.\n";
            return;
        }

        // Load the number of sites and temperature
        in_file >> n_sites;
        in_file >> temperature;

        // Resize J and H based on the number of sites
        J = std::vector<std::vector<double>>(n_sites, std::vector<double>(n_sites, 0.0));
        H = std::vector<double>(n_sites, 0.0);

        // Load the interaction matrix J
        for (int i = 0; i < n_sites; ++i) {
            for (int j = 0; j < n_sites; ++j) {
                in_file >> J[i][j];
            }
        }

        // Load the external field H
        for (int i = 0; i < n_sites; ++i) {
            in_file >> H[i];
        }

        in_file.close();
    };
    void random_initialize(double j_min = -1.0, double j_max = 1.0, double h_min = -1.0, double h_max = 1.0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> j_dist(j_min, j_max);
        std::uniform_real_distribution<double> h_dist(h_min, h_max);

        // Randomly initialize J (symmetric matrix with zero diagonal)
        for (int i = 0; i < n_sites; ++i) {
            for (int j = i + 1; j < n_sites; ++j) {
                double random_value = j_dist(gen);
                J[i][j] = random_value;
                J[j][i] = random_value;  // Ensure symmetry
            }
            J[i][i] = 0.0;  // Zero diagonal
        }

        // Randomly initialize H
        for (int i = 0; i < n_sites; ++i) {
            H[i] = h_dist(gen);
        }
    }
};
#endif
