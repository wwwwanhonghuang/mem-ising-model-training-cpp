#include <vector>
#include "utils/ising_model_utils.hpp"

std::vector<char> to_binary_representation(int n_bits, int configuration){
    std::vector<char> binary_representation(n_bits, 0);
    int i = 0;
    while(configuration){
        binary_representation[i++] = configuration & 1;
        configuration >>= 1;
    }
    return binary_representation;
};
