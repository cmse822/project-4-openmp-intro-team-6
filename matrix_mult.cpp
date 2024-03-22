#include <iostream>
#include <vector>
#include <stdexcept>
#include <random>
#include <fstream>
#include <algorithm>
#include <omp.h>
extern "C" {
#include "get_walltime.c"
}

const size_t REPEAT = 20;
const double lower_bound = 0.0f;
const double upper_bound = 10000.0f;
std::default_random_engine re;
std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
size_t N = 100;

/**
 * Performs matrix multiplication of two input matrices and stores the result in the third matrix.
 * @param a The first matrix for multiplication
 * @param b The second matrix for multiplication
 * @param c The matrix to store the result of the multiplication
 * @throws std::runtime_error If the incoming matrices are empty or not square
 */
void MatMul(    const   std::vector<std::vector<double>>& a, 
                const   std::vector<std::vector<double>>& b,
                        std::vector<std::vector<double>>& c) {    
    const auto& rows = a.size();
    const auto& cols = a[0].size();

    if (a.empty() || b.empty()) {
        throw std::runtime_error("Incoming matrix should not be empty!");
        exit(1);
    }

    if(rows != cols){
        throw std::runtime_error("Incoming matrix should be square!");
        exit(1);
    }

    #pragma omp parallel for collapse(2) 
    for (size_t i = 0; i < a.size(); i++) {
        for(size_t k=0; k < a[0].size(); k++){
            for(size_t j=0; j < a[0].size(); j++){
                // 8 Bytes * 3 (2 read, 1 store) = 24 bytes; 2 FLOPS (add, mult) per iter. 2/24 = 0.08333
                c[i][j] += a[i][k] * b[k][j];            
            }
        }
    }
}

/**
 * Prints the given matrix of type T.
 * @param matrix the matrix to be printed
 * @return void
 * @throws None
 */
template <typename T>
void PrintMatrix(std::vector<std::vector<T>> matrix){
    const auto& rows = matrix.size();
    const auto& cols = matrix[0].size();
    
    printf("###### Matrix #######\n");
    for (size_t i=0; i<rows; i++){
        printf("[");
        for (size_t j=0; j<cols; j++)
            printf("%f ", matrix[i][j]);
        printf("]\n");
    }
}

/**
 * Save results to a CSV file.
 * @param time_to_compute the computation time to do matrix multiplication
 * @throws None
 */
void SaveResultsToCSV(const std::string& filename, const std::string& thread_count, const std::string& time_to_compute) {
    if(filename.substr(filename.size() - 4) != ".csv"){
        std::cout << "Error: Invalid file format. Please specify a CSV file." << std::endl;
        return;
    }

    std::ifstream infile(filename, std::ios::in | std::ios::out | std::ios::binary);

    if(!infile){
        std::ofstream outfile(filename);
        outfile.close();
    }

    std::ofstream outfile(filename, std::ios::in | std::ios::out | std::ios::binary);
    std::string header_line = "matrix_size,thread_count,time_to_compute";

    outfile.seekp(0, std::ios::end);

    if(outfile.tellp() == 0){
        outfile << '\xEF' << '\xBB' << '\xBF';
        outfile << header_line << std::endl;
    }

    outfile.seekp(0, std::ios::end);
    outfile << N << "," << thread_count << "," << time_to_compute << std::endl;
    infile.close();
    outfile.close();
}

/**
 * Generates a random double using the unif pseudo-random number generator.
 * @return the generated random double
 */
double GenerateRandomNumber(){
    double a_random_double = unif(re);
    return a_random_double;
}

int main(int argc, char* argv[]) {
    if (argc > 1) {
        if (!std::isdigit(*argv[1]) || std::stoi(argv[1]) < 1 || std::stoi(argv[1]) > 10000) {
            throw std::runtime_error("Invalid matrix size argument. Please enter a numeric value between 1 and 10000.");
        }
        N = std::stoi(argv[1]);
    } else {
        std::cout << "No matrix size argument provided. Defaulting to N=100." << std::endl;
        N = 100;
    }

    int NUM_THREADS = 1;

    if(argc > 2) {
        if (!std::isdigit(*argv[2]) || std::stoi(argv[2]) < 1 || std::stoi(argv[2]) > 100) {
            throw std::runtime_error("Invalid number of threads argument. Please enter a numeric value between 1 and 100.");
        }
        NUM_THREADS = std::stoi(argv[2]);
    } else {
        std::cout << "No number of threads argument provided. Defaulting to 1 thread." << std::endl;
        NUM_THREADS = 1;
    }

    std::string FILENAME = "data_omp.csv";
    if(argc > 3) {
        printf("Filename: %s\n", argv[3]);
        FILENAME = std::string(argv[3]);
    }

    omp_set_num_threads(NUM_THREADS);

    std::vector<std::vector<double>> mat1(N, std::vector<double>(N));
    std::vector<std::vector<double>> mat2(N, std::vector<double>(N));
    
    printf("Starting the program with %ldx%ld matrices\n", N, N);

    for(size_t j=0; j < mat1[0].size(); j++){
        for(size_t k=0; k < mat1[0].size(); k++){
            mat1[j][k] = GenerateRandomNumber();
            mat2[j][k] = GenerateRandomNumber();
        }
    }

    double start_time = 0.0f;
    double end_time = 0.0f;
    double total_time = 0.0f;
    std::vector<std::vector<double>> result_mat(mat1[0].size(), std::vector<double>(mat1[0].size()));

    for(size_t i=0; i<REPEAT; i++){

        get_walltime(&start_time);
        MatMul(mat1, mat2, result_mat);
        get_walltime(&end_time);

        const double elapsed_time = end_time - start_time;
        total_time += elapsed_time;
    }

    printf("Writing results to %s\n", FILENAME.c_str());
    SaveResultsToCSV(FILENAME, std::to_string(NUM_THREADS), std::to_string(total_time));
    printf("Finished writing results to %s\nExiting.\n", FILENAME.c_str());

    printf("Finished matrix multiplication results using OpenMP.\n");
}