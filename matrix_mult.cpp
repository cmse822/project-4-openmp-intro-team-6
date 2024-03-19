#include <iostream>
#include <vector>
#include <stdexcept>
#include <random>
#include <fstream>
#include <algorithm>
extern "C" {
#include "get_walltime.c"
}

const std::string FILENAME = "data.csv";
const size_t REPEAT = 20;
const double lower_bound = 0.0f;
const double upper_bound = 10000.0f;
std::default_random_engine re;
std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
size_t N = 100;


struct ComputeComponent{
    std::string name;
    int num_cores;
    float clock_speed_ghz;
    int num_fpus;
};

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
 * Calculates the MFLOPS based on the total time taken.
 * @param total_time the total time taken for the operations
 * @return the calculated MFLOPS
 * @throws None
 */
double CalculateMFLOPS(double total_time){
    double flop = 2 * N * N * N * REPEAT;
    double mflops = flop / std::fmax(0.001f, (total_time * 1e6f));
    return mflops;
}

/**
 * Calculate the theoretical peak performance based on the given compute component.
 * @param comp the compute component containing the number of cores, number of FPUs, and clock speed in GHz
 * @return the calculated theoretical peak performance
 * @throws None
 */
double CalculateTheoreticalPeakPerformance(const ComputeComponent& comp){
    double peak_perf = comp.num_cores * comp.num_fpus * comp.clock_speed_ghz * 1000.0f;
    return peak_perf;
}

/**
 * Save results to a CSV file.
 * @param filename the name of the CSV file
 * @param peak_performance the peak performance value
 * @param achieved_performance the achieved performance value
 * @throws None
 */
void SaveResultsToCSV(const std::string& peak_performance, const std::string& achieved_performance){
    if(FILENAME.substr(FILENAME.size() - 4) != ".csv"){
        std::cout << "Error: Invalid file format. Please specify a CSV file." << std::endl;
        return;
    }

    std::ifstream infile(FILENAME, std::ios::in | std::ios::out | std::ios::binary);

    if(!infile){
        std::ofstream outfile(FILENAME);
        outfile.close();
    }

    std::ofstream outfile(FILENAME, std::ios::in | std::ios::out | std::ios::binary);
    std::string header_line = "matrix_size,peak_performance,achieved_performance";

    outfile.seekp(0, std::ios::end);

    if(outfile.tellp() == 0){
        outfile << '\xEF' << '\xBB' << '\xBF';
        outfile << header_line << std::endl;
    }

    outfile.seekp(0, std::ios::end);
    outfile << N << "," << peak_performance << "," << achieved_performance << std::endl;
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

    double mflops = CalculateMFLOPS(total_time);

    ComputeComponent cc = {"Chris", 4, 3.0f, 1};
    ComputeComponent co = {"Onur", 8, 4.2f, 1};

    const auto& peak_perf = CalculateTheoreticalPeakPerformance(co);

    printf("Theoretical Performance [%s] is: %f Mflop/s\n", co.name.c_str(), peak_perf); 
    printf("Total Mflop/s with %ld repeats: %f Mflop/s\n", REPEAT, mflops);
    printf("Writing results to %s\n", FILENAME.c_str());
    SaveResultsToCSV(std::to_string(peak_perf * 1e-3), std::to_string(mflops * 1e-3));
    printf("Finished writing results to %s\nExiting.", FILENAME.c_str());
}