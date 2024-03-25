#include <iostream>
#include <vector>
#include <stdexcept>
#include <random>
#include <fstream>
#include <algorithm>
#include <omp.h>
#include <mpi.h>


extern "C" {
#include "get_walltime.c"
}

const size_t REPEAT = 20;
const double lower_bound = 0.0f;
const double upper_bound = 10.0f;
std::default_random_engine re;
std::uniform_real_distribution<double> unif(lower_bound,upper_bound);


/**
 * Performs matrix multiplication of two matrices of type T.
 * @param mat1 the first matrix
 * @param mat1_dims the dimensions of the first matrix
 * @param mat2 the second matrix
 * @param mat2_dims the dimensions of the second matrix
 * @param result_vec the vector to store the result of the matrix multiplication
 * @return void
 * @throws None
 */

void MatMul(    const std::vector<double> mat1,
                const std::pair<int, int> mat1_dims,
                const std::vector<double> mat2,
                const std::pair<int, int> mat2_dims,
                std::vector<double>& result_vec) {    
    
    
    if (mat1.empty() || mat2.empty()) {
        throw std::runtime_error("Incoming matrix should not be empty!");
        exit(1);
    }

    // Check if two matrices are multipliable
    if (mat1_dims.second != mat2_dims.first) {
        throw std::runtime_error("Incoming matrices are not multipliable!");
        exit(1);
    }

    #pragma omp parallel for collapse(2) shared(mat1, mat2, result_vec) schedule(dynamic)
    for (size_t i = 0; i < mat1_dims.first; i++) {
        for(size_t j=0; j < mat1_dims.second; j++){
            for(size_t k=0; k < mat1_dims.second; k++){

                result_vec[i * mat2_dims.second + j] += mat1[i * mat1_dims.second + k] * mat2[k * mat2_dims.second + j];
                
            }
        }
    }
}


// /**
//  * Save results to a CSV file.
//  * @param filename the name of the CSV file
//  * @param peak_performance the peak performance value
//  * @param achieved_performance the achieved performance value
//  * @throws None
//  */
void SaveResultsToCSV(const int matrix_size, const int num_ranks, const int num_threads_per_rank, const double avg_time, std::string FILENAME = "data.csv"){
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
    std::string header_line = "MatrixSize,NumRanks,NumThreadsPerRank,AvgTime";

    outfile.seekp(0, std::ios::end);

    if(outfile.tellp() == 0){
        outfile << '\xEF' << '\xBB' << '\xBF';
        outfile << header_line << std::endl;
    }

    outfile.seekp(0, std::ios::end);
    outfile << matrix_size << "," << num_ranks << "," << num_threads_per_rank << "," << avg_time << std::endl;
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


void PrintMatrix(std::vector<double> matrix, int rows, int cols){
    printf("###### Matrix (%d x %d) #######\n", rows, cols);
    for (size_t i=0; i<rows; i++){
        printf("[");
        for (size_t j=0; j<cols; j++)
            printf("%f ", matrix[i * cols + j]);
        printf("]\n");
    }
}

bool TEST_MatMul(std::vector<double> mat1, std::vector<double> mat2, int MATRIX_SIZE, std::vector<double> parallel_result_vec){

    double eps = 1.0e-4;
    std::vector<double> result_vec(MATRIX_SIZE * MATRIX_SIZE, 0.0f);

    for (size_t i = 0; i < MATRIX_SIZE; i++) {
        for(size_t k=0; k < MATRIX_SIZE; k++){
            for(size_t j=0; j < MATRIX_SIZE; j++){
                result_vec[i * MATRIX_SIZE + j] += mat1[i * MATRIX_SIZE + k] * mat2[k * MATRIX_SIZE + j];            
            }
        }
    }

    // Check if the result_vec and parallel_result_vec are equal
    for (size_t i = 0; i < MATRIX_SIZE; i++) {
        for(size_t j=0; j < MATRIX_SIZE; j++){
            double cur_ground_truth = result_vec[i * MATRIX_SIZE + j];
            double cur_received = parallel_result_vec[i * MATRIX_SIZE + j];
            if (std::abs(cur_ground_truth - cur_received) > eps) {
                printf("Test failed: Result mismatch at index (%ld, %ld). Expected: %f, Received: %f\n", i, j, cur_ground_truth, cur_received);
                return false;
            }
        }
    }

    printf("Test passed: Result match\n");
    return true;
    
}


int main(int argc, char* argv[]) {

    int CUR_RANK, WORLD_SIZE;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &CUR_RANK);
    MPI_Comm_size(MPI_COMM_WORLD, &WORLD_SIZE);


    int MATRIX_SIZE = 100;
    if (argc > 1) {
        if (!std::isdigit(*argv[1]) || std::stoi(argv[1]) < 1 || std::stoi(argv[1]) > 10000) {
            throw std::runtime_error("Invalid matrix size argument. Please enter a numeric value between 1 and 10000.");
        }
        MATRIX_SIZE = std::stoi(argv[1]);

        // Ensure that the matrix size is a multiple of the number of MPI ranks
        if(MATRIX_SIZE % WORLD_SIZE != 0){
            throw std::runtime_error("Matrix size should be a multiple of the number of MPI ranks.");
        }
    } else {
        std::cout << "No matrix size argument provided. Defaulting to N=100." << std::endl;
        MATRIX_SIZE = 100;
    }

    int NUM_THREADS_PER_RANK = 1;

    if(argc > 2){
        if (!std::isdigit(*argv[2]) || std::stoi(argv[2]) < 1 || std::stoi(argv[2]) > 100) {
            throw std::runtime_error("Invalid number of threads argument. Please enter a numeric value between 1 and 100.");
        }
        NUM_THREADS_PER_RANK = std::stoi(argv[2]);
    } else {
        std::cout << "No number of threads argument provided. Defaulting to 1 thread." << std::endl;
        NUM_THREADS_PER_RANK = 1;
    }

    std::string FILENAME = "data.csv";
    if(argc > 3){
        printf("Filename: %s\n", argv[3]);
        FILENAME = std::string(argv[3]);
    }
    

    bool TEST_IT = false;
    if(argc > 4){
        if (std::string(argv[3]) == "test") {
            TEST_IT = true;
        }
    }


    // Set the number of threads to be used by each MPI rank
    omp_set_num_threads(NUM_THREADS_PER_RANK);

    // Synchronize all MPI ranks
    MPI_Barrier(MPI_COMM_WORLD);

    int num_rows_per_rank = MATRIX_SIZE / WORLD_SIZE;
    if (CUR_RANK == 0) {
        printf("| Configuration: |\n");
        printf("\t|? Matrix size: %d\n", MATRIX_SIZE);
        printf("\t|? Number of MPI ranks: %d\n", WORLD_SIZE);
        printf("\t|? Number of rows per rank: %d\n", num_rows_per_rank);
        printf("\t|? Number of threads per rank: %d\n", NUM_THREADS_PER_RANK);
        printf("\t|? Test: %s\n", TEST_IT ? "Yes" : "No");
    }

    std::vector<double> mat1(MATRIX_SIZE * MATRIX_SIZE);
    std::vector<double> mat2(MATRIX_SIZE * MATRIX_SIZE);
    
    if(CUR_RANK == 0){

        printf("[INFO] Generating random matrices MAT1 & MAT2\n");
        for(size_t j=0; j < mat1.size(); j++){                
            mat1[j] = GenerateRandomNumber();
            mat2[j] = GenerateRandomNumber();
        }
    }


    MPI_Barrier(MPI_COMM_WORLD);

    // Broadcast the mat2 to all MPI ranks
    MPI_Bcast(&mat2[0], MATRIX_SIZE * MATRIX_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter the matrices to all MPI ranks
    std::vector<double> mat1_local(num_rows_per_rank * MATRIX_SIZE);
    MPI_Scatter(&mat1[0], num_rows_per_rank * MATRIX_SIZE, MPI_DOUBLE, &mat1_local[0], num_rows_per_rank * MATRIX_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<double> timings_vec(REPEAT, 0.0f);

    for (size_t i = 0; i < REPEAT; i++) {
        // Vector to hold the results of matmul, num_rows_per_rank * MATRIX_SIZE
        std::vector<double> result_vec(num_rows_per_rank *  MATRIX_SIZE, 0.0f);

        // Perform matrix multiplication
        double start_time;
        get_walltime(&start_time);

        MatMul(mat1_local, std::make_pair(num_rows_per_rank, MATRIX_SIZE), mat2, std::make_pair(MATRIX_SIZE, MATRIX_SIZE), result_vec);

        double end_time;
        get_walltime(&end_time);
        auto diff_time = end_time - start_time;
        
        timings_vec[i] = diff_time;
    }

    double avg_time = 0.0f;
    for (size_t i = 0; i < REPEAT; i++) {
        avg_time += timings_vec[i];
    }
    avg_time /= REPEAT;

    if (CUR_RANK == 0) {
        printf("\t|! Average time: %f\n", avg_time);
        SaveResultsToCSV(MATRIX_SIZE, WORLD_SIZE, NUM_THREADS_PER_RANK, avg_time, FILENAME);
    }



    if (TEST_IT){

        // Vector to hold the results of matmul, num_rows_per_rank * MATRIX_SIZE
        std::vector<double> result_vec(num_rows_per_rank *  MATRIX_SIZE, 0.0f);
        MatMul(mat1_local, std::make_pair(num_rows_per_rank, MATRIX_SIZE), mat2, std::make_pair(MATRIX_SIZE, MATRIX_SIZE), result_vec);


        // Gather the results from all MPI ranks
        std::vector<double> gathered_results(MATRIX_SIZE * MATRIX_SIZE);
        MPI_Gather(&result_vec[0], num_rows_per_rank * MATRIX_SIZE, MPI_DOUBLE, &gathered_results[0], num_rows_per_rank * MATRIX_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        if(CUR_RANK == 0){
            TEST_MatMul(mat1, mat2, MATRIX_SIZE, gathered_results);
        }
    }

    MPI_Finalize();
}