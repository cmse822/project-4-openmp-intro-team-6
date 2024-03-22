OUTPUT_REPORT_FILE="part3_matrix_mult.csv"
PART3_EXECUTABLE="./part3_matrix_mult"

rm -f $OUTPUT_REPORT_FILE

# Run the part3_matrix_mult executable
MATRIX_SIZES=( 10 100 200 300 400 500 600 700 800 900 1000 )
NUM_MPI_RANKS=( 1 2 5 10 20)
NUM_THREADS_PER_RANK=( 1 2 5 10 20)

for matrix_size in "${MATRIX_SIZES[@]}"; do
    for num_mpi_ranks in "${NUM_MPI_RANKS[@]}"; do
        for num_threads_per_rank in "${NUM_THREADS_PER_RANK[@]}"; do
            echo "Running part3_matrix_mult with matrix size: $matrix_size, num_mpi_ranks: $num_mpi_ranks, num_threads_per_rank: $num_threads_per_rank"
            mpiexec -n $num_mpi_ranks $PART3_EXECUTABLE $matrix_size $num_threads_per_rank $OUTPUT_REPORT_FILE false
        done
    done
done
