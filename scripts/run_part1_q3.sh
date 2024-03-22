OUTPUT_REPORT_FILE="data/data_omp_thread_to_thread_speedup.csv"
PART1_Q3_EXECUTABLE="./matrix_mult_omp"

rm -f $OUTPUT_REPORT_FILE

# Run the matrix_mult_omp executable
MATRIX_SIZES=( 20 100 1000 )
NUM_THREADS_PER_RANK=( 1 2 4 8 16 32 64 )

for matrix_size in "${MATRIX_SIZES[@]}"; do
    for num_threads_per_rank in "${NUM_THREADS_PER_RANK[@]}"; do
        export OMP_NUM_THREADS=$num_threads_per_rank
        echo "Running matrix_mult_omp with matrix size: $matrix_size, num_threads_per_rank: $num_threads_per_rank"
        $PART1_Q3_EXECUTABLE $matrix_size $num_threads_per_rank $OUTPUT_REPORT_FILE
    done
done