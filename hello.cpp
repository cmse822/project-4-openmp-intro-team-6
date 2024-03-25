#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>

using namespace std;

int main(int argc, char **argv) 
{
    int provided;
    int rank;
    int numtasks;
    int thread_id;
    int num_threads;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    #pragma omp parallel private(thread_id, num_threads)
    {
        thread_id = omp_get_thread_num();
        num_threads = omp_get_num_threads();

        printf("Hello, World! This is thread id %d of %d threads on process rank %d\n", thread_id, num_threads, rank);
    }
    MPI_Finalize();

    return 0;
}