# Project 4: OpenMP and Hybrid Parallelism 

In this project, you will gain experience with the basics of shared-memory parallelism using OpenMP. 
You will also combine distributed- and shared-memory approaches by implementing hybrid parallelism with MPI and OpenMP. 
In addition to the course text book, you may wish to refer to the excellent OpenMP resource pages at <https://computing.llnl.gov/tutorials/openMP/>.

## Part 1: OpenMP Matrix-Matrix Multiplication

Consider the simple matrix-matrix multiplication,

```C
for i = 1, N
  for j = 1, N
    for k = 1, N
      C[i,j] += A[i,k] * B[k,j]
```

What strategies could you use to add parallelism using OpenMP threading to this kernel? Is each of the three loops threadable?

For this kernel, I would collapse the outer two loops. I would not do all three due to race issues with writing over the same shared C matrix. And adding atomic operations will only degrade performance further. So this collapse 2 apporach would be best. You essentially just collapse those two for loops into one large iteration so the forked threads can get their divided share of work.

It's threadable, but I would not recommend it for the reasons I stated above, which is race conditions of writing to the same variable. I think it would be wise to leave each thread to do it's own iterations for that inner loop. I guess it could be possible with using reduction on a sum variable to get partial sums for the inner for loop, then outside of the reduction block you can set the sum to the C matrix. That would also work, but could be more inefficient.

Now, let's implement so OpenMP loop parallelism.

1. Modify your MMM code from Project 1 to implement OpenMP threading by adding appropriate compiler directives to the outer loop of the MMM kernel. When compiling the OpenMP version of your code be sure to include the appropriate compiler flag (`-fopenmp` for GCC).

    Done.

2. Compute the time-to-solution of your MMM code for 1 thread (e.g., `export OMP_NUM_THREADS=1`) to the non-OpenMP version (i.e., compiled without the `-fopenmp` flag). Any matrix size `N` will do here. Does it perform as you expect? If not, consider the OpenMP directives you are using.

    Over matrix size N=1000, using 20 repeats, 1 thread, here are the results for the non-OpenMP version:
    | Matrix Size | Time to Solution |
    |------------:|-----------------:|
    |         100 |         0.007370 |
    |        1000 |         6.380824 |

    And here is the results for the OpenMP version:
    | Matrix Size | Time to Solution |
    |------------:|-----------------:|
    |         100 |         0.006763 |
    |        1000 |         6.415191 |

    As expected, the performance here is the same between Non-OpenMP and OpenMP due to the fact that we set the OMP_NUM_THREADS to 1. It won't take advantage of OpenMP's multi-threaded capabilities, meaning there's no parallel execution.

3. Perform a thread-to-thread speedup study of your MMM code either on your laptop or HPCC. Compute the total time to solution for a few thread counts (in powers of 2): `1,2,4,...T`, where T is the maximum number of threads available on the machine you are using. Do this for matrix sizes of `N=20,100,1000`.

    I used HPCC for the results and T=64. The results can be found in `data_omp_thread_to_thread_speedup.csv`

4. Plot the times-to-solution for the MMM for each value of `N` separately as functions of the the thread count `T`. Compare the scaling of the MMM for different matrix dimensions.

    ![plot](https://github.com/cmse822/project-4-openmp-intro-team-6/assets/94200328/bb9c07af-d03d-4114-a9f0-c1738664acbc)

5. Verify that for the same input matrices that the solution does not depend on the number of threads.

    This has been verified. It definitetly is deterministic which should be the case and indicates our implementation is correct. The same result is given, no matter if running on 1 thread or various different number of threads.

## Part 2: Adding OpenMP threading to a simple MPI application

Take a look at the Hello World applications that we have used in past assignments that include basic MPI functionality. Modify one of these applications to include OpenMP.

1. Wrap the print statements in an `omp parallel` region.

    **Done.**

2. Make sure to modify the `MPI_Init` call accordingly to allow for threads! What level of thread support do you need?

The thread support we chose is MPI_THREAD_FUNNELED because we only want the main thread making the MPI calls while the other threads are responsible for the parallel region of the code we defined.

3. Compile the code including the appropriate flag for OpenMP support. For a GCC-based MPI installation, this would be, e.g., `mpic++ -fopenmp hello.cpp`.

    **Done.**

4. Run the code using 2 MPI ranks and 4 OpenMP threads per rank. To do this, prior to executing the run command, set the number of threads environment variable as `> export OMP_NUM_THREADS=4`. Then you can simply execute the application with the `mpiexec` command: `> mpiexec -n 2 ./a.out`.

    **Done.**

5. Explain the output.

    For the output, this is what we got:

    ```bash
    Hello, World! This is thread id 2 of 4 threads on process rank 1
    Hello, World! This is thread id 0 of 4 threads on process rank 1
    Hello, World! This is thread id 1 of 4 threads on process rank 1
    Hello, World! This is thread id 3 of 4 threads on process rank 1
    Hello, World! This is thread id 1 of 4 threads on process rank 0
    Hello, World! This is thread id 2 of 4 threads on process rank 0
    Hello, World! This is thread id 0 of 4 threads on process rank 0
    Hello, World! This is thread id 3 of 4 threads on process rank 0
    ```

    We just print a hello world statement along with the thread id, number of threads, and process rank.

    Of course, the instructions told us to use 2 ranks, thats why we see only 0 and 1 ranks. And we set the number of threads to be 4, so 4 threads are created on each process rank.

    The number of threads and thread ID number are both private so that each thread has it's own copy of those variables and they are competing updating the same memory location.

    There is no gurantee of the thread order since they are all forked and what ever thread completes first is done. Thats why the thread ID's are out of order for both process ranks.

## Part 3: Hybrid Parallel Matrix Multiplication

Now, let's combine OpenMP and MPI functionality into a hybrid parallel version of the MMM.

1. Add MPI to  you OpenMP MMM code by distributing the rows of one of the input matrices across MPI ranks. Have each MPI rank perform its portion of the MMM using OpenMP threading. Think very carefully about the structure of the main MMM loops! Once done, gather the resulting matrix on rank 0 and output the result. Verify that for the same input matrices the result does not depend on either the number of MPI ranks or the number of OpenMP threads per rank.

   We have verified that the result does not depend on the number of MPI ranks or OpenMP threads per rank. The resulting output is the same, regardless.

2. On HPCC, carry out a performance study in which you vary the number of MPI ranks, the number of OpenMP threads per rank, and the matrix size. Make plots showing the times to solution for the various cases. Explain your results.

    TODO.

## What to turn in

To your git project repo, commit your final working code for the above exercises and a concise write-up including all plots, and detailed responses to the questions posed concerning your results.

## How to run

### Setup

Run the following commands:

```bash
module purge
module load intel/2020a
export OMP_NUM_THREADS=<num_threads>
```

### For running the matrix_mult.cpp code

Run the following commands:

```bash
gcc -fopenmp -o matrix_mult_omp matrix_mult.cpp -lstdc++ -Wall -O3
./matrix_mult <matrix_size>
```

### For running the hello.cpp code

Run the following commands:

```bash
mpicxx -fopenmp -o hello hello.cpp
mpiexec -n 2 ./hello
```
