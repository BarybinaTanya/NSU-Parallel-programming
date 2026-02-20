#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#define RANGE_MODULE 100
#define N 1000

void FillVectors(int* vector_a, int size_a, int local_start_vector_a, int* vector_b, int size_b) {
    for (int i = 0; i < size_a; i++) {
        vector_a[i] = (local_start_vector_a + i) % RANGE_MODULE;
    }
    for (int i = 0; i < size_b; i++) {
        vector_b[i] = (size_b - i) % RANGE_MODULE;
    }
}

int Minimum(int num_1, int num_2) {
    if (num_1 < num_2) { return num_1; }
    else { return num_2; }
}

int SequentialProgram(int argc, char **argv) {
    printf("I'm sequential program :) \n");

    MPI_Init(&argc, &argv);

    unsigned long long sum = 0;
    int *vector_a = (int*) malloc(N * sizeof(int));
    int *vector_b = (int*) malloc(N * sizeof(int));
    FillVectors(vector_a, N, 0, vector_b, N);

    double time_start = MPI_Wtime();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            sum += vector_a[i] * vector_b[j];
        }
    }

    double time_end = MPI_Wtime();

    printf("sum = %lld \ntime: %f \n", sum, time_end - time_start);

    free(vector_a);
    free(vector_b);
    MPI_Finalize();
    return 0;
}

#define ROOT_NUMBER 0

int ParallelPointToPoint (int argc, char **argv) {
    printf("I'm parallel point to point program ;) \n");


    return 0;
}

int ParallelGroupCommunications (int argc, char **argv) {

    int process_rank, count_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &count_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    printf("I'm parallel group program :D process %d \n", process_rank);

    const int local_N = (N / count_processes) + ( process_rank < (N % count_processes) );
    const int local_start = (N / count_processes) * process_rank
                            + Minimum(process_rank, N % count_processes);

    unsigned long long local_sum = 0;
    int *local_vector_a = (int*) malloc (local_N * sizeof(int));
    int *vector_b = (int*) malloc (N * sizeof(int));
    FillVectors(local_vector_a, local_N,local_start, vector_b, N);
    // now every process has its own initialized part of vector_a.

    double time_start = MPI_Wtime();

    for (int i = 0; i < local_N; i++) {
        for (int j = 0; j < N; j++) {
            local_sum += local_vector_a[i] * vector_b[j];
        }
    }

    double time_end = MPI_Wtime();

    unsigned long long general_sum;
    MPI_Reduce(&local_sum, &general_sum, 1, MPI_UNSIGNED_LONG_LONG,
               MPI_SUM, ROOT_NUMBER, MPI_COMM_WORLD);

    if (process_rank == 0) {
        printf("general_sum = %lld \ntime: %f \n", general_sum, time_end - time_start);
    }

    free(local_vector_a);
    free(vector_b);
    MPI_Finalize();
    return 0;
}

int main(int argc, char **argv) {
    int ret;
    if (argc > 1) {
        if (strcmp(argv[1], "-pg") == 0) {
            ret = ParallelGroupCommunications(argc, argv);
        } else if (strcmp(argv[1], "-pp") == 0) {
            ret = ParallelPointToPoint(argc, argv);
        } else if (strcmp(argv[1], "-s") == 0) {
            ret = SequentialProgram(argc, argv);
        } else {
            printf("sequential program is starting \nTo run sequential one add flag -s "
                   "\nTo run parallel point to point program add -pp \nTo run parallel group "
                   "communication program type -pg\n");
            ret = SequentialProgram(argc, argv);
        }
    } else {
        ret = SequentialProgram(argc, argv);
        printf("To run sequential one add flag -s\n"
               "To run parallel point to point program add -pp\n"
               "To run parallel group communication program type -pg\n");
    }
    return ret;
}
