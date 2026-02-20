#include <malloc.h>
#include <string.h>
#include <stdio.h>
#include <mpi.h>

#define RANGE_MODULE 100
#define N 100000
#define ROOT_NUMBER 0

void FillVectors(int* vector_a, int size_a, int local_start_vector_a, int* vector_b, int size_b) {
    for (int i = 0; i < size_a; i++) {
        vector_a[i] = (local_start_vector_a + i) % RANGE_MODULE;
    }
    for (int i = 0; i < size_b; i++) {
        vector_b[i] = (size_b - i) % RANGE_MODULE;
    }
}

int Minimum(int num_1, int num_2) {
    return (num_1 < num_2) ? num_1 : num_2;
}

int SequentialProgram() {
    printf("I'm sequential program :) \n");

    unsigned long sum = 0;
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

    printf("\ntime: %f \n", time_end - time_start);

    free(vector_a);
    free(vector_b);
    return 0;
}

int ParallelPointToPoint() {
    int process_rank, count_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &count_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    const int local_N = (N / count_processes) + (process_rank < (N % count_processes));
    const int local_start = (N / count_processes) * process_rank
                            + Minimum(process_rank, N % count_processes);

    unsigned long local_sum = 0;
    int *local_vector_a = (int*) malloc(local_N * sizeof(int));
    int *vector_b = (int*) malloc(N * sizeof(int));
    FillVectors(local_vector_a, local_N, local_start, vector_b, N);

    double time_start = MPI_Wtime();

    for (int i = 0; i < local_N; i++) {
        for (int j = 0; j < N; j++) {
            local_sum += local_vector_a[i] * vector_b[j];
        }
    }

    double time_end = MPI_Wtime();
    unsigned long general_sum = 0;

    if (process_rank != ROOT_NUMBER) {
        MPI_Send(&local_sum, 1, MPI_UNSIGNED_LONG,
                 ROOT_NUMBER, process_rank, MPI_COMM_WORLD);
    } else {
        for (int source = 1; source < count_processes; ++source) {
            unsigned long received_sum;
            MPI_Recv(&received_sum, 1, MPI_UNSIGNED_LONG,
                     source, source, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            general_sum += received_sum;
        }
        general_sum += local_sum;
        printf("\ntime = %f\nnumber of processes = %d\n",
               time_end - time_start, count_processes);
    }

    free(local_vector_a);
    free(vector_b);
    return 0;
}

int ParallelGroupCommunications() {
    int process_rank, count_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &count_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    const int local_N = (N / count_processes) + (process_rank < (N % count_processes));
    const int local_start = (N / count_processes) * process_rank
                            + Minimum(process_rank, N % count_processes);

    unsigned long local_sum = 0;
    int *local_vector_a = (int*) malloc(local_N * sizeof(int));
    int *vector_b = (int*) malloc(N * sizeof(int));
    FillVectors(local_vector_a, local_N, local_start, vector_b, N);

    double time_start = MPI_Wtime();

    for (int i = 0; i < local_N; i++) {
        for (int j = 0; j < N; j++) {
            local_sum += local_vector_a[i] * vector_b[j];
        }
    }

    double time_end = MPI_Wtime();

    // Now gather all local sums into one by point to point communications.
    // That means, receive function will be in a cycle.
    unsigned long general_sum = 0;
    MPI_Reduce(&local_sum, &general_sum, 1, MPI_UNSIGNED_LONG,
               MPI_SUM, ROOT_NUMBER, MPI_COMM_WORLD);

    if (process_rank == 0) {
        printf("time = %f \n number of processes = %d\n",
               time_end - time_start, count_processes);
    }

    free(local_vector_a);
    free(vector_b);
    return 0;
}

int main(int argc, char **argv) {
    int ret = 0;
    int rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc > 1) {
        if (strcmp(argv[1], "-pg") == 0) {
            ret = ParallelGroupCommunications();
        } else if (strcmp(argv[1], "-pp") == 0) {
            ret = ParallelPointToPoint();
        } else if (strcmp(argv[1], "-s") == 0) {
            if (rank == 0) {
                ret = SequentialProgram();
            }
        } else {
            if (rank == 0) {
                printf("Unknown flag. Usage:\n"
                       "  %s -s   (sequential)\n"
                       "  %s -pp  (point-to-point parallel)\n"
                       "  %s -pg  (group communications parallel)\n",
                       argv[0], argv[0], argv[0]);
            }
        }
    } else {
        if (rank == 0) {
            ret = SequentialProgram();
        }
    }

    MPI_Finalize();
    return ret;
}
