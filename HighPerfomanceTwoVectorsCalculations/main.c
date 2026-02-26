#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define RANGE_MODULE 100
#define ROOT_PROCESS_NUMBER 0
int N;

void FillFullVectors(int *full_a, int *full_b) {
    for (int i = 0; i < N; i++) {
        full_a[i] = i % RANGE_MODULE;
        full_b[i] = (N - i) % RANGE_MODULE;
    }
}

int Minimum(int a, int b) {
    return (a < b) ? a : b;
}
//======================================================================================================================
//=============================================Sequential=program=======================================================
//======================================================================================================================
int SequentialProgram(void) {
    printf("Sequential program started\n");

    unsigned long sum = 0;
    int *a = (int*)malloc(N * sizeof(int));
    int *b = (int*)malloc(N * sizeof(int));
    FillFullVectors(a, b);

    double start = MPI_Wtime();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            sum += a[i] * b[j];
        }
    }
    double end = MPI_Wtime();

    printf("Sum = %lu\n", sum);
    printf("Time = %f seconds\n", end - start);

    free(a);
    free(b);
    return 0;
}
//======================================================================================================================
//=========================================Parallel=point=to=point=program==============================================
//======================================================================================================================
int ParallelPointToPoint(void) {

    int rank, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    int local_N = N / num_processes + (rank < N % num_processes);
    int local_start = (N / num_processes) * rank + Minimum(rank, N % num_processes);

    unsigned long local_sum = 0;
    int *local_a = (int*)malloc(local_N * sizeof(int));
    int *b = (int*)malloc(N * sizeof(int));

    //--------------------------------Sending-each-part-of-the-vector-a-to-its-process----------------------------------

    if (rank == ROOT_PROCESS_NUMBER) {
        int *full_vector_a = (int*)malloc(N * sizeof(int));
        int *full_vector_b = (int*)malloc(N * sizeof(int));
        FillFullVectors(full_vector_a, full_vector_b);

        // Copying root process's parts to its local buffers............................................................

        for (int i = 0; i < local_N; i++) local_a[i] = full_vector_a[local_start + i];
        for (int i = 0; i < N; i++) b[i] = full_vector_b[i];

        // Sending each part of vector_a to a corresponding process.....................................................

        int offset = local_N;
        for (int dest = 1; dest < num_processes; dest++) {
            int dest_local_N = N / num_processes + (dest < N % num_processes);
            MPI_Send(full_vector_a + offset, dest_local_N,
                     MPI_INT, dest, 0, MPI_COMM_WORLD);
            offset += dest_local_N;
        }

        // Sending full vector_b........................................................................................

        for (int dest = 1; dest < num_processes; dest++) {
            MPI_Send(full_vector_b, N, MPI_INT, dest, 1, MPI_COMM_WORLD);
        }

        free(full_vector_a);
        free(full_vector_b);
    }

    else {
        //--------------------------Receiving-process's-part-of-vector-a-and-vector-b-----------------------------------

        MPI_Recv(local_a, local_N, MPI_INT, ROOT_PROCESS_NUMBER,
                 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(b, N, MPI_INT, ROOT_PROCESS_NUMBER, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    //------------------------------------------------Calculations------------------------------------------------------

    double start = MPI_Wtime();
    for (int i = 0; i < local_N; i++) {
        for (int j = 0; j < N; j++) {
            local_sum += (unsigned long)local_a[i] * b[j];
        }
    }
    double end = MPI_Wtime();

    //-------------------Collecting-each-part-of-the-total-sum-from-all-processes-to-the-root-one-----------------------

    unsigned long total_sum = local_sum;
    if (rank == ROOT_PROCESS_NUMBER) {
        for (int source_process = 1; source_process < num_processes; source_process++) {
            unsigned long received_value;
            MPI_Recv(&received_value, 1, MPI_UNSIGNED_LONG, source_process,
                     source_process, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_sum += received_value;
        }
        printf("Sum = %lu\n", total_sum);
        printf("Time = %f seconds\n", end - start);
        printf("Number of processes = %d\n", num_processes);
    } else {
        MPI_Send(&local_sum, 1, MPI_UNSIGNED_LONG,
                 ROOT_PROCESS_NUMBER, rank, MPI_COMM_WORLD);
    }

    free(local_a);
    free(b);
    return 0;
}
//======================================================================================================================
// =====================================Parallel=group=communications=program===========================================
//======================================================================================================================
int ParallelGroupCommunications(void) {

    int rank, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    int local_N = N / num_processes + (rank < N % num_processes);
    int *local_a = (int*)malloc(local_N * sizeof(int));
    int *b = (int*)malloc(N * sizeof(int));

    //---------------------Sending-vectors-a-and-b-from-root-to-others-using-group-communications-----------------------

    if (rank == ROOT_PROCESS_NUMBER) {

        int *full_vector_a = (int*)malloc(N * sizeof(int));
        int *full_vector_b = (int*)malloc(N * sizeof(int));
        FillFullVectors(full_vector_a, full_vector_b);

        // Preparing technical arrays for Scatter_v to cut a vector in a custom way.....................................

        int *processes_local_Ns_array = (int*)malloc(num_processes * sizeof(int));
        int *processes_local_starts_array = (int*)malloc(num_processes * sizeof(int));
        int offset = 0;
        for (int i = 0; i < num_processes; i++) {
            processes_local_Ns_array[i] = N / num_processes + (i < N % num_processes);
            processes_local_starts_array[i] = offset;
            offset += processes_local_Ns_array[i];
        }

        // Cut full_vector_a our custom way and send it to everyone else at MPI_COMM_WORLD..............................

        MPI_Scatterv(full_vector_a, processes_local_Ns_array,
                     processes_local_starts_array, MPI_INT,
                     local_a, local_N, MPI_INT,
                     ROOT_PROCESS_NUMBER, MPI_COMM_WORLD);

        // B_cast sends the buffer b to other processes, that will also put their received data
        // to the buffer b (in their memory, but buffer pointer is the same). Of course, buffer b must be
        // initialized at the root process.
        for (int i = 0; i < N; i++) b[i] = full_vector_b[i];
        MPI_Bcast(b, N, MPI_INT, ROOT_PROCESS_NUMBER, MPI_COMM_WORLD);

        free(full_vector_a);
        free(full_vector_b);
        free(processes_local_Ns_array);
        free(processes_local_starts_array);
    }

    else {
    //---------------------------Processes-receive-their-part-of-vector-a-and-full-vector-b-----------------------------

        // Receiving Scatter_v and B_cast. The last ones arguments are absolutely the same as in the sending process.
        // Receiving Scatter_v buffer and cutting information are set to NULL.
        MPI_Scatterv(NULL, NULL, NULL, MPI_INT,
                     local_a, local_N, MPI_INT,
                     ROOT_PROCESS_NUMBER, MPI_COMM_WORLD);
        MPI_Bcast(b, N, MPI_INT, ROOT_PROCESS_NUMBER, MPI_COMM_WORLD);
    }

    //------------------------------------------------Calculations------------------------------------------------------

    double start = MPI_Wtime();
    unsigned long local_sum = 0;
    for (int i = 0; i < local_N; i++) {
        for (int j = 0; j < N; j++) {
            local_sum += (unsigned long)local_a[i] * b[j];
        }
    }
    double end = MPI_Wtime();

    //--------------------------------------Collecting-results-using-MPI-Reduce-----------------------------------------
    unsigned long total_sum = 0;
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_UNSIGNED_LONG,
               MPI_SUM, ROOT_PROCESS_NUMBER, MPI_COMM_WORLD);

    if (rank == ROOT_PROCESS_NUMBER) {
        printf("Sum = %lu\n", total_sum);
        printf("Time = %f seconds\n", end - start);
        printf("Number of processes = %d\n", num_processes);
    }

    free(local_a);
    free(b);
    return 0;
}
//======================================================================================================================
//===========================================Processing=main=function===================================================
//======================================================================================================================
int main(int argc, char *argv[]) {

    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //---------------------------------Read-N-from-command-prompt-arguments---------------------------------------------

    int default_N = 100000;
    if (argc > 2) {
        if (rank == 0) {
            char *end_pointer;
            long val = strtol(argv[2], &end_pointer, 10);
            if (end_pointer == argv[2] || *end_pointer != '\0' || val <= 0) {
                printf("Invalid N value, using default %d\n", default_N);
                N = default_N;
            } else {
                N = (int)val;
            }
        }
        MPI_Bcast(&N, 1, MPI_INT, ROOT_PROCESS_NUMBER, MPI_COMM_WORLD);
    } else {
        N = default_N;
        MPI_Bcast(&N, 1, MPI_INT, ROOT_PROCESS_NUMBER, MPI_COMM_WORLD);
    }

    //----------------------------------------Read-the-program-type-----------------------------------------------------

    int ret = 0;
    if (argc > 1) {
        if (strcmp(argv[1], "-s") == 0) {
            if (rank == 0) ret = SequentialProgram();
        } else if (strcmp(argv[1], "-pp") == 0) {
            ret = ParallelPointToPoint();
        } else if (strcmp(argv[1], "-pg") == 0) {
            ret = ParallelGroupCommunications();
        } else {
            if (rank == 0) {
                printf("Unknown flag. Usage:\n");
                printf("  %s -s              (sequential)\n", argv[0]);
                printf("  %s -pp [N]         (point-to-point parallel)\n", argv[0]);
                printf("  %s -pg [N]         (group communications parallel)\n", argv[0]);
                printf("  N – vector size (positive integer, default %d)\n", default_N);
            }
        }
    } else {
        if (rank == 0) {
            printf("No flags specified. Running sequential by default.\n");
            ret = SequentialProgram();
        }
    }

    MPI_Finalize();
    return ret;
}