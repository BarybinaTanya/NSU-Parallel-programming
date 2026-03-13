#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define RANGE_MODULE 10
#define ROOT_PROCESS_NUMBER 0
#define TRUE 0
#define NUMBER_SYSTEM_BASE_10 10
#define FILL_WITH_ZEROS 0
#define FILL_RANDOM 1
#define STOP_CRITERIA_INIT_VALUE 100
#define EPSILON_DEFAULT_VALUE 0.00001
#define GLOBAL_N_DEFAULT_VALUE 10000
double epsilon;
int global_N;
int global_matrix_row;
int global_matrix_column;

void FillVector(double** vector_x, short fill_zero_flag) {
    if (fill_zero_flag == FILL_WITH_ZEROS) {
        for (int i = 0; i < global_N; i++) {
            *vector_x[i] = 0;
        }
    } else {
        for (int i = 0; i < global_N; i++) {
            *vector_x[i] = (global_N - i) % RANGE_MODULE;
        }
    }

}

void AllocateVector(double** vector) {
    *vector = (double*)malloc(global_N * sizeof(double));
}

void FillMatrix(double*** full_matrix_A) {
    for (global_matrix_row = 0; global_matrix_row < global_N; global_matrix_row++) {
        for (global_matrix_column = 0; global_matrix_column < global_N; ++global_matrix_column) {

            if (global_matrix_row == global_matrix_column) {
                *full_matrix_A[global_matrix_row][global_matrix_column] =
                        global_N + (global_matrix_column % RANGE_MODULE);
            } else {
                *full_matrix_A[global_matrix_row][global_matrix_column] =
                        global_matrix_column % RANGE_MODULE;
            }
        }
    }
}

void FreeMatrix(double*** full_matrix_A) {
    for (global_matrix_row = 0; global_matrix_row < global_N; global_matrix_row++) {
        free(*full_matrix_A[global_matrix_row]);
    }
    free(*full_matrix_A);
}

void AllocateMatrix(double*** full_matrix_A) {
    *full_matrix_A = (double** )malloc(global_N * sizeof(double*));
    for (int i = 0; i < global_N; ++i) {
        *full_matrix_A[i] = (double*)malloc(global_N * sizeof(double));
    }
}

int Minimum(int a, int b) {
    return (a < b) ? a : b;
}

double* SubtractVectorFromVector (const double* minuend_vector, const double* subtrahend_vector) {
    double* res_vector;
    AllocateVector(&res_vector);
    for (int index = 0; index < global_N; ++index) {
        res_vector[index] = minuend_vector[index] - subtrahend_vector[index];
    }
    return res_vector;
}

double* MultiplyLocalMatrixToVector(int local_matrix_rows_count, double** local_matrix_A, const double* vector) {
    double *res_vector;
    AllocateVector(&res_vector);
    FillVector(&res_vector, FILL_WITH_ZEROS);

    for (int local_matrix_row_iter = 0; local_matrix_row_iter <
    local_matrix_rows_count; ++local_matrix_row_iter) {

        for (int vector_row_iter = 0; vector_row_iter < global_N; ++vector_row_iter) {
            res_vector[local_matrix_row_iter] +=
                    local_matrix_A[local_matrix_row_iter][vector_row_iter] * vector[vector_row_iter];
        }
    }
    return res_vector;
}

double ScalarProduct(const double* vector_1, const double* vector_2) {
    double product = 0;
    for (int iter = 0; iter < global_N; ++iter) {
        product += vector_1[iter] * vector_2[iter];
    }
    return product;
}

double* MultiplyVectorToScalar(const double* vector, double scalar) {
    double* vector_scalar;
    AllocateVector(&vector_scalar);

    for (int iter = 0; iter < global_N; ++iter) {
        vector_scalar[iter] = vector[iter] * scalar;
    }
    return vector_scalar;
}
//======================================================================================================================
//=============================================Sequential=program=======================================================
//======================================================================================================================
int SequentialProgram() {
    printf("Sequential program started\n");

    double** matrixA;
    AllocateMatrix( &matrixA);
    FillMatrix(&matrixA);

    double* vectorXn;
    AllocateVector(&vectorXn);
    FillVector(&vectorXn, FILL_WITH_ZEROS);

    double* vectorB;
    AllocateVector(&vectorB);
    FillVector(&vectorB, FILL_RANDOM);

    double* vectorAXn;
    AllocateVector(&vectorAXn);

    double* vectorYn;
    AllocateVector(&vectorYn);

    double* vectorAYn;
    AllocateVector(&vectorAYn);

    double tau;

    double* vectorXnPlus1;
    AllocateVector(&vectorXnPlus1);
    double stop_criteria_value = STOP_CRITERIA_INIT_VALUE;

    double start = MPI_Wtime();
    while (stop_criteria_value >= epsilon) {
        //----------------------------------------vectorXnPlus1-calculation-------------------------------------------------
        vectorAXn = MultiplyLocalMatrixToVector(global_N, matrixA, vectorXn);
        vectorYn = SubtractVectorFromVector(vectorAXn, vectorB);
        vectorAYn = MultiplyLocalMatrixToVector(global_N, matrixA, vectorYn);
        tau = (ScalarProduct(vectorYn, vectorAYn)) /
              (ScalarProduct(vectorAYn, vectorAYn));
        vectorXnPlus1 = SubtractVectorFromVector(vectorXn,
                                                 MultiplyVectorToScalar(vectorYn, tau));
        //---------------------------------------stop-flag-counting-and-checking--------------------------------------------
        double* vectorAxnMinusVectorB;
        AllocateVector(&vectorAxnMinusVectorB);
        vectorAxnMinusVectorB = SubtractVectorFromVector(vectorAXn, vectorB);
        stop_criteria_value = ScalarProduct(vectorAxnMinusVectorB, vectorAxnMinusVectorB) /
                              ScalarProduct(vectorB, vectorB);
    }
    double end = MPI_Wtime();
    //------------------------------------------------------------------------------------------------------------------
    printf("Time = %f seconds\n", end - start);

    free(vectorXn);
    free(vectorB);
    FreeMatrix(&matrixA);
    return 0;
}
//======================================================================================================================
// =====================================Parallel=program===========================================
//======================================================================================================================
int ParallelProgram() {
    printf("Parallel program started\n");
//    int rank, num_processes;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
//
//    int local_N = global_N / num_processes + (rank < global_N % num_processes);
//    double *local_a = (double*)malloc(local_N * sizeof(double));
//    double *b = (double*)malloc(global_N * sizeof(double));
//
//    //---------------------Sending-vectors-a-and-b-from-root-to-others-using-group-communications-----------------------
//
//    if (rank == ROOT_PROCESS_NUMBER) {
//
//        double *full_vector_a = (double*)malloc(global_N * sizeof(double));
//        double *full_vector_b = (double*)malloc(global_N * sizeof(double));
//        FillVector(&full_vector_a);
//
//        // Preparing technical arrays for Scatter_v to cut a vector in a custom way.....................................
//
//        int *processes_local_Ns_array = (int*)malloc(num_processes * sizeof(int));
//        int *processes_local_starts_array = (int*)malloc(num_processes * sizeof(int));
//        int offset = 0;
//        for (int i = 0; i < num_processes; i++) {
//            processes_local_Ns_array[i] = global_N / num_processes + (i < global_N % num_processes);
//            processes_local_starts_array[i] = offset;
//            offset += processes_local_Ns_array[i];
//        }
//
//        // Cut full_vector_a our custom way and send it to everyone else at MPI_COMM_WORLD..............................
//
//        MPI_Scatterv(full_vector_a, processes_local_Ns_array,
//                     processes_local_starts_array, MPI_INT,
//                     local_a, local_N, MPI_INT,
//                     ROOT_PROCESS_NUMBER, MPI_COMM_WORLD);
//
//        // B_cast sends the buffer b to other processes, that will also put their received data
//        // to the buffer b (in their memory, but buffer pointer is the same). Of course, buffer b must be
//        // initialized at the root process.
//        for (int i = 0; i < global_N; i++) b[i] = full_vector_b[i];
//        MPI_Bcast(b, global_N, MPI_INT, ROOT_PROCESS_NUMBER, MPI_COMM_WORLD);
//
//        free(full_vector_a);
//        free(full_vector_b);
//        free(processes_local_Ns_array);
//        free(processes_local_starts_array);
//    }
//
//    else {
//        //---------------------------Processes-receive-their-part-of-vector-a-and-full-vector-b-----------------------------
//
//        // Receiving Scatter_v and B_cast. The last ones arguments are absolutely the same as in the sending process.
//        // Receiving Scatter_v buffer and cutting information are set to NULL.
//        MPI_Scatterv(NULL, NULL, NULL, MPI_INT,
//                     local_a, local_N, MPI_INT,
//                     ROOT_PROCESS_NUMBER, MPI_COMM_WORLD);
//        MPI_Bcast(b, global_N, MPI_INT, ROOT_PROCESS_NUMBER, MPI_COMM_WORLD);
//    }
//
//    //------------------------------------------------Calculations------------------------------------------------------
//
//    double start = MPI_Wtime();
//    unsigned long local_sum = 0;
//    for (int i = 0; i < local_N; i++) {
//        for (int j = 0; j < global_N; j++) {
//            local_sum += (unsigned long)local_a[i] * b[j];
//        }
//    }
//    double end = MPI_Wtime();
//
//    //--------------------------------------Collecting-results-using-MPI-Reduce-----------------------------------------
//    unsigned long total_sum = 0;
//    MPI_Reduce(&local_sum, &total_sum, 1, MPI_UNSIGNED_LONG,
//               MPI_SUM, ROOT_PROCESS_NUMBER, MPI_COMM_WORLD);
//
//    if (rank == ROOT_PROCESS_NUMBER) {
//        printf("Sum = %lu\n", total_sum);
//        printf("Time = %f seconds\n", end - start);
//        printf("Number of processes = %d\n", num_processes);
//    }
//
//    free(local_a);
//    free(b);
    return 0;
}
//======================================================================================================================
//===========================================Processing=main=function===================================================
//======================================================================================================================
int main(int argc, char *argv[]) {

    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc > 3) {
        if (rank == ROOT_PROCESS_NUMBER) {
            char *argv_N_values_digits_end_pointer;
            double val = strtod(argv[3], &argv_N_values_digits_end_pointer);

            if (argv_N_values_digits_end_pointer == argv[3] || // check if first symbol is not digit-symbol at all
                *argv_N_values_digits_end_pointer != '\0') {
                epsilon = EPSILON_DEFAULT_VALUE;
            } else {
                epsilon = val;
            }
        }
        MPI_Bcast(&epsilon, 1, MPI_DOUBLE, ROOT_PROCESS_NUMBER, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(&epsilon, 1, MPI_DOUBLE, ROOT_PROCESS_NUMBER, MPI_COMM_WORLD);
    }

    //---------------------------------Read-global_N-from-command-prompt-arguments---------------------------------------------

    int default_N = GLOBAL_N_DEFAULT_VALUE;
    if (argc > 2) {
        if (rank == ROOT_PROCESS_NUMBER) {
            char *argv_N_values_digits_end_pointer;
            long val = strtol(argv[2], &argv_N_values_digits_end_pointer,
                              NUMBER_SYSTEM_BASE_10);

            if (argv_N_values_digits_end_pointer == argv[2] || // check if first symbol is not digit-symbol at all
                *argv_N_values_digits_end_pointer != '\0' || val <= 0) {
                printf("Invalid global_N value, using default %d\n", default_N);
                global_N = default_N;
            } else {
                global_N = (int)val;
            }
        }
        MPI_Bcast(&global_N, 1, MPI_INT, ROOT_PROCESS_NUMBER, MPI_COMM_WORLD);
    } else {
        global_N = default_N;
        MPI_Bcast(&global_N, 1, MPI_INT, ROOT_PROCESS_NUMBER, MPI_COMM_WORLD);
    }

    //----------------------------------------Read-the-program-type-----------------------------------------------------

    int ret = 0;
    if (argc > 1) {
        if (strcmp(argv[1], "-s") == TRUE) {
            if (rank == ROOT_PROCESS_NUMBER) ret = SequentialProgram();
        } else if (strcmp(argv[1], "-pp") == TRUE) {
            ret = ParallelProgram();
        } else {
            if (rank == ROOT_PROCESS_NUMBER) {
                printf("Unknown flag. Usage:\n");
                printf("  %s [global_N] -s           (sequential)\n", argv[0]);
                printf("  %s [global_N] -pp          (point-to-point parallel)\n", argv[0]);
                printf("      global_N - vector size (positive integer, default %d)\n", default_N);
            }
        }
    } else {
        if (rank == ROOT_PROCESS_NUMBER) {
            printf("No flags specified. Running sequential by default.\n"
                   "global_N is equal to %d\n", default_N);
            ret = ParallelProgram();
        }
    }
    MPI_Finalize();
    return ret;
}