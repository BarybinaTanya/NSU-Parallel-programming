#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <mpi.h>
#include <limits.h>

#define RANGE_MODULE 100
#define ROOT_PROCESS_NUMBER 0
#define TRUE 0
#define NUMBER_SYSTEM_BASE_10 10
#define FILL_WITH_ZEROS 0
#define FILL_RANDOM 1
#define STOP_CRITERIA_INIT_VALUE 100
#define EPSILON_DEFAULT_VALUE 0.00001
#define GLOBAL_N_DEFAULT_VALUE 10000
#define ITERATIONS_COUNT_ALLOWED 1000

double epsilon;
int global_N;

void FillVector(double** vector_x, short fill_zero_flag, int vector_size) {
    if (fill_zero_flag == FILL_WITH_ZEROS) {
        for (int i = 0; i < vector_size; i++) {
            (*vector_x)[i] = 0;
        }
    } else {
        for (int i = 0; i < vector_size; i++) {
            (*vector_x)[i] = (global_N - i) % RANGE_MODULE;
        }
    }

}

void AllocateVector(double** vector, int vector_size) {
    *vector = (double*)malloc(vector_size * sizeof(double));
}

void AllocateMatrixContinuously(double** matrix, int rows_number) {
    *matrix = (double*)malloc(global_N * rows_number * sizeof(double));
}

void FillContinuousMatrix(double** matrix) {
    for (int i = 0; i < global_N; ++i) {
        for (int j = 0; j < global_N; ++j) {
            if (i == j) {
                (*matrix)[global_N * i + j] = (double)(global_N * (j % RANGE_MODULE));
            } else {
                (*matrix)[global_N * i + j] = (double)(j % RANGE_MODULE);
            }
        }
    }
}

void SubtractVectorFromVector (const double* minuend_vector, const double* subtrahend_vector,
                               double* res_vector, int vectors_size) {
    for (int i = 0; i < vectors_size; ++i) {
        res_vector[i] = minuend_vector[i] - subtrahend_vector[i];
    }
}

void MultiplyLocalMatrixToVector(int mtx_rows_count, const double* local_matrix_A,
                                 const double* vector, double* res_vector) {
    FillVector(&res_vector, FILL_WITH_ZEROS, mtx_rows_count);
    for (int i = 0; i < mtx_rows_count; ++i) {
        for (int j = 0; j < global_N; ++j) {
            res_vector[i] += local_matrix_A[i * global_N + j] * vector[j];
        }
    }
}

double ScalarProduct(const double* vector_1, const double* vector_2) {
    double product = 0;
    for (int i = 0; i < global_N; ++i) {
        product += vector_1[i] * vector_2[i];
    }
    return product;
}

void MultiplyVectorToScalar(double* vector, double scalar) {
    for (int i = 0; i < global_N; ++i) {
        vector[i] *= scalar;
    }
}

void CopyVectorsValueToOtherVector(const double* vector, double* other_vectors, int vectors_length) {
    for (int i = 0; i < vectors_length; ++i) {
        other_vectors[i] = vector[i];
    }
}
//======================================================================================================================
//=============================================Sequential=program=======================================================
//======================================================================================================================
int SequentialProgram() {
    printf("Sequential program started\n");

    double* matrix_A;
    int local_matrix_rows_count = global_N;
    AllocateMatrixContinuously(&matrix_A, local_matrix_rows_count);
    FillContinuousMatrix(&matrix_A);

    if (matrix_A == NULL) {
        perror("Failed to allocate matrix!\n");
        return 1;
    }

    double* vector_Xn;
    AllocateVector(&vector_Xn, local_matrix_rows_count);
    FillVector(&vector_Xn, FILL_WITH_ZEROS, local_matrix_rows_count);

    double* vector_B;
    AllocateVector(&vector_B, global_N);
    FillVector(&vector_B, FILL_RANDOM, global_N);
    vector_B[global_N - 1] += global_N * 2; // diagonal dominance of the augmented matrix
    // of a system of linear equations. Constant 2 here means nothing - it can be easily
    // replaced by any other integer

    double* vector_AXn;
    AllocateVector(&vector_AXn, local_matrix_rows_count);

    double* vector_Yn;
    AllocateVector(&vector_Yn, local_matrix_rows_count);

    double* vector_AYn;
    AllocateVector(&vector_AYn, local_matrix_rows_count);

    double tauN;

    double* vector_Xn_plus_1;
    AllocateVector(&vector_Xn_plus_1, local_matrix_rows_count);
    double stop_criteria_value = STOP_CRITERIA_INIT_VALUE;

    double start = MPI_Wtime();
    unsigned long long iterations_count = 0;

    //------------------------------------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------------------------------------
    while (stop_criteria_value >= epsilon && iterations_count < ITERATIONS_COUNT_ALLOWED) {
        //----------------------------------------vector_Xn_plus_1-calculation---------------------------------------------
        MultiplyLocalMatrixToVector(global_N, matrix_A,
                                    vector_Xn, vector_AXn); // matrix_A * vector_Xn = vector_AXn

        SubtractVectorFromVector(vector_AXn, vector_B,
                                 vector_Yn, local_matrix_rows_count);
        //vector_Yn = matrix_A * vector_Xn - vector_B

        MultiplyLocalMatrixToVector(local_matrix_rows_count, matrix_A,
                                    vector_Yn, vector_AYn); // matrix_A * vector_Yn = AYn

        tauN = (ScalarProduct(vector_Yn, vector_AYn)) /
               (ScalarProduct(vector_AYn, vector_AYn)); // tauN = (Yn, AYn) / (AYn, AYn)

        stop_criteria_value = ScalarProduct(vector_Yn, vector_Yn) /
                              ScalarProduct(vector_B, vector_B); // |AXn - vector_B|^2 / |vector_B|^2

        MultiplyVectorToScalar(vector_Yn, tauN); // counting vector_tau_Yn

        SubtractVectorFromVector(vector_Xn, vector_Yn, vector_Xn_plus_1,
                                 local_matrix_rows_count); // vector_Xn_plus_1 = vector_Xn - vector_tau_Yn
        //---------------------------------------stop-flag-counting-and-checking----------------------------------------
        iterations_count++;
        CopyVectorsValueToOtherVector(vector_Xn_plus_1, vector_Xn, local_matrix_rows_count);
    }
    if (iterations_count < ITERATIONS_COUNT_ALLOWED && stop_criteria_value < epsilon) {
        printf("SLE solved!\n");
    } else {
        printf("The SLE can't be solved by this iteration method. "
               "Too many iterations. Try to increase diagonal dominance.\n");
    }
    //------------------------------------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------------------------------------

    double end = MPI_Wtime();
    printf("Time = %f seconds\n%lld iterations\n", end - start, iterations_count);

    free(vector_Xn);
    free(vector_B);
    free(vector_AXn);
    free(vector_Yn);
    free(vector_AYn);
    free(vector_Xn_plus_1);
    free(matrix_A);
    return 0;
}
//======================================================================================================================
// =============================================Parallel=program========================================================
//======================================================================================================================
int ParallelProgram(int rank) {
    printf("Parallel program started\n");
    int ranks_count;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks_count);

    int* rank_rows_manager = (int*) malloc (ranks_count * sizeof(int));

    for (int iter = 0; iter < ranks_count; ++iter) {
        rank_rows_manager[iter] = global_N / ranks_count + (iter < global_N % ranks_count);
    }

    double* local_matrix_A;
    int local_matrix_rows_count = rank_rows_manager[rank];
    AllocateMatrixContinuously(&local_matrix_A, local_matrix_rows_count);

    double* vector_B;
    AllocateVector(&vector_B, global_N);

    double* full_matrix_A = NULL;
    if (rank == 0) {
        AllocateMatrixContinuously(&full_matrix_A, global_N);
        if (full_matrix_A == NULL) {
            perror("Failed to allocate matrix!\n");
            free(rank_rows_manager);
            free(vector_B);
            free(full_matrix_A);
            return 1;
        }
        FillContinuousMatrix(&full_matrix_A);
        FillVector(&vector_B, FILL_RANDOM, global_N);
    }

    int* ranks_receive_counts = (int*) malloc (ranks_count * sizeof (int));
    int* displaces = (int*) malloc (ranks_count * sizeof (int));
    int offset = 0;

    for (int i = 0; i < ranks_count; i++) {
        ranks_receive_counts[i] = rank_rows_manager[i];
        displaces[i] = offset;
        offset += rank_rows_manager[i];
    }

    double* vector_Xn;
    AllocateVector(&vector_Xn, global_N);
    FillVector(&vector_Xn, FILL_WITH_ZEROS, global_N);

    double* vector_local_AXn;
    AllocateVector(&vector_local_AXn, local_matrix_rows_count);

    double* full_vector_AXn;
    AllocateVector(&full_vector_AXn, global_N);

    double* vector_Yn;
    AllocateVector(&vector_Yn, global_N);

    double* vector_local_AYn;
    AllocateVector(&vector_local_AYn, local_matrix_rows_count);

    double* full_vector_AYn;
    AllocateVector(&full_vector_AYn, global_N);

    double tau;

    double* vector_Xn_plus_1;
    AllocateVector(&vector_Xn_plus_1, global_N);

    MPI_Scatterv(full_matrix_A,
                 ranks_receive_counts,
                 displaces,
                 MPI_DOUBLE,
                 local_matrix_A,
                 ranks_receive_counts[rank],
                 MPI_DOUBLE,
                 ROOT_PROCESS_NUMBER,
                 MPI_COMM_WORLD);

    MPI_Bcast(vector_B, global_N, MPI_DOUBLE, ROOT_PROCESS_NUMBER, MPI_COMM_WORLD);

    double start = MPI_Wtime();
    unsigned long long iterations_count = 0;
    double stop_criteria_value = STOP_CRITERIA_INIT_VALUE;

    while (stop_criteria_value >= epsilon && iterations_count < ITERATIONS_COUNT_ALLOWED) {
        //----------------------------------------vector_Xn_plus_1-calculation---------------------------------------------
        MultiplyLocalMatrixToVector(local_matrix_rows_count, local_matrix_A,
                                    vector_Xn, vector_local_AXn); // matrix_A * vector_Xn = vector_local_AXn

        MPI_Allgatherv(vector_local_AXn,
                       local_matrix_rows_count,
                       MPI_DOUBLE,
                       full_vector_AXn,
                       ranks_receive_counts,
                       displaces, MPI_DOUBLE,
                       MPI_COMM_WORLD);

        SubtractVectorFromVector(full_vector_AXn, vector_B,
                                 vector_Yn, global_N);
        //vector_Yn = matrix_A * vector_Xn - vector_B

        MultiplyLocalMatrixToVector(local_matrix_rows_count, local_matrix_A,
                                    vector_Yn, vector_local_AYn); // matrix_A * vector_Yn = AYn

        MPI_Allgatherv(vector_local_AYn,
                       local_matrix_rows_count,
                       MPI_DOUBLE,
                       full_vector_AYn,
                       ranks_receive_counts,
                       displaces,
                       MPI_DOUBLE,
                       MPI_COMM_WORLD);

        tau = (ScalarProduct(vector_Yn, full_vector_AYn)) /
              (ScalarProduct(full_vector_AYn, full_vector_AYn)); // tauN = (Yn, AYn) / (AYn, AYn)

        stop_criteria_value = ScalarProduct(vector_Yn, vector_Yn) /
                              ScalarProduct(vector_B, vector_B); // |AXn - vector_B|^2 / |vector_B|^2

        MultiplyVectorToScalar(vector_Yn, tau); // counting vector_tau_Yn

        SubtractVectorFromVector(vector_Xn, vector_Yn, vector_Xn_plus_1,
                                 global_N); // vector_Xn_plus_1 = vector_Xn - vector_tau_Yn
        //---------------------------------------stop-flag-counting-and-checking----------------------------------------
        iterations_count++;
        CopyVectorsValueToOtherVector(vector_Xn_plus_1, vector_Xn, global_N);
    }
    if (iterations_count < ITERATIONS_COUNT_ALLOWED && stop_criteria_value < epsilon) {
        printf("SLE solved!\n");
    } else {
        printf("The SLE can't be solved by this iteration method. "
               "Too many iterations. Try to increase diagonal dominance.\n");
    }

    free(rank_rows_manager);
    free(ranks_receive_counts);
    free(displaces);
    if (rank == 0) {
        free(full_matrix_A);
    }
    free(vector_Xn);
    free(vector_B);
    free(vector_local_AXn);
    free(vector_Yn);
    free(vector_local_AYn);
    free(vector_Xn_plus_1);
    free(full_vector_AXn);
    free(full_vector_AYn);
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
            ret = ParallelProgram(rank);
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
            ret = SequentialProgram();
        }
    }
    MPI_Finalize();
    return ret;
}