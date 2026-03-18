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

//int Minimum(int a, int b) {
//    return (a < b) ? a : b;
//}

//void PrintVector(double* vector, int vectors_length) {
//    for (int iter = 0; iter < vectors_length; ++iter) {
//        printf("%f\n", vector[iter]);
//    }
//}

//void FillMatrix(double*** full_matrix_A) {
//    for (global_matrix_row = 0; global_matrix_row < global_N; global_matrix_row++) {
//        for (global_matrix_column = 0; global_matrix_column < global_N; ++global_matrix_column) {
//
//            if (global_matrix_row == global_matrix_column) {
//                (*full_matrix_A)[global_matrix_row][global_matrix_column] =
//                        INT_MAX - (global_matrix_column % RANGE_MODULE);
//            } else {
//                (*full_matrix_A)[global_matrix_row][global_matrix_column] =
//                        global_matrix_column % RANGE_MODULE;
//            }
//        }
//    }
//}

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
    for (int index = 0; index < vectors_size; ++index) {
        res_vector[index] = minuend_vector[index] - subtrahend_vector[index];
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
    for (int iter = 0; iter < global_N; ++iter) {
        product += vector_1[iter] * vector_2[iter];
    }
    return product;
}

void MultiplyVectorToScalar(double* vector, double scalar) {
    for (int iter = 0; iter < global_N; ++iter) {
        vector[iter] *= scalar;
    }
}

void CopyVector(const double* vector, double* vectors_copy, int vectors_length) {
    for (int iterator = 0; iterator < vectors_length; ++iterator) {
        vectors_copy[iterator] = vector[iterator];
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
        CopyVector(vector_Xn_plus_1, vector_Xn, local_matrix_rows_count);
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

    int* counts_rank_elements = (int*) malloc (ranks_count * sizeof (int));
    int* displaces = (int*) malloc (ranks_count * sizeof (int));

    for (int iter = 0; iter < ranks_count; ++iter) {
        counts_rank_elements[iter] = rank_rows_manager[iter] * global_N;
    }
    for (int iter = 0; iter < ranks_count; ++iter) {
        if (iter == 0) {
            displaces[iter] = 0;
        } else {
            displaces[iter] = displaces[iter - 1] + counts_rank_elements[iter - 1];
        }
    }

    double* vector_Xn;
    AllocateVector(&vector_Xn, local_matrix_rows_count);
    FillVector(&vector_Xn, FILL_WITH_ZEROS, local_matrix_rows_count);

    double* vector_AXn;
    AllocateVector(&vector_AXn, local_matrix_rows_count);

    double* vector_Yn;
    AllocateVector(&vector_Yn, local_matrix_rows_count);

    double* vector_AYn;
    AllocateVector(&vector_AYn, local_matrix_rows_count);

    double tau;

    double* vector_Xn_plus_1;
    AllocateVector(&vector_Xn_plus_1, local_matrix_rows_count);


    MPI_Scatterv(full_matrix_A,
                 counts_rank_elements,
                 displaces,
                 MPI_DOUBLE,
                 local_matrix_A,
                 counts_rank_elements[rank],
                 MPI_DOUBLE,
                 ROOT_PROCESS_NUMBER,
                 MPI_COMM_WORLD);

    MPI_Bcast(vector_B, global_N, MPI_DOUBLE, ROOT_PROCESS_NUMBER, MPI_COMM_WORLD);

    double start = MPI_Wtime();
    unsigned long long iterations_count = 0;
    double stop_criteria_value = STOP_CRITERIA_INIT_VALUE;

    while (stop_criteria_value >= epsilon && iterations_count < ITERATIONS_COUNT_ALLOWED) {
        //----------------------------------------vector_Xn_plus_1-calculation---------------------------------------------
        MultiplyLocalMatrixToVector(global_N, local_matrix_A,
                                    vector_Xn, vector_AXn); // matrix_A * vector_Xn = vector_AXn

        SubtractVectorFromVector(vector_AXn, vector_B,
                                 vector_Yn, local_matrix_rows_count);
        //vector_Yn = matrix_A * vector_Xn - vector_B

        MultiplyLocalMatrixToVector(local_matrix_rows_count, local_matrix_A,
                                    vector_Yn, vector_AYn); // matrix_A * vector_Yn = AYn

        tau = (ScalarProduct(vector_Yn, vector_AYn)) /
               (ScalarProduct(vector_AYn, vector_AYn)); // tauN = (Yn, AYn) / (AYn, AYn)

        stop_criteria_value = ScalarProduct(vector_Yn, vector_Yn) /
                              ScalarProduct(vector_B, vector_B); // |AXn - vector_B|^2 / |vector_B|^2

        MultiplyVectorToScalar(vector_Yn, tau); // counting vector_tau_Yn

        SubtractVectorFromVector(vector_Xn, vector_Yn, vector_Xn_plus_1,
                                 local_matrix_rows_count); // vector_Xn_plus_1 = vector_Xn - vector_tau_Yn
        //---------------------------------------stop-flag-counting-and-checking----------------------------------------
        iterations_count++;
        CopyVector(vector_Xn_plus_1, vector_Xn, local_matrix_rows_count);
    }
    if (iterations_count < ITERATIONS_COUNT_ALLOWED && stop_criteria_value < epsilon) {
        printf("SLE solved!\n");
    } else {
        printf("The SLE can't be solved by this iteration method. "
               "Too many iterations. Try to increase diagonal dominance.\n");
    }

    free(rank_rows_manager);
    free(counts_rank_elements);
    free(displaces);
    if (rank == 0) {
        free(full_matrix_A);
    }
    free(vector_Xn);
    free(vector_B);
    free(vector_AXn);
    free(vector_Yn);
    free(vector_AYn);
    free(vector_Xn_plus_1);
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