#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdio.h>
#include <mpi.h>

#define RANGE_MODULE 100
#define ROOT_PROCESS_NUMBER 0
#define TRUE 0
#define NUMBER_SYSTEM_BASE_10 10
#define FILL_WITH_ZEROS 0
#define FILL_RANDOM 1
#define STOP_CRITERIA_INIT_VALUE 100
#define EPSILON_DEFAULT_VALUE 0.00001
#define GLOBAL_N_DEFAULT_VALUE 10000
#define ITERATIONS_COUNT_ALLOWED 100

double epsilon;
int global_N;
int global_matrix_row;
int global_matrix_column;

int Minimum(int a, int b) {
    return (a < b) ? a : b;
}

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

void FillMatrix(double*** full_matrix_A) {
    for (global_matrix_row = 0; global_matrix_row < global_N; global_matrix_row++) {
        for (global_matrix_column = 0; global_matrix_column < global_N; ++global_matrix_column) {

            if (global_matrix_row == global_matrix_column) {
                (*full_matrix_A)[global_matrix_row][global_matrix_column] =
                        global_N - (global_matrix_column % RANGE_MODULE);
            } else {
                (*full_matrix_A)[global_matrix_row][global_matrix_column] =
                        global_matrix_column % RANGE_MODULE;
            }
        }
    }
}

void FreeMatrix(double*** full_matrix_A) {
    for (global_matrix_row = 0; global_matrix_row < global_N; global_matrix_row++) {
        free((*full_matrix_A)[global_matrix_row]);
    }
    free(*full_matrix_A);
}

void AllocateMatrix(double*** full_matrix_A, int num_rows) {
    *full_matrix_A = (double** )malloc(num_rows * sizeof(double*));
    for (int i = 0; i < num_rows; ++i) {
        (*full_matrix_A)[i] = (double*)malloc(global_N * sizeof(double));
    }
}

void SubtractVectorFromVector (const double* minuend_vector, const double* subtrahend_vector,
                               double* res_vector, int vectors_size) {
    for (int index = 0; index < vectors_size; ++index) {
        res_vector[index] = minuend_vector[index] - subtrahend_vector[index];
    }
}

void MultiplyLocalMatrixToVector(int local_matrix_rows_count, double** local_matrix_A,
                                    const double* vector, double* res_vector) {
    FillVector(&res_vector, FILL_WITH_ZEROS, local_matrix_rows_count);

    for (int local_matrix_row_iter = 0; local_matrix_row_iter <
    local_matrix_rows_count; ++local_matrix_row_iter) {

        for (int vector_row_iter = 0; vector_row_iter < global_N; ++vector_row_iter) {
            res_vector[local_matrix_row_iter] +=
                    local_matrix_A[local_matrix_row_iter][vector_row_iter] * vector[vector_row_iter];
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

void MultiplyVectorToScalar(const double* vector, double* res_vector, double scalar) {
    for (int iter = 0; iter < global_N; ++iter) {
        res_vector[iter] = vector[iter] * scalar;
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

    double** matrixA;
    int local_matrix_rows_count = global_N;
    AllocateMatrix( &matrixA, local_matrix_rows_count);
    FillMatrix(&matrixA);

    double* vectorXn;
    AllocateVector(&vectorXn, local_matrix_rows_count);
    FillVector(&vectorXn, FILL_WITH_ZEROS, local_matrix_rows_count);

    double* vectorB;
    AllocateVector(&vectorB, global_N);
    FillVector(&vectorB, FILL_RANDOM, global_N);
    vectorB[global_N - 1] += global_N * 2; // diagonal dominance of the augmented matrix
    // of a system of linear equations. Constant 2 here means nothing - it can be easily
    // replaced by any other integer

    double* vectorAXn;
    AllocateVector(&vectorAXn, local_matrix_rows_count);

    double* vectorYn;
    AllocateVector(&vectorYn, local_matrix_rows_count);

    double* vectorAYn;
    AllocateVector(&vectorAYn, local_matrix_rows_count);

    double tauN;

    double* vectorXnPlus1;
    AllocateVector(&vectorXnPlus1, local_matrix_rows_count);
    double stop_criteria_value = STOP_CRITERIA_INIT_VALUE;

    double* vectorTauYn;
    AllocateVector(&vectorTauYn, local_matrix_rows_count);
    double start = MPI_Wtime();
    unsigned long long iterations_count = 0;

    //------------------------------------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------------------------------------
    while (stop_criteria_value >= epsilon && iterations_count < ITERATIONS_COUNT_ALLOWED) {
        //----------------------------------------vectorXnPlus1-calculation---------------------------------------------
        MultiplyLocalMatrixToVector(global_N, matrixA,
                                    vectorXn, vectorAXn); // matrixA * vectorXn = vectorAXn

        SubtractVectorFromVector(vectorAXn, vectorB,
                                 vectorYn, local_matrix_rows_count);
        //vectorYn = matrixA * vectorXn - vectorB

        MultiplyLocalMatrixToVector(local_matrix_rows_count, matrixA,
                                    vectorYn, vectorAYn); // matrixA * vectorYn = AYn

        tauN = (ScalarProduct(vectorYn, vectorAYn)) /
               (ScalarProduct(vectorAYn, vectorAYn)); // tauN = (Yn, AYn) / (AYn, AYn)

        MultiplyVectorToScalar(vectorYn, vectorTauYn, tauN); // counting vectorTauYn

        SubtractVectorFromVector(vectorXn, vectorTauYn,vectorXnPlus1,
                                 local_matrix_rows_count); // vectorXnPlus1 = vectorXn - vectorTauYn
        //---------------------------------------stop-flag-counting-and-checking----------------------------------------

        stop_criteria_value = ScalarProduct(vectorYn, vectorYn) /
                              ScalarProduct(vectorB, vectorB); // |AXn - vectorB|^2 / |vectorB|^2
        iterations_count++;
        CopyVector(vectorXnPlus1, vectorXn, local_matrix_rows_count);
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

    free(vectorXn);
    free(vectorB);
    free(vectorAXn);
    free(vectorYn);
    free(vectorAYn);
    free(vectorXnPlus1);
    free(vectorTauYn);
    FreeMatrix(&matrixA);
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
    int* rank_elements_counts = (int*) malloc (ranks_count * sizeof (int*));
    int* displaces = (int*) malloc (ranks_count * sizeof (int*));

    for (int iter = 0; iter < ranks_count; ++iter) {
        rank_rows_manager[iter] = global_N / ranks_count + (iter < global_N % ranks_count);
    }
    for (int iter = 0; iter < ranks_count; ++iter) {
        rank_elements_counts[iter] = rank_rows_manager[iter] * global_N;
    }
    for (int iter = 0; iter < ranks_count; ++iter) {
        if (iter == 0) {
            displaces[iter] = 0;
        } else {
            displaces[iter] = displaces[iter - 1] + rank_elements_counts[iter - 1]
        }
    }

    free(rank_rows_manager);
    free(rank_elements_counts);
    free(displaces);
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