#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <mpi.h>

#define RANGE_MODULE 100
#define HARDEN_MATRIX_TRACE 1.0005
#define ROOT_PROCESS_NUMBER 0
#define TRUE 0
#define NUMBER_SYSTEM_BASE_10 10
#define FILL_WITH_ZEROS 0
#define FILL_RANDOM 1
#define STOP_CRITERIA_INIT_VALUE 100
#define EPSILON_DEFAULT_VALUE 0.00001
#define GLOBAL_N_DEFAULT_VALUE 10000
#define ITERATIONS_PER_PROCESS_ALLOWED 10000

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
                (*matrix)[global_N * i + j] = (double)(global_N * HARDEN_MATRIX_TRACE);
            } else {
                (*matrix)[global_N * i + j] = (double)(j % RANGE_MODULE);
            }
        }
    }
}

void SubtractVectorFromVector(const double* minuend_vector, const double* subtrahend_vector,
                              double* res_vector, int vectors_size) {
    for (int i = 0; i < vectors_size; ++i) {
        res_vector[i] = minuend_vector[i] - subtrahend_vector[i];
    }
}

void MultiplyLocalMatrixToVector(int mtx_rows_count, const double* matrix,
                                 const double* vector, double* res_vector) {
    for (int i = 0; i < mtx_rows_count; ++i) {
        res_vector[i] = 0.0;
        for (int j = 0; j < global_N; ++j) {
            res_vector[i] += matrix[i * global_N + j] * vector[j];
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

    double* vector_Xn = NULL;
    AllocateVector(&vector_Xn, local_matrix_rows_count);
    if (vector_Xn == NULL) {
        perror("Failed to allocate vector_Xn!\n");
        free(matrix_A);
        return 1;
    }
    FillVector(&vector_Xn, FILL_WITH_ZEROS, local_matrix_rows_count);

    double* vector_B;
    AllocateVector(&vector_B, global_N);
    if (vector_B == NULL) {
        perror("Failed to allocate vector_B!\n");
        free(vector_Xn);
        free(matrix_A);
        return 1;
    }
    FillVector(&vector_B, FILL_RANDOM, global_N);
    vector_B[global_N - 1] += (double)(global_N * HARDEN_MATRIX_TRACE);

    double* vector_AXn = NULL;
    AllocateVector(&vector_AXn, local_matrix_rows_count);
    double* vector_Yn = NULL;
    AllocateVector(&vector_Yn, local_matrix_rows_count);
    double* vector_AYn = NULL;
    AllocateVector(&vector_AYn, local_matrix_rows_count);
    double* vector_Xn_plus_1 = NULL;
    AllocateVector(&vector_Xn_plus_1, local_matrix_rows_count);

    if (vector_AXn == NULL || vector_Yn == NULL || vector_AYn == NULL || vector_Xn_plus_1 == NULL) {
        perror("Failed to allocate vectors!\n");
        free(vector_Xn); free(vector_B); free(matrix_A);
        free(vector_AXn); free(vector_Yn); free(vector_AYn); free(vector_Xn_plus_1);
        return 1;
    }

    double stop_criteria_value = STOP_CRITERIA_INIT_VALUE;
    double start = MPI_Wtime();
    double time_out = 0;
    unsigned long long iterations_count = 0;

    while (stop_criteria_value >= epsilon &&
           iterations_count < ITERATIONS_PER_PROCESS_ALLOWED &&
           time_out - start < 40) {
        MultiplyLocalMatrixToVector(global_N, matrix_A, vector_Xn, vector_AXn);
        SubtractVectorFromVector(vector_AXn, vector_B,
                                 vector_Yn, local_matrix_rows_count);
        MultiplyLocalMatrixToVector(local_matrix_rows_count, matrix_A,
                                    vector_Yn, vector_AYn);

        double tauN = ScalarProduct(vector_Yn, vector_AYn) /
                      ScalarProduct(vector_AYn, vector_AYn);
        stop_criteria_value = ScalarProduct(vector_Yn, vector_Yn) /
                              ScalarProduct(vector_B, vector_B);

        MultiplyVectorToScalar(vector_Yn, tauN);
        SubtractVectorFromVector(vector_Xn, vector_Yn,
                                 vector_Xn_plus_1, local_matrix_rows_count);
        iterations_count++;
        time_out = MPI_Wtime();
        CopyVectorsValueToOtherVector(vector_Xn_plus_1, vector_Xn,
                                      local_matrix_rows_count);
    }

    double end = MPI_Wtime();
    if (iterations_count < ITERATIONS_PER_PROCESS_ALLOWED && stop_criteria_value < epsilon &&
        time_out - start <= 40.0)
        printf("SLE solved!\n");
    else
        printf("The SLE can't be solved by this iteration method. Too many iterations."
               " Try to increase diagonal dominance.\n");
    printf("Time = %f seconds\n%lld iterations\n", end - start, iterations_count);

    free(vector_Xn); free(vector_B);
    free(vector_AXn); free(vector_Yn);
    free(vector_AYn); free(vector_Xn_plus_1);
    free(matrix_A);
    return 0;
}

int ParallelProgram(int rank) {
    if (rank == 0) {
        printf("Parallel program started\n");
    }

    return 0;
}

int main(int argc, char *argv[]) {
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    epsilon = EPSILON_DEFAULT_VALUE;

    if (argc > 3) {
        if (rank == ROOT_PROCESS_NUMBER) {
            char *end;
            double val = strtod(argv[3], &end);
            if (end == argv[3] || *end != '\0')
                epsilon = EPSILON_DEFAULT_VALUE;
            else
                epsilon = val;
        }
    }
    MPI_Bcast(&epsilon, 1, MPI_DOUBLE, ROOT_PROCESS_NUMBER, MPI_COMM_WORLD);

    int default_N = GLOBAL_N_DEFAULT_VALUE;
    if (argc > 2) {
        if (rank == ROOT_PROCESS_NUMBER) {
            char *end;
            long val = strtol(argv[2], &end, NUMBER_SYSTEM_BASE_10);
            if (end == argv[2] || *end != '\0' || val <= 0) {
                printf("Invalid global_N value, using default %d\n", default_N);
                global_N = default_N;
            } else {
                global_N = (int)val;
            }
        }
    } else {
        global_N = default_N;
    }
    MPI_Bcast(&global_N, 1, MPI_INT, ROOT_PROCESS_NUMBER, MPI_COMM_WORLD);

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