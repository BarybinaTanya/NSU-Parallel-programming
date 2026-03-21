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
    int ranks_count;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks_count);

    int* rank_rows_manager = (int*)malloc(ranks_count * sizeof(int));
    for (int iter = 0; iter < ranks_count; ++iter) {
        rank_rows_manager[iter] = global_N / ranks_count + (iter < global_N % ranks_count);
    }

    int local_matrix_rows_count = rank_rows_manager[rank];
    double* local_matrix_A = NULL;
    AllocateMatrixContinuously(&local_matrix_A, local_matrix_rows_count);
    if (local_matrix_A == NULL) {
        free(rank_rows_manager);
        return 1;
    }

    double* vector_B = NULL;
    AllocateVector(&vector_B, global_N);
    if (vector_B == NULL) {
        free(rank_rows_manager);
        free(local_matrix_A);
        return 1;
    }

    double* full_matrix_A = NULL;
    if (rank == 0) {
        AllocateMatrixContinuously(&full_matrix_A, global_N);
        if (full_matrix_A == NULL) {
            perror("Failed to allocate matrix!\n");
            free(rank_rows_manager);
            free(vector_B);
            return 1;
        }
        FillContinuousMatrix(&full_matrix_A);
        FillVector(&vector_B, FILL_RANDOM, global_N);
        vector_B[global_N - 1] += (double)(global_N * HARDEN_MATRIX_TRACE); // !!!
    }

    int *mat_send_counts = (int*)malloc(ranks_count * sizeof(int));
    int *mat_displaces = (int*)malloc(ranks_count * sizeof(int));
    int *vec_receive_counts = (int*)malloc(ranks_count * sizeof(int));
    int *vec_displaces = (int*)malloc(ranks_count * sizeof(int));

    if (mat_send_counts == NULL || mat_displaces == NULL ||
     vec_receive_counts == NULL || vec_displaces == NULL) {
        perror("Failed to allocate communication arrays!\n");
        free(rank_rows_manager);
        free(vector_B);
        if (rank == 0) free(full_matrix_A);
        free(local_matrix_A);
        return 1;
    }

    int offset_mat = 0;
    int offset_vec = 0;
    for (int i = 0; i < ranks_count; i++) {
        vec_receive_counts[i] = rank_rows_manager[i];
        vec_displaces[i] = offset_vec;
        offset_vec += rank_rows_manager[i];

        mat_send_counts[i] = rank_rows_manager[i] * global_N;
        mat_displaces[i] = offset_mat;
        offset_mat += mat_send_counts[i];
    }

    double* vector_Xn = NULL;
    AllocateVector(&vector_Xn, global_N);
    FillVector(&vector_Xn, FILL_WITH_ZEROS, global_N);

    double* vector_local_AXn = NULL;
    AllocateVector(&vector_local_AXn, local_matrix_rows_count);

    double* full_vector_AXn = NULL;
    AllocateVector(&full_vector_AXn, global_N);

    double* vector_Yn = NULL;
    AllocateVector(&vector_Yn, global_N);

    double* vector_local_AYn = NULL;
    AllocateVector(&vector_local_AYn, local_matrix_rows_count);

    double* full_vector_AYn = NULL;
    AllocateVector(&full_vector_AYn, global_N);

    double* vector_Xn_plus_1 = NULL;
    AllocateVector(&vector_Xn_plus_1, global_N);

    if (vector_Xn == NULL || vector_local_AXn == NULL || full_vector_AXn == NULL ||
        vector_Yn == NULL || vector_local_AYn == NULL || full_vector_AYn == NULL || vector_Xn_plus_1 == NULL) {
        perror("Failed to allocate work vectors!\n");
        free(rank_rows_manager); free(vector_B);
        if (rank == 0) {
            free(full_matrix_A);
        }
        free(mat_send_counts); free(mat_displaces);
        free(vec_receive_counts); free(vec_displaces);
        free(local_matrix_A);
        free(vector_Xn); free(vector_local_AXn); free(full_vector_AXn);
        free(vector_Yn); free(vector_local_AYn);
        free(full_vector_AYn); free(vector_Xn_plus_1);
        return 1;
    }

    MPI_Scatterv(full_matrix_A,
                 mat_send_counts,
                 mat_displaces,
                 MPI_DOUBLE,
                 local_matrix_A,
                 mat_send_counts[rank],
                 MPI_DOUBLE,
                 ROOT_PROCESS_NUMBER,
                 MPI_COMM_WORLD);

    MPI_Bcast(vector_B,
              global_N,
              MPI_DOUBLE,
              ROOT_PROCESS_NUMBER,
              MPI_COMM_WORLD);

    double counting_cycle_start = MPI_Wtime();
    double cycle_time_out = 0;
    unsigned long long local_proc_iterations_count = 0;
    double stop_criteria_value = STOP_CRITERIA_INIT_VALUE;

    while (stop_criteria_value >= epsilon &&
           local_proc_iterations_count < ITERATIONS_PER_PROCESS_ALLOWED &&
           cycle_time_out - counting_cycle_start < 40.0) {

        MultiplyLocalMatrixToVector(local_matrix_rows_count,
                                    local_matrix_A,
                                    vector_Xn,
                                    vector_local_AXn);

        MPI_Allgatherv(vector_local_AXn,
                       local_matrix_rows_count,
                       MPI_DOUBLE,
                       full_vector_AXn,
                       vec_receive_counts,
                       vec_displaces,
                       MPI_DOUBLE,
                       MPI_COMM_WORLD);

        SubtractVectorFromVector(full_vector_AXn,
                                 vector_B,
                                 vector_Yn,
                                 global_N);

        MultiplyLocalMatrixToVector(local_matrix_rows_count,
                                    local_matrix_A,
                                    vector_Yn,
                                    vector_local_AYn);

        MPI_Allgatherv(vector_local_AYn,
                       local_matrix_rows_count,
                       MPI_DOUBLE,
                       full_vector_AYn,
                       vec_receive_counts,
                       vec_displaces,
                       MPI_DOUBLE,
                       MPI_COMM_WORLD);

        double tau = ScalarProduct(vector_Yn, full_vector_AYn) /
                     ScalarProduct(full_vector_AYn, full_vector_AYn);

        stop_criteria_value = ScalarProduct(vector_Yn, vector_Yn) /
                              ScalarProduct(vector_B, vector_B);

        MultiplyVectorToScalar(vector_Yn, tau);
        SubtractVectorFromVector(vector_Xn,
                                 vector_Yn,
                                 vector_Xn_plus_1,
                                 global_N);

        local_proc_iterations_count++;
        CopyVectorsValueToOtherVector(vector_Xn_plus_1,
                                      vector_Xn,
                                      global_N);
        cycle_time_out = MPI_Wtime();
    }

    if (local_proc_iterations_count < ITERATIONS_PER_PROCESS_ALLOWED && stop_criteria_value < epsilon) {
        if (rank == 0) {
            printf("SLE solved! Time: %f seconds\n", cycle_time_out - counting_cycle_start);
        }
    } else {
        if (rank == 0) {
            printf("The SLE can't be solved by this iteration method. "
                   "Too many iterations. Try to increase diagonal dominance.\n");
            printf("Time: %f seconds\n", cycle_time_out - counting_cycle_start);
        }
    }

    free(rank_rows_manager);
    free(mat_send_counts); free(mat_displaces);
    free(vec_receive_counts); free(vec_displaces);
    free(local_matrix_A);
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