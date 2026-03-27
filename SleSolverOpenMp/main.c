#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define RANGE_MODULE 100
#define HARDEN_MATRIX_TRACE 10
#define SUCCESS 0
#define NUMBER_SYSTEM_BASE_10 10
#define FILL_WITH_ZEROS 0
#define FILL_RANDOM 1
#define TRUE 1
#define EPSILON_DEFAULT_VALUE 0.00001
#define GLOBAL_N_DEFAULT_VALUE 10000
#define ITERATIONS_PER_PROCESS_ALLOWED 1000000
#define CHUNK_SIZE_DEFAULT_VALUE 1

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

int ParallelProgramScheduleStatic(char* keyword, int threads_num, int chunk_size) {
    printf("Parallel schedule static program started\nthreads_num = %d\nchunk_size = %d\n",
           threads_num, chunk_size);

    double* matrix_A;
    AllocateMatrixContinuously(&matrix_A, global_N);
    FillContinuousMatrix(&matrix_A);

    if (matrix_A == NULL) {
        perror("Failed to allocate matrix!\n");
        return 1;
    }

    double* vector_Xn;
    AllocateVector(&vector_Xn, global_N);
    if (vector_Xn == NULL) {
        perror("Failed to allocate vector_Xn!\n");
        free(matrix_A);
        return 1;
    }
    FillVector(&vector_Xn, FILL_WITH_ZEROS, global_N);

    double* vector_B;
    AllocateVector(&vector_B, global_N);
    if (vector_B == NULL) {
        perror("Failed to allocate vector_B!\n");
        free(vector_Xn); free(matrix_A);
        return 1;
    }
    FillVector(&vector_B, FILL_RANDOM, global_N);
    vector_B[global_N - 1] += (double)(global_N * HARDEN_MATRIX_TRACE);

    double* vector_AXn = NULL;
    AllocateVector(&vector_AXn, global_N);
    double* vector_Yn = NULL;
    AllocateVector(&vector_Yn, global_N);
    double* vector_AYn = NULL;
    AllocateVector(&vector_AYn, global_N);
    double scalar_product_Yn_AYn = 0, scalar_product_AYn_AYn = 0, scalar_product_Yn_Yn = 0;

    if (vector_AXn == NULL || vector_Yn == NULL || vector_AYn == NULL) {
        perror("Failed to allocate vectors!\n");
        free(vector_Xn); free(vector_B); free(matrix_A);
        free(vector_AXn); free(vector_Yn); free(vector_AYn);
        return 1;
    }

    double norm_B = 0;

    double stop_criteria_value;
    unsigned long long iterations_count = 0;
    double time_out;
    int done = 0;
    double start = omp_get_wtime();
    double tau_n;

    omp_set_num_threads(threads_num);
#pragma omp parallel
    {
#pragma omp for reduction(+: norm_B) schedule(static, chunk_size)
        for (int i = 0; i < global_N; ++i) {
            norm_B += vector_B[i] * vector_B[i];
        }
        norm_B = sqrt(norm_B);
        while (TRUE) {
        // AXn
#pragma omp for schedule(static, chunk_size)
        for (int i = 0; i < global_N; ++i) {
            double row_res = 0;
            for (int j = 0; j < global_N; ++j) {
                row_res += matrix_A[i * global_N + j] * vector_Xn[j];
            }
            vector_AXn[i] = row_res;
        }

        // Yn = AXn - B
#pragma omp for schedule(static, chunk_size)
        for (int i = 0; i < global_N; ++i) {
            vector_Yn[i] = vector_AXn[i] - vector_B[i];
        }

        // AYn
#pragma omp for schedule(static, chunk_size)
        for (int i = 0; i < global_N; ++i) {
            double row_res = 0;
            for (int j = 0; j < global_N; ++j) {
                row_res += matrix_A[i * global_N + j] * vector_Yn[j];
            }
            vector_AYn[i] = row_res;
        }

#pragma omp for reduction(+:scalar_product_Yn_AYn, scalar_product_AYn_AYn, scalar_product_Yn_Yn) schedule(static, chunk_size)
        for (int i = 0; i < global_N; ++i) {
            scalar_product_Yn_AYn += vector_Yn[i] * vector_AYn[i];
            scalar_product_AYn_AYn += vector_AYn[i] * vector_AYn[i];
            scalar_product_Yn_Yn += vector_Yn[i] * vector_Yn[i];
        }

#pragma omp single
        {
            tau_n = scalar_product_Yn_AYn / scalar_product_AYn_AYn;
            stop_criteria_value = (sqrt(scalar_product_Yn_Yn)) / norm_B;
            time_out = omp_get_wtime();
            if (stop_criteria_value < epsilon || iterations_count >= ITERATIONS_PER_PROCESS_ALLOWED ||
            time_out - start > 10.0) {
                done = TRUE;
            } else {
                iterations_count++;
            }
            scalar_product_Yn_AYn = 0;
            scalar_product_AYn_AYn = 0;
            scalar_product_Yn_Yn = 0;
        }
        if (done) {
            break;
        }
        // Yn * tau_n
#pragma omp for schedule(static, chunk_size)
        for (int i = 0; i < global_N; ++i) {
            vector_Yn[i] *= tau_n;
        }

        // Xn+1 = Xn - tau_n*Yn
#pragma omp for schedule(static, chunk_size)
        for (int i = 0; i < global_N; ++i) {
            vector_Xn[i] = vector_Xn[i] - vector_Yn[i];
        }
        } // while
    } // parallel

    double end = omp_get_wtime();

    if (iterations_count < ITERATIONS_PER_PROCESS_ALLOWED && stop_criteria_value < epsilon)
        printf("SLE solved!\nTime = %f seconds\n", end - start);
    else
        printf("The SLE can't be solved by this iteration method. Too many iterations.\n");

    free(matrix_A);
    free(vector_B);
    free(vector_Xn);
    free(vector_AXn);
    free(vector_Yn);
    free(vector_AYn);
    return 0;
}

int ParallelProgramScheduleDynamic(int chunk_size) {
    printf("Parallel schedule dynamic program started\n");

    return SUCCESS;
}

int ParallelProgramScheduleGuided(int chunk_size) {
    printf("Parallel schedule guided program started\n");

    return 0;
}

int main(int argc, char *argv[]) {
    int chunk_size = CHUNK_SIZE_DEFAULT_VALUE;
    if (argc > 5) {
        char *end;
        long val = strtol(argv[5], &end, NUMBER_SYSTEM_BASE_10);
        if (end == argv[5] || *end != '\0' || val <= 0) {
            printf("Invalid chunk size value, using default %d - argc\n", argc);
        } else {
            chunk_size = (int)val;
        }
    }
    if (argc > 4) {
        char *end;
        double val = strtod(argv[4], &end);
        if (end == argv[4] || *end != '\0')
            epsilon = EPSILON_DEFAULT_VALUE;
        else
            epsilon = val;
    }

    int ret = 0;
    if (argc > 2) {
        char *end;
        long val = strtol(argv[2], &end, NUMBER_SYSTEM_BASE_10);
        if (end == argv[2] || *end != '\0' || val <= 0) {
            printf("Invalid global_N value, using default %d\n", GLOBAL_N_DEFAULT_VALUE);
            global_N = GLOBAL_N_DEFAULT_VALUE;
        } else {
            global_N = (int)val;
        }
    } else {
        global_N = GLOBAL_N_DEFAULT_VALUE;
    }

    int threads_num = 1;
    if (argc > 1) {
        char *end;
        long val = strtol(argv[1], &end, NUMBER_SYSTEM_BASE_10);
        if (end == argv[1] || *end != '\0' || val <= 0) {
            printf("Invalid chunk size value, using default %d - argc\n", argc);
        } else {
            threads_num = (int)val;
        }
    }

    if (argc > 3) {
        if (strcmp(argv[3], "-s") == SUCCESS) {

        } else if (strcmp(argv[3], "-ps") == SUCCESS) {
            char keyword[] = "static";
            ret = ParallelProgramScheduleStatic(keyword, threads_num, chunk_size);
        } else if (strcmp(argv[3], "-pd") == SUCCESS) {
            char keyword[] = "dynamic";
            ret = ParallelProgramScheduleStatic(keyword, threads_num, chunk_size);
        } else if (strcmp(argv[3], "-pg") == SUCCESS) {
            char keyword[] = "guided";
            ret = ParallelProgramScheduleStatic(keyword, threads_num, chunk_size);
        }
        else {
            printf("Unknown flag. Usage:\n");
            printf("  %s [global_N] -s           (sequential)\n", argv[0]);
            printf("  %s [global_N] -ps          (parallel with schedule static)\n", argv[0]);
            printf("  %s [global_N] -pd          (parallel with schedule dynamic)\n", argv[0]);
            printf("  %s [global_N] -pg          (parallel with schedule guided)\n", argv[0]);
            printf("      global_N - vector size (positive integer, default %d)\n", GLOBAL_N_DEFAULT_VALUE);
        }
    } else {
        printf("No flags specified. Running sequential by default.\n"
               "global_N is equal to %d\n", GLOBAL_N_DEFAULT_VALUE);
        ret = -1;
    }

    return ret;
}