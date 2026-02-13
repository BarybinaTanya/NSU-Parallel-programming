#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#define RANGE_MODULE 1000
#define N 1000

void FillVectors(int* vector_a, int* vector_b) {
    for (int i = 0; i < N; i++) {
        vector_a[i] = i % RANGE_MODULE;
    }
    for (int i = 0; i < N; i++) {
        vector_b[i] = (N - i) % RANGE_MODULE;
    }
}

int SequentialProgram(int argc, char **argv) {
    printf("I'm sequential program :) \n");

    MPI_Init(&argc, &argv);
    double time_start = MPI_Wtime();

    unsigned long long sum = 0;
    int *vector_a = (int*) malloc(N * sizeof(int));
    int *vector_b = (int*) malloc(N * sizeof(int));
    FillVectors(vector_a, vector_b);

    for (int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            sum += vector_a[i] * vector_b[j];

    double time_end = MPI_Wtime();
    printf("sum = %lld \ntime: %f \n", sum, time_end - time_start);

    free(vector_a);
    free(vector_b);
    MPI_Finalize();
    return 0;
}

int ParallelPointToPoint (int argc, char **argv) {
    printf("I'm parallel point to point program ;) \n");
    return 0;
}

int ParallelGroupCommunications (int argc, char **argv) {
    printf("I'm parallel group program :D \n");
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
