#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Function to print a matrix given as a flat pointer
void printMatrix(int rows, int cols, int *matrix) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%3d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int NUM_MATRICES, MATRIX_ROWS, MATRIX_COLS, MATRIX_B_COLS;
    double startTime, endTime;

    // Input only by root process
    if (rank == 0) {
        printf("Enter Number of matrices: ");
        fflush(stdout);
        scanf("%d", &NUM_MATRICES);
    
        printf("Enter Number of rows in matrix A: ");
        fflush(stdout);
        scanf("%d", &MATRIX_ROWS);
    
        printf("Enter Number of columns in matrix A: ");
        fflush(stdout);
        scanf("%d", &MATRIX_COLS);
    
        printf("Enter Number of columns in matrix B: ");
        fflush(stdout);
        scanf("%d", &MATRIX_B_COLS);
    }

    // Broadcast matrix info to all processes
    MPI_Bcast(&NUM_MATRICES, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&MATRIX_ROWS, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&MATRIX_COLS, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&MATRIX_B_COLS, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (NUM_MATRICES % size != 0) {
        if (rank == 0) {
            printf("Number of matrices must be divisible by number of processes.\n");
        }
        MPI_Finalize();
        return 1;
        
    }

    int matrices_per_proc = NUM_MATRICES / size;

    // Allocate memory for matrices on root process
    int *matrices = NULL;
    int *matrixB = NULL;
    if (rank == 0) {
        matrices = (int*)malloc(NUM_MATRICES * MATRIX_ROWS * MATRIX_COLS * sizeof(int));
        matrixB = (int*)malloc(NUM_MATRICES * MATRIX_COLS * MATRIX_B_COLS * sizeof(int));

        // Initialize matrices with random numbers 0-9
        for (int k = 0; k < NUM_MATRICES; k++) {
            for (int i = 0; i < MATRIX_ROWS; i++)
                for (int j = 0; j < MATRIX_COLS; j++)
                    matrices[k * MATRIX_ROWS * MATRIX_COLS + i * MATRIX_COLS + j] = rand() % 10;

            for (int i = 0; i < MATRIX_COLS; i++)
                for (int j = 0; j < MATRIX_B_COLS; j++)
                    matrixB[k * MATRIX_COLS * MATRIX_B_COLS + i * MATRIX_B_COLS + j] = rand() % 10;
        }
    }

    // Allocate local matrices for each process
    int *localA = (int*)malloc(matrices_per_proc * MATRIX_ROWS * MATRIX_COLS * sizeof(int));
    int *localB = (int*)malloc(matrices_per_proc * MATRIX_COLS * MATRIX_B_COLS * sizeof(int));
    int *localR = (int*)malloc(matrices_per_proc * MATRIX_ROWS * MATRIX_B_COLS * sizeof(int));

    MPI_Barrier(MPI_COMM_WORLD);
    startTime = MPI_Wtime();

    // Scatter matrices to all processes
    MPI_Scatter(matrices, matrices_per_proc * MATRIX_ROWS * MATRIX_COLS, MPI_INT,
                localA, matrices_per_proc * MATRIX_ROWS * MATRIX_COLS, MPI_INT,
                0, MPI_COMM_WORLD);

    MPI_Scatter(matrixB, matrices_per_proc * MATRIX_COLS * MATRIX_B_COLS, MPI_INT,
                localB, matrices_per_proc * MATRIX_COLS * MATRIX_B_COLS, MPI_INT,
                0, MPI_COMM_WORLD);

    // Multiply local matrices
    for (int k = 0; k < matrices_per_proc; k++) {
        for (int i = 0; i < MATRIX_ROWS; i++) {
            for (int j = 0; j < MATRIX_B_COLS; j++) {
                localR[k * MATRIX_ROWS * MATRIX_B_COLS + i * MATRIX_B_COLS + j] = 0;
                for (int l = 0; l < MATRIX_COLS; l++) {
                    localR[k * MATRIX_ROWS * MATRIX_B_COLS + i * MATRIX_B_COLS + j] +=
                        localA[k * MATRIX_ROWS * MATRIX_COLS + i * MATRIX_COLS + l] *
                        localB[k * MATRIX_COLS * MATRIX_B_COLS + l * MATRIX_B_COLS + j];
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    endTime = MPI_Wtime();

    // Gather results to root process
    int *gatheredR = NULL;
    if (rank == 0) {
        gatheredR = (int*)malloc(NUM_MATRICES * MATRIX_ROWS * MATRIX_B_COLS * sizeof(int));
    }

    MPI_Gather(localR, matrices_per_proc * MATRIX_ROWS * MATRIX_B_COLS, MPI_INT,
               gatheredR, matrices_per_proc * MATRIX_ROWS * MATRIX_B_COLS, MPI_INT,
               0, MPI_COMM_WORLD);

    // Gather computation times
    double *allTimes = NULL;
    if (rank == 0)
        allTimes = (double*)malloc(size * sizeof(double));

    double elapsed = endTime - startTime;
    MPI_Gather(&elapsed, 1, MPI_DOUBLE, allTimes, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Root prints times in order
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            printf("Process %d: Time taken = %f seconds\n", i, allTimes[i]);
        }
        free(allTimes);
    }

    // Root prints all matrices
    if (rank == 0) {
        printf("\nAll %d Result Matrices:\n", NUM_MATRICES);
        for (int k = 0; k < NUM_MATRICES; k++) {
            printf("Matrix %d:\n", k);
            printMatrix(MATRIX_ROWS, MATRIX_B_COLS, &gatheredR[k * MATRIX_ROWS * MATRIX_B_COLS]);
        }
    }

    // Free memory
    free(localA); free(localB); free(localR);
    if (rank == 0) { free(matrices); free(matrixB); free(gatheredR); }

    MPI_Finalize();
    return 0;
}

//mpicc -o matrix matrix.c
//mpirun -np 4 ./matrix
