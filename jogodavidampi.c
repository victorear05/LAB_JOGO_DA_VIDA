#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <string.h>

#define ind2d(i, j) (i) * (tam + 2) + j
#define POWMIN 3
#define POWMAX 10

double wall_time(void)
{
    struct timeval tv;
    struct timezone tz;

    gettimeofday(&tv, &tz);
    return (tv.tv_sec + tv.tv_usec / 1000000.0);
}

void UmaVidaMPI(int *tabulIn, int *tabulOut, int tam, int start_row, int end_row) {
    int i, j, vizviv;

    for (i = start_row; i <= end_row; i++) {
        for (j = 1; j <= tam; j++) {
            vizviv = tabulIn[ind2d(i - 1, j - 1)] + tabulIn[ind2d(i - 1, j)] +
                     tabulIn[ind2d(i - 1, j + 1)] + tabulIn[ind2d(i, j - 1)] +
                     tabulIn[ind2d(i, j + 1)] + tabulIn[ind2d(i + 1, j - 1)] +
                     tabulIn[ind2d(i + 1, j)] + tabulIn[ind2d(i + 1, j + 1)];

            if (tabulIn[ind2d(i, j)] && vizviv < 2)
                tabulOut[ind2d(i, j)] = 0;
            else if (tabulIn[ind2d(i, j)] && vizviv > 3)
                tabulOut[ind2d(i, j)] = 0;
            else if (!tabulIn[ind2d(i, j)] && vizviv == 3)
                tabulOut[ind2d(i, j)] = 1;
            else
                tabulOut[ind2d(i, j)] = tabulIn[ind2d(i, j)];
        }
    }
}

void DumpTabul(int *tabul, int tam, int first, int last, char *msg, int rank) {
    int i, ij;

    if (rank != 0) {
        return;
    }

    printf("%s; Dump posicoes [%d:%d, %d:%d] de tabuleiro %d x %d\n", msg, first, last, first, last, tam, tam);
    for (i = first; i <= last; i++) {
        printf("=");
    }
    printf("=\n");
    for (i = ind2d(first, 0); i <= ind2d(last, 0); i += ind2d(1, 0)) {
        for (ij = i + first; ij <= i + last; ij++) {
            printf("%c", tabul[ij] ? 'X' : '.');
        }
        printf("\n");
    }
    for (i = first; i <= last; i++) {
        printf("=");
    }
    printf("=\n");
}

void InitTabul(int *tabulIn, int *tabulOut, int tam, int rank) {
    int ij;

    for (ij = 0; ij < (tam + 2) * (tam + 2); ij++) {
        tabulIn[ij] = 0;
        tabulOut[ij] = 0;
    }

    if (rank == 0) {
        tabulIn[ind2d(1, 2)] = 1;
        tabulIn[ind2d(2, 3)] = 1;
        tabulIn[ind2d(3, 1)] = 1;
        tabulIn[ind2d(3, 2)] = 1;
        tabulIn[ind2d(3, 3)] = 1;
    }
}

int Correto(int *tabul, int tam) {
    int ij, cnt;

    cnt = 0;
    for (ij = 0; ij < (tam + 2) * (tam + 2); ij++) {
        cnt = cnt + tabul[ij];
    }

    return (cnt == 5 && tabul[ind2d(tam - 2, tam - 1)] &&
            tabul[ind2d(tam - 1, tam)] && tabul[ind2d(tam, tam - 2)] &&
            tabul[ind2d(tam, tam - 1)] && tabul[ind2d(tam, tam)]);
}

int main(int argc, char **argv) {
    int pow;
    int i, tam, *tabulIn, *tabulOut;
    double t0, t1, t2, t3;
    int local_start, local_end, rows_per_process, remainder;
    int result_local, result_global;
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (pow = POWMIN; pow <= POWMAX; pow++) {
        tam = 1 << pow;

        // Calculo da distribuicao de trabalho
        rows_per_process = tam / size;
        remainder = tam % size;

        // Definir inicio e fim das linhas para cada processo
        if (rank < remainder) {
            local_start = rank * (rows_per_process + 1) + 1;
            local_end = local_start + rows_per_process;
        }
        else {
            local_start = rank * rows_per_process + remainder + 1;
            local_end = local_start + rows_per_process - 1;
        }

        // Alocacao e inicializacao dos tabuleiros
        if (rank == 0) {
            t0 = wall_time();
        }

        tabulIn = (int *)malloc((tam + 2) * (tam + 2) * sizeof(int));
        tabulOut = (int *)malloc((tam + 2) * (tam + 2) * sizeof(int));
        InitTabul(tabulIn, tabulOut, tam, rank);

        // Broadcast do tabuleiro inicial para todos os processos
        MPI_Bcast(tabulIn, (tam + 2) * (tam + 2), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(tabulOut, (tam + 2) * (tam + 2), MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            t1 = wall_time();
        }

        // Loop principal de evolucao
        for (i = 0; i < 2 * (tam - 3); i++) {
            // Primeira evolucao: tabulIn -> tabulOut
            UmaVidaMPI(tabulIn, tabulOut, tam, local_start, local_end);

            // Sincronizacao: gather de todas as partes
            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                          tabulOut, (tam + 2) * (tam + 2) / size + (tam + 2) * 2, MPI_INT, MPI_COMM_WORLD);

            // Segunda evolucao: tabulOut -> tabulIn
            UmaVidaMPI(tabulOut, tabulIn, tam, local_start, local_end);

            // Sincronizacao: gather de todas as partes
            MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                          tabulIn, (tam + 2) * (tam + 2) / size + (tam + 2) * 2, MPI_INT, MPI_COMM_WORLD);
        }

        if (rank == 0) {
            t2 = wall_time();
        }

        // Verificacao do resultado
        result_local = Correto(tabulIn, tam);
        MPI_Allreduce(&result_local, &result_global, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

        if (rank == 0) {
            if (result_global) {
                printf("**RESULTADO CORRETO**\n");
            }
            else {
                printf("**RESULTADO ERRADO**\n");
            }
            t3 = wall_time();
            printf("tam=%d; processos=%d; tempos: init=%7.7f, comp=%7.7f, fim=%7.7f, tot=%7.7f \n",
                   tam, size, t1 - t0, t2 - t1, t3 - t2, t3 - t0);
        }

        free(tabulIn);
        free(tabulOut);
    }

    MPI_Finalize();
    return 0;
}