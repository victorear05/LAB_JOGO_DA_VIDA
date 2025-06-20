#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <string.h>

#define ind2d(i, j) (i) * (tam + 2) + j
#define POWMIN 3
#define POWMAX 10


double wall_time(void) {
    struct timeval tv;
    struct timezone tz;

    gettimeofday(&tv, &tz);
    return (tv.tv_sec + tv.tv_usec / 1000000.0);
}

void UmaVidaMPI_Distribuido(int *tabulIn, int *tabulOut, int tam, int start_row, int end_row) {
    int i, j, vizviv;

    for (i = start_row; i <= end_row; i++) {
        for (j = 1; j <= tam; j++) {
            vizviv = tabulIn[ind2d(i - 1, j - 1)] + tabulIn[ind2d(i - 1, j)] +
                     tabulIn[ind2d(i - 1, j + 1)] + tabulIn[ind2d(i, j - 1)] +
                     tabulIn[ind2d(i, j + 1)] + tabulIn[ind2d(i + 1, j - 1)] +
                     tabulIn[ind2d(i + 1, j)] + tabulIn[ind2d(i + 1, j + 1)];

            if (tabulIn[ind2d(i, j)] && vizviv < 2) {
                tabulOut[ind2d(i, j)] = 0;
            }
            else if (tabulIn[ind2d(i, j)] && vizviv > 3) {
                tabulOut[ind2d(i, j)] = 0;
            }
            else if (!tabulIn[ind2d(i, j)] && vizviv == 3) {
                tabulOut[ind2d(i, j)] = 1;
            }
            else {
                tabulOut[ind2d(i, j)] = tabulIn[ind2d(i, j)];
            }
        }
    }
}

/* Funcao para dump do tabuleiro - apenas o processo 0 imprime */
void DumpTabul(int *tabul, int tam, int first, int last, char *msg, int rank) {
    int i, ij;

    if (rank != 0) {
        return; // Apenas processo 0 imprime
    }

    printf("%s; Dump posicoes [%d:%d, %d:%d] de tabuleiro %d x %d\n", 
           msg, first, last, first, last, tam, tam);

    for (i = first; i <= last; i++) {
        printf("=");
    }
    printf("=\n");

    for (i = ind2d(first, 0); i <= ind2d(last, 0); i += ind2d(1, 0)) {
        for (ij = i + first; ij <= i + last; ij++)
            printf("%c", tabul[ij] ? 'X' : '.');
        printf("\n");
    }

    for (i = first; i <= last; i++) {
        printf("=");
    }
    printf("=\n");
}

/* Inicializacao do tabuleiro */
void InitTabul(int *tabulIn, int *tabulOut, int tam) {
    int ij;

    for (ij = 0; ij < (tam + 2) * (tam + 2); ij++) {
        tabulIn[ij] = 0;
        tabulOut[ij] = 0;
    }

    // Configuracao inicial do veleiro no canto superior esquerdo
    tabulIn[ind2d(1, 2)] = 1;
    tabulIn[ind2d(2, 3)] = 1;
    tabulIn[ind2d(3, 1)] = 1;
    tabulIn[ind2d(3, 2)] = 1;
    tabulIn[ind2d(3, 3)] = 1;
}

/* Verificacao se o resultado esta correto */
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
    int rank, size;
    int pow;
    int i, tam, *tabulIn, *tabulOut, *temp_tabul;
    double t0, t1, t2, t3;
    int local_start, local_end, rows_per_process;
    int *recvcounts, *displs;

    // Inicializacao MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Alocar arrays para gather
    recvcounts = (int *)malloc(size * sizeof(int));
    displs = (int *)malloc(size * sizeof(int));

    // Loop para todos os tamanhos do tabuleiro
    for (pow = POWMIN; pow <= POWMAX; pow++) {
        tam = 1 << pow;

        if (rank == 0) {
            t0 = wall_time();
        }

        // Alocacao dos tabuleiros
        tabulIn = (int *)calloc((tam + 2) * (tam + 2), sizeof(int));
        tabulOut = (int *)calloc((tam + 2) * (tam + 2), sizeof(int));
        temp_tabul = (int *)calloc((tam + 2) * (tam + 2), sizeof(int));

        if (!tabulIn || !tabulOut || !temp_tabul) {
            if (rank == 0) {
                printf("Erro de alocacao de memoria\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Inicializar apenas no processo 0
        if (rank == 0) {
            InitTabul(tabulIn, tabulOut, tam);
        }

        // Broadcast do tabuleiro inicial
        MPI_Bcast(tabulIn, (tam + 2) * (tam + 2), MPI_INT, 0, MPI_COMM_WORLD);

        // Mostrar estado inicial
        DumpTabul(tabulIn, tam, 1, tam, "Estado Inicial - Veleiro no canto superior esquerdo", rank);

        if (rank == 0) {
            t1 = wall_time();
        }

        // Calcular divisao de trabalho
        rows_per_process = tam / size;
        int remainder = tam % size;

        // Configurar arrays para gather
        for (i = 0; i < size; i++) {
            if (i < remainder) {
                recvcounts[i] = (rows_per_process + 1) * (tam + 2);
                displs[i] = i * (rows_per_process + 1) * (tam + 2) + (tam + 2);
            }
            else {
                recvcounts[i] = rows_per_process * (tam + 2);
                displs[i] = (remainder * (rows_per_process + 1) + (i - remainder) * rows_per_process) * (tam + 2) + (tam + 2);
            }
        }

        // Definir range de trabalho para este processo
        if (rank < remainder) {
            local_start = rank * (rows_per_process + 1) + 1;
            local_end = local_start + rows_per_process;
        }
        else {
            local_start = remainder * (rows_per_process + 1) + (rank - remainder) * rows_per_process + 1;
            local_end = local_start + rows_per_process - 1;
        }

        // Garantir limites validos
        if (local_start > tam) {
            local_start = tam + 1;
        }
        if (local_end > tam) {
            local_end = tam;
        }
        if (local_start > local_end) {
            local_start = local_end = 0; // Sem trabalho
        }

        // Loop principal de evolucao
        for (i = 0; i < 2 * (tam - 3); i++) {
            // Primeira evolucao: tabulIn -> temp_tabul
            memcpy(temp_tabul, tabulIn, (tam + 2) * (tam + 2) * sizeof(int));

            if (local_start <= local_end && local_start > 0) {
                UmaVidaMPI_Distribuido(tabulIn, temp_tabul, tam, local_start, local_end);
            }

            // Gather dos resultados parciais
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, temp_tabul, recvcounts, displs, MPI_INT, MPI_COMM_WORLD);

            // Segunda evolucao: temp_tabul -> tabulIn
            memcpy(tabulIn, temp_tabul, (tam + 2) * (tam + 2) * sizeof(int));

            if (local_start <= local_end && local_start > 0) {
                UmaVidaMPI_Distribuido(temp_tabul, tabulIn, tam, local_start, local_end);
            }

            // Gather dos resultados parciais
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, tabulIn, recvcounts, displs, MPI_INT, MPI_COMM_WORLD);
        }

        if (rank == 0) {
            t2 = wall_time();
        }

        // Mostrar estado final
        DumpTabul(tabulIn, tam, 1, tam, "Estado Final - Veleiro no canto inferior direito", rank);

        // Verificacao do resultado apenas no processo 0
        if (rank == 0) {
            if (Correto(tabulIn, tam)) {
                printf("**RESULTADO CORRETO**\n");
            }
            else {
                printf("**RESULTADO ERRADO**\n");
            }

            t3 = wall_time();
            printf("tam=%d; processos=%d; tempos: init=%7.7f, comp=%7.7f, fim=%7.7f, tot=%7.7f \n", tam, size, t1 - t0, t2 - t1, t3 - t2, t3 - t0);
        }

        free(tabulIn);
        free(tabulOut);
        free(temp_tabul);
    }

    free(recvcounts);
    free(displs);

    MPI_Finalize();

    return 0;
}