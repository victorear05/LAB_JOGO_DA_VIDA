#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

#define ind2d(i, j) (i) * (tam + 2) + j
#define POWMIN 3
#define POWMAX 10

double wall_time(void) {
    struct timeval tv;
    struct timezone tz;

    gettimeofday(&tv, &tz);
    return (tv.tv_sec + tv.tv_usec / 1000000.0);
}

/* Funcao para aplicar as regras do Jogo da Vida com OpenMP GPU offloading */
void UmaVidaOMPGPU(int *tabulIn, int *tabulOut, int tam) {
    int total_cells = (tam + 2) * (tam + 2);

    // Offloading para GPU com OpenMP target
    #pragma omp target teams distribute parallel for collapse(2)                    \
    map(to : tabulIn[0 : total_cells])                                              \
    map(from : tabulOut[0 : total_cells])
    for (int i = 1; i <= tam; i++) {
        for (int j = 1; j <= tam; j++) {
            int vizviv = tabulIn[ind2d(i - 1, j - 1)] + tabulIn[ind2d(i - 1, j)] +
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

/* Versao alternativa com gestao explicita de dados */
void UmaVidaOMPGPU_Managed(int *tabulIn, int *tabulOut, int tam) {
    int total_cells = (tam + 2) * (tam + 2);

    // Mapeamento de dados mais explicito
    #pragma omp target data map(to : tabulIn[0 : total_cells]) map(from : tabulOut[0 : total_cells])
    {
        #pragma omp target teams distribute parallel for collapse(2)
        for (int i = 1; i <= tam; i++) {
            for (int j = 1; j <= tam; j++) {
                int idx = ind2d(i, j);
                int vizviv = tabulIn[ind2d(i - 1, j - 1)] + tabulIn[ind2d(i - 1, j)] +
                             tabulIn[ind2d(i - 1, j + 1)] + tabulIn[ind2d(i, j - 1)] +
                             tabulIn[ind2d(i, j + 1)] + tabulIn[ind2d(i + 1, j - 1)] +
                             tabulIn[ind2d(i + 1, j)] + tabulIn[ind2d(i + 1, j + 1)];

                if (tabulIn[idx] && vizviv < 2)
                    tabulOut[idx] = 0;
                else if (tabulIn[idx] && vizviv > 3)
                    tabulOut[idx] = 0;
                else if (!tabulIn[idx] && vizviv == 3)
                    tabulOut[idx] = 1;
                else
                    tabulOut[idx] = tabulIn[idx];
            }
        }
    }
}

void InitTabul(int *tabulIn, int *tabulOut, int tam) {
    int ij;
    int total_cells = (tam + 2) * (tam + 2);

    #pragma omp parallel for
    for (ij = 0; ij < total_cells; ij++) {
        tabulIn[ij] = 0;
        tabulOut[ij] = 0;
    }

    tabulIn[ind2d(1, 2)] = 1;
    tabulIn[ind2d(2, 3)] = 1;
    tabulIn[ind2d(3, 1)] = 1;
    tabulIn[ind2d(3, 2)] = 1;
    tabulIn[ind2d(3, 3)] = 1;
}

int Correto(int *tabul, int tam) {
    int ij, cnt = 0;
    int total_cells = (tam + 2) * (tam + 2);

    #pragma omp parallel for reduction(+ : cnt)
    for (ij = 0; ij < total_cells; ij++) {
        cnt += tabul[ij];
    }

    return (cnt == 5 && tabul[ind2d(tam - 2, tam - 1)] &&
            tabul[ind2d(tam - 1, tam)] && tabul[ind2d(tam, tam - 2)] &&
            tabul[ind2d(tam, tam - 1)] && tabul[ind2d(tam, tam)]);
}

void check_gpu_devices() {
    int num_devices = omp_get_num_devices();
    printf("Numero de dispositivos OpenMP detectados: %d\n", num_devices);

    if (num_devices > 0) {
        printf("Dispositivos disponiveis:\n");
        for (int i = 0; i < num_devices; i++) {
            printf("  Dispositivo %d: ", i);
            #pragma omp target device(i) 
            {
                #pragma omp teams
                {
                    if (omp_get_team_num() == 0) {
                        printf("GPU ativa\n");
                    }
                }
            }
        }
    }
    else {
        printf("AVISO: Nenhum dispositivo GPU encontrado. Executando na CPU.\n");
    }
}

int main(void) {
    int pow;
    int i, tam, *tabulIn, *tabulOut;
    double t0, t1, t2, t3;
    int num_threads, num_devices;

    // Verificar dispositivos GPU disponiveis
    check_gpu_devices();

    // Configuracao OpenMP
    num_threads = omp_get_max_threads();
    num_devices = omp_get_num_devices();

    printf("Executando OpenMP GPU Offloading\n");
    printf("Threads CPU: %d\n", num_threads);
    printf("Dispositivos GPU: %d\n", num_devices);

    // Configurar dispositivo default (se disponivel)
    if (num_devices > 0) {
        omp_set_default_device(0);
        printf("Usando dispositivo GPU 0 como default\n");
    }

    // Loop para todos os tamanhos do tabuleiro
    for (pow = POWMIN; pow <= POWMAX; pow++) {
        tam = 1 << pow;
        int total_size = (tam + 2) * (tam + 2) * sizeof(int);

        printf("\n--- Processando tabuleiro %dx%d ---\n", tam, tam);

        // Alocacao e inicializacao dos tabuleiros
        t0 = wall_time();
        tabulIn = (int *)malloc(total_size);
        tabulOut = (int *)malloc(total_size);

        if (!tabulIn || !tabulOut)
        {
            printf("Erro na alocacao de memoria!\n");
            exit(1);
        }

        InitTabul(tabulIn, tabulOut, tam);
        t1 = wall_time();

        // Usar a versao com gestao de dados se GPU disponivel, senao usar versao simples
        if (num_devices > 0) {
            printf("Executando na GPU com gestao explicita de dados\n");

            // Loop principal de evolucao com gestao otimizada de dados
            #pragma omp target data map(tofrom : tabulIn[0 : (tam + 2) * (tam + 2)]) map(alloc : tabulOut[0 : (tam + 2) * (tam + 2)])
            {
                for (i = 0; i < 2 * (tam - 3); i++) {
                    // Primeira evolucao: tabulIn -> tabulOut
                    #pragma omp target teams distribute parallel for collapse(2)
                    for (int ii = 1; ii <= tam; ii++) {
                        for (int jj = 1; jj <= tam; jj++) {
                            int idx = ind2d(ii, jj);
                            int vizviv = tabulIn[ind2d(ii - 1, jj - 1)] + tabulIn[ind2d(ii - 1, jj)] +
                                         tabulIn[ind2d(ii - 1, jj + 1)] + tabulIn[ind2d(ii, jj - 1)] +
                                         tabulIn[ind2d(ii, jj + 1)] + tabulIn[ind2d(ii + 1, jj - 1)] +
                                         tabulIn[ind2d(ii + 1, jj)] + tabulIn[ind2d(ii + 1, jj + 1)];

                            if (tabulIn[idx] && vizviv < 2)
                                tabulOut[idx] = 0;
                            else if (tabulIn[idx] && vizviv > 3)
                                tabulOut[idx] = 0;
                            else if (!tabulIn[idx] && vizviv == 3)
                                tabulOut[idx] = 1;
                            else
                                tabulOut[idx] = tabulIn[idx];
                        }
                    }

                    // Segunda evolucao: tabulOut -> tabulIn
                    #pragma omp target teams distribute parallel for collapse(2)
                    for (int ii = 1; ii <= tam; ii++) {
                        for (int jj = 1; jj <= tam; jj++) {
                            int idx = ind2d(ii, jj);
                            int vizviv = tabulOut[ind2d(ii - 1, jj - 1)] + tabulOut[ind2d(ii - 1, jj)] +
                                         tabulOut[ind2d(ii - 1, jj + 1)] + tabulOut[ind2d(ii, jj - 1)] +
                                         tabulOut[ind2d(ii, jj + 1)] + tabulOut[ind2d(ii + 1, jj - 1)] +
                                         tabulOut[ind2d(ii + 1, jj)] + tabulOut[ind2d(ii + 1, jj + 1)];

                            if (tabulOut[idx] && vizviv < 2) {
                                tabulIn[idx] = 0;
                            }
                            else if (tabulOut[idx] && vizviv > 3) {
                                tabulIn[idx] = 0;
                            }
                            else if (!tabulOut[idx] && vizviv == 3) {
                                tabulIn[idx] = 1;
                            }
                            else {
                                tabulIn[idx] = tabulOut[idx];
                            }
                        }
                    }
                }
            }
        }
        else {
            printf("Executando na CPU (fallback)\n");
            // Fallback para CPU se GPU nao disponivel
            for (i = 0; i < 2 * (tam - 3); i++) {
                UmaVidaOMPGPU(tabulIn, tabulOut, tam);
                UmaVidaOMPGPU(tabulOut, tabulIn, tam);
            }
        }

        t2 = wall_time();

        if (Correto(tabulIn, tam)) {
            printf("**RESULTADO CORRETO**\n");
        }
        else {
            printf("**RESULTADO ERRADO**\n");
        }

        t3 = wall_time();
        printf("tam=%d; dispositivos=%d; tempos: init=%7.7f, comp=%7.7f, fim=%7.7f, tot=%7.7f \n", tam, num_devices, t1 - t0, t2 - t1, t3 - t2, t3 - t0);

        free(tabulIn);
        free(tabulOut);
    }

    return 0;
}