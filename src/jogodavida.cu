#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define ind2d(i, j) (i) * (tam + 2) + j
#define POWMIN 3
#define POWMAX 10

// Macro para verificacao de erros CUDA
#define CUDA_CHECK(call)                                             \
    do                                                               \
    {                                                                \
        cudaError_t err = call;                                      \
        if (err != cudaSuccess)                                      \
        {                                                            \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err));                         \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

double wall_time(void)
{
    struct timeval tv;
    struct timezone tz;

    gettimeofday(&tv, &tz);
    return (tv.tv_sec + tv.tv_usec / 1000000.0);
}

/* Kernel CUDA para aplicar as regras do Jogo da Vida */
__global__ void UmaVidaCUDA(int *tabulIn, int *tabulOut, int tam)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1; // +1 para pular a borda
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1; // +1 para pular a borda

    // Verificar se estamos dentro dos limites validos
    if (i <= tam && j <= tam)
    {
        int vizviv = tabulIn[ind2d(i - 1, j - 1)] + tabulIn[ind2d(i - 1, j)] +
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

/* Kernel otimizado com memoria compartilhada */
__global__ void UmaVidaCUDA_Shared(int *tabulIn, int *tabulOut, int tam)
{
    // Definir memoria compartilhada com halo
    __shared__ int shared_tabul[18][18]; // 16x16 + 2 de halo em cada direcao

    int global_i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int global_j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int local_i = threadIdx.y + 1;
    int local_j = threadIdx.x + 1;

    // Carregar dados para memoria compartilhada
    if (global_i <= tam + 1 && global_j <= tam + 1)
    {
        shared_tabul[local_i][local_j] = tabulIn[ind2d(global_i, global_j)];

        // Carregar bordas
        if (threadIdx.y == 0 && global_i > 0)
        {
            shared_tabul[0][local_j] = tabulIn[ind2d(global_i - 1, global_j)];
        }
        if (threadIdx.y == blockDim.y - 1 && global_i < tam + 1)
        {
            shared_tabul[local_i + 1][local_j] = tabulIn[ind2d(global_i + 1, global_j)];
        }
        if (threadIdx.x == 0 && global_j > 0)
        {
            shared_tabul[local_i][0] = tabulIn[ind2d(global_i, global_j - 1)];
        }
        if (threadIdx.x == blockDim.x - 1 && global_j < tam + 1)
        {
            shared_tabul[local_i][local_j + 1] = tabulIn[ind2d(global_i, global_j + 1)];
        }
    }

    __syncthreads();

    // Processar apenas celulas validas
    if (global_i <= tam && global_j <= tam)
    {
        int vizviv = shared_tabul[local_i - 1][local_j - 1] + shared_tabul[local_i - 1][local_j] +
                     shared_tabul[local_i - 1][local_j + 1] + shared_tabul[local_i][local_j - 1] +
                     shared_tabul[local_i][local_j + 1] + shared_tabul[local_i + 1][local_j - 1] +
                     shared_tabul[local_i + 1][local_j] + shared_tabul[local_i + 1][local_j + 1];

        int current = shared_tabul[local_i][local_j];

        if (current && vizviv < 2)
            tabulOut[ind2d(global_i, global_j)] = 0;
        else if (current && vizviv > 3)
            tabulOut[ind2d(global_i, global_j)] = 0;
        else if (!current && vizviv == 3)
            tabulOut[ind2d(global_i, global_j)] = 1;
        else
            tabulOut[ind2d(global_i, global_j)] = current;
    }
}

void DumpTabul(int *tabul, int tam, int first, int last, char *msg)
{
    int i, ij;

    printf("%s; Dump posicoes [%d:%d, %d:%d] de tabuleiro %d x %d\n",
           msg, first, last, first, last, tam, tam);
    for (i = first; i <= last; i++)
        printf("=");
    printf("=\n");
    for (i = ind2d(first, 0); i <= ind2d(last, 0); i += ind2d(1, 0))
    {
        for (ij = i + first; ij <= i + last; ij++)
            printf("%c", tabul[ij] ? 'X' : '.');
        printf("\n");
    }
    for (i = first; i <= last; i++)
        printf("=");
    printf("=\n");
}

void InitTabul(int *tabulIn, int *tabulOut, int tam)
{
    int ij;

    for (ij = 0; ij < (tam + 2) * (tam + 2); ij++)
    {
        tabulIn[ij] = 0;
        tabulOut[ij] = 0;
    }

    tabulIn[ind2d(1, 2)] = 1;
    tabulIn[ind2d(2, 3)] = 1;
    tabulIn[ind2d(3, 1)] = 1;
    tabulIn[ind2d(3, 2)] = 1;
    tabulIn[ind2d(3, 3)] = 1;
}

int Correto(int *tabul, int tam)
{
    int ij, cnt;

    cnt = 0;
    for (ij = 0; ij < (tam + 2) * (tam + 2); ij++)
        cnt = cnt + tabul[ij];
    return (cnt == 5 && tabul[ind2d(tam - 2, tam - 1)] &&
            tabul[ind2d(tam - 1, tam)] && tabul[ind2d(tam, tam - 2)] &&
            tabul[ind2d(tam, tam - 1)] && tabul[ind2d(tam, tam)]);
}

int main(void)
{
    int pow;
    int i, tam, *h_tabulIn, *h_tabulOut; // Host arrays
    int *d_tabulIn, *d_tabulOut;         // Device arrays
    double t0, t1, t2, t3;

    // Informacoes da GPU
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        printf("Nenhuma GPU CUDA encontrada!\n");
        return -1;
    }

    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Usando GPU: %s\n", deviceProp.name);
    printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Memoria Global: %lu bytes\n", deviceProp.totalGlobalMem);

    // Loop para todos os tamanhos do tabuleiro
    for (pow = POWMIN; pow <= POWMAX; pow++)
    {
        tam = 1 << pow;
        int total_size = (tam + 2) * (tam + 2) * sizeof(int);

        printf("\n--- Processando tabuleiro %dx%d ---\n", tam, tam);

        // Alocacao na CPU
        t0 = wall_time();
        h_tabulIn = (int *)malloc(total_size);
        h_tabulOut = (int *)malloc(total_size);

        // Alocacao na GPU
        CUDA_CHECK(cudaMalloc((void **)&d_tabulIn, total_size));
        CUDA_CHECK(cudaMalloc((void **)&d_tabulOut, total_size));

        // Inicializacao
        InitTabul(h_tabulIn, h_tabulOut, tam);

        // Copia inicial para GPU
        CUDA_CHECK(cudaMemcpy(d_tabulIn, h_tabulIn, total_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tabulOut, h_tabulOut, total_size, cudaMemcpyHostToDevice));

        t1 = wall_time();

        // Configuracao dos blocos e threads
        dim3 blockSize(16, 16);
        dim3 gridSize((tam + blockSize.x - 1) / blockSize.x,
                      (tam + blockSize.y - 1) / blockSize.y);

        printf("Grid: %dx%d, Block: %dx%d\n", gridSize.x, gridSize.y, blockSize.x, blockSize.y);

        // Loop principal de evolucao
        for (i = 0; i < 2 * (tam - 3); i++)
        {
            // Primeira evolucao: d_tabulIn -> d_tabulOut
            UmaVidaCUDA_Shared<<<gridSize, blockSize>>>(d_tabulIn, d_tabulOut, tam);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Segunda evolucao: d_tabulOut -> d_tabulIn
            UmaVidaCUDA_Shared<<<gridSize, blockSize>>>(d_tabulOut, d_tabulIn, tam);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Copia resultado de volta para CPU
        CUDA_CHECK(cudaMemcpy(h_tabulIn, d_tabulIn, total_size, cudaMemcpyDeviceToHost));

        t2 = wall_time();

        // Verificacao do resultado
        if (Correto(h_tabulIn, tam))
            printf("**RESULTADO CORRETO**\n");
        else
            printf("**RESULTADO ERRADO**\n");

        t3 = wall_time();
        printf("tam=%d; tempos: init=%7.7f, comp=%7.7f, fim=%7.7f, tot=%7.7f \n",
               tam, t1 - t0, t2 - t1, t3 - t2, t3 - t0);

        // Limpeza de memoria
        free(h_tabulIn);
        free(h_tabulOut);
        CUDA_CHECK(cudaFree(d_tabulIn));
        CUDA_CHECK(cudaFree(d_tabulOut));
    }

    return 0;
}