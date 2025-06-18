# Laboratório Jogo da Vida Paralelo

## 📋 Visão Geral

Este projeto implementa diferentes versões paralelas do **Jogo da Vida de Conway** para comparação de performance entre paradigmas de programação paralela:

- **MPI** (Message Passing Interface) - memória distribuída
- **OpenMP** - memória compartilhada
- **CUDA** - computação em GPU NVIDIA
- **OpenMP GPU** - offloading para GPU com OpenMP

## 🎯 Objetivos

- Experimentar diferentes paradigmas de programação paralela
- Comparar performance entre as implementações
- Analisar escalabilidade e eficiência
- Validar a movimentação correta do "veleiro" no tabuleiro

## 🏗️ Estrutura do Projeto

```
jogodavida-paralelo/
├── src/
│   ├── jogodavida.c          # Versão sequencial original
│   ├── jogodavidampi.c       # Versão MPI
│   ├── jogodavidaomp.c       # Versão OpenMP
│   ├── jogodavida.cu         # Versão CUDA
│   └── jogodavidaomp_gpu.c   # Versão OpenMP GPU
├── Makefile                  # Makefile
├── run_benchmark.sh          # Script de benchmark automatizado
├── results/                  # Resultados dos experimentos
└── README.md                 # Esta documentação
```

## 🔧 Compilação

### Pré-requisitos

- **GCC** 7.0+ com suporte OpenMP
- **Open MPI** ou Intel MPI
- **CUDA Toolkit** 10.0+ (para versão CUDA)
- **OpenMP 4.5+** com suporte GPU (para versão OpenMP GPU)

### Verificar Dependências

```bash
make check-deps
```

### Compilar Todas as Versões

```bash
make all
```

### Compilar Versões Específicas

```bash
# Versão sequencial
make jogodavida

# Versão MPI
make jogodavidampi

# Versão OpenMP
make jogodavidaomp

# Versão CUDA
make jogodavida_cuda

# Versão OpenMP GPU
make jogodavidaomp_gpu
```

## 🚀 Execução

### Testes Básicos

```bash
# Executar todos os testes básicos
make test

# Versão sequencial
./jogodavida

# Versão OpenMP (definir número de threads)
export OMP_NUM_THREADS=4
./jogodavidaomp

# Versão MPI (definir número de processos)
mpirun -np 4 ./jogodavidampi

# Versão CUDA
./jogodavida_cuda

# Versão OpenMP GPU
./jogodavidaomp_gpu
```

### Benchmark Automatizado

```bash
# Executar benchmark completo
make benchmark
# ou
./run_benchmark.sh
```

## 📊 Análise de Resultados

### Métricas Coletadas

- **Tempo de inicialização** (`t_init`)
- **Tempo de computação** (`t_comp`) 
- **Tempo de finalização** (`t_fim`)
- **Tempo total** (`t_total`)

### Cálculo de Performance

```
Speedup = T_sequencial / T_paralelo
Eficiência = Speedup / Número_de_Processadores
Throughput = Células_processadas / Tempo_computação
```

### Validação

Todas as versões devem:
- ✅ Produzir resultado "**RESULTADO CORRETO**"
- ✅ Mover o veleiro do canto superior esquerdo ao inferior direito
- ✅ Manter exatamente 5 células vivas ao final

## 🔍 Detalhes das Implementações

### 1. Versão MPI (`jogodavidampi.c`)

**Estratégia**: Divisão horizontal do tabuleiro entre processos

**Características**:
- Processo 0 atua como coordenador
- Distribuição equilibrada de linhas entre workers
- Comunicação coletiva (`MPI_Bcast`, `MPI_Allgather`)
- Sincronização a cada geração

**Comando de execução**:
```bash
mpirun -np <num_processos> ./jogodavidampi
```

### 2. Versão OpenMP (`jogodavidaomp.c`)

**Estratégia**: Paralelização de loops com threads

**Características**:
- `#pragma omp parallel for` nos loops principais
- Scheduling estático para balanceamento
- Redução paralela na verificação
- Inicialização paralela

**Variáveis de ambiente**:
```bash
export OMP_NUM_THREADS=<num_threads>
export OMP_SCHEDULE=static
```

### 3. Versão CUDA (`jogodavida.cu`)

**Estratégia**: Computação massivamente paralela em GPU

**Características**:
- Kernel otimizado com memória compartilhada
- Grid 2D de blocos e threads
- Gestão explícita de memória GPU
- Dois kernels: básico e otimizado

**Configuração de kernel**:
```cpp
dim3 blockSize(16, 16);           // 256 threads por bloco
dim3 gridSize((tam+15)/16, (tam+15)/16);  // Cobertura do tabuleiro
```

### 4. Versão OpenMP GPU (`jogodavidaomp_gpu.c`)

**Estratégia**: Offloading para GPU usando diretivas OpenMP

**Características**:
- `#pragma omp target` para offloading
- Gestão automática e explícita de dados
- Fallback para CPU se GPU indisponível
- Teams e distribute para paralelização GPU

**Diretivas principais**:
```cpp
#pragma omp target teams distribute parallel for collapse(2) \
        map(to: tabulIn[0:total_cells]) \
        map(from: tabulOut[0:total_cells])
```

## 📈 Resultados Esperados

### Performance Típica

| Versão | Tabuleiro 64x64 | Tabuleiro 256x256 | Speedup |
|--------|-----------------|-------------------|---------|
| Sequencial | 0.0120s | 0.1850s | 1.0x |
| OpenMP (4 threads) | 0.0035s | 0.0520s | 3.4x |
| MPI (4 processos) | 0.0040s | 0.0580s | 3.2x |
| CUDA | 0.0015s | 0.0080s | 8.0x |
| OpenMP GPU | 0.0020s | 0.0120s | 6.2x |

*Valores ilustrativos - resultados variam conforme hardware*

### Escalabilidade

- **OpenMP**: Linear até número de cores físicos
- **MPI**: Boa escalabilidade em clusters
- **CUDA**: Excelente para tabuleiros grandes (>512x512)
- **OpenMP GPU**: Boa para tabuleiros médios-grandes

## 🐛 Troubleshooting

### Problemas Comuns

**1. Erro de compilação MPI**
```bash
# Verificar instalação MPI
which mpicc
mpicc --version

# Instalar se necessário (Ubuntu/Debian)
sudo apt-get install libopenmpi-dev openmpi-bin
```

**2. Erro CUDA "No CUDA-capable device"**
```bash
# Verificar GPU NVIDIA
nvidia-smi

# Verificar instalação CUDA
nvcc --version
```

**3. OpenMP GPU não funciona**
```bash
# Verificar suporte do compilador
gcc -fopenmp -foffload=nvptx-none --version

# Alternativa: usar Clang com OpenMP GPU
clang -fopenmp -fopenmp-targets=nvptx64 -O3 -o jogodavidaomp_gpu jogodavidaomp_gpu.c
```

**4. Resultado incorreto**
- Verificar se o tabuleiro tem bordas adequadas
- Conferir índices nos kernels/loops paralelos
- Validar comunicação entre processos (MPI)

### Logs de Debug

Habilitar logs detalhados:
```bash
# MPI
export OMPI_MCA_verbose=1
mpirun -np 4 ./jogodavidampi

# CUDA
export CUDA_LAUNCH_BLOCKING=1
./jogodavida_cuda

# OpenMP
export OMP_DISPLAY_ENV=true
./jogodavidaomp
```

## 📚 Referências

1. **Conway's Game of Life**: Gardner, M. "Mathematical Games", Scientific American 223, Oct 1970
2. **MPI Documentation**: [https://www.open-mpi.org/doc/](https://www.open-mpi.org/doc/)
3. **OpenMP Specification**: [https://www.openmp.org/specifications/](https://www.openmp.org/specifications/)
4. **CUDA Programming Guide**: [https://docs.nvidia.com/cuda/](https://docs.nvidia.com/cuda/)
5. **OpenMP GPU Offloading**: [https://www.openmp.org/updates/openmp-accelerator-support-gpus/](https://www.openmp.org/updates/openmp-accelerator-support-gpus/)

## 🤝 Contribuição

Este projeto é parte do laboratório de **Programação para Sistemas Paralelos e Distribuídos**.

### Estrutura do Relatório

O relatório deve incluir:

1. **Identificação**
   - Disciplina, turma, grupo e laboratório

2. **Códigos Comentados**
   - Listagem com explicação das paralelizações
   - Instruções de compilação e execução
   - Dificuldades encontradas e soluções

3. **Experimentação**
   - Cenários de teste executados
   - Tabela comparativa de tempos
   - Análise de speedup e eficiência

4. **Conclusões**
   - Percentual de ganho entre soluções
   - GPU mais eficiente no cluster
   - Recomendações de uso

### Critérios de Avaliação

- ✅ **Funcionalidade**: Todas as versões executam corretamente
- ✅ **Performance**: Speedup demonstrável nas versões paralelas
- ✅ **Documentação**: Código bem comentado e relatório completo
- ✅ **Experimentação**: Análise comparativa rigorosa
- ✅ **Apresentação**: Vídeo demonstrativo (4-6 min/aluno)

## 📞 Suporte

Para dúvidas sobre:
- **Compilação**: Verificar seção Troubleshooting
- **Execução**: Conferir comandos na seção Execução  
- **Resultados**: Analisar logs em `results/`
- **Relatório**: Seguir estrutura descrita acima

---

**Desenvolvido para**: Programação para Sistemas Paralelos e Distribuídos  
**Plataforma alvo**: Cluster chococino (164.41.20.252)  
**Linguagens**: C, CUDA  
**Paradigmas**: MPI, OpenMP, CUDA, OpenMP GPU