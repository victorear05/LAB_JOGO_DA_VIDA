# Makefile para Laboratório Jogo da Vida Paralelo
# Autor: Laboratório PSPD
# Data: $(date)

# Compiladores
CC = gcc
MPICC = mpicc
NVCC = nvcc

# Flags de compilação
CFLAGS = -O3 -Wall
OMPFLAGS = -fopenmp
MPICFLAGS = -O3
CUDAFLAGS = -O3 -arch=sm_35
OMPGPUFLAGS = -fopenmp -foffload=nvptx-none -foffload=-lm -fno-stack-protector -fcf-protection=none

# Nomes dos executáveis
TARGETS = jogodavida jogodavidampi jogodavidaomp jogodavida_cuda jogodavidaomp_gpu

# Regra principal
all: $(TARGETS)

# Versão sequencial original
jogodavida: src/jogodavida.c
	$(CC) $(CFLAGS) -o compilados/$@ $<

# Versão MPI
jogodavidampi: src/jogodavidampi.c
	$(MPICC) $(MPICFLAGS) -o compilados/$@ $<

# Versão OpenMP
jogodavidaomp: src/jogodavidaomp.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -o compilados/$@ $<

# Versão CUDA
jogodavida_cuda: src/jogodavida.cu
	$(NVCC) $(CUDAFLAGS) -o compilados/$@ $<

# Versão OpenMP GPU
jogodavidaomp_gpu: src/jogodavidaomp_gpu.c
	$(CC) $(CFLAGS) $(OMPGPUFLAGS) -o compilados/$@ $<

# Limpeza
clean:
	rm -f compilados/

# Regras para teste
test: all
	@echo "=== Testando versão sequencial ==="
	./jogodavida
	@echo "\n=== Testando versão OpenMP ==="
	export OMP_NUM_THREADS=4 && ./jogodavidaomp
	@echo "\n=== Testando versão MPI ==="
	mpirun -np 4 ./jogodavidampi
	@echo "\n=== Testando versão CUDA ==="
	./jogodavida_cuda
	@echo "\n=== Testando versão OpenMP GPU ==="
	./jogodavidaomp_gpu

# Regras para benchmark
benchmark: all
	@echo "Executando benchmark completo..."
	./run_benchmark.sh

.PHONY: all clean test benchmark