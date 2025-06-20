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
all: exec $(TARGETS)

# Criar pasta com os executáveis
exec:
	mkdir -p exec

# Versão sequencial original
jogodavida: src/jogodavida.c | exec
	$(CC) $(CFLAGS) -o exec/$@ $<

# Versão MPI
jogodavidampi: src/jogodavidampi.c | exec
	$(MPICC) $(MPICFLAGS) -o exec/$@ $<

# Versão OpenMP
jogodavidaomp: src/jogodavidaomp.c | exec
	$(CC) $(CFLAGS) $(OMPFLAGS) -o exec/$@ $<

# Versão CUDA
jogodavida_cuda: src/jogodavida.cu | exec
	$(NVCC) $(CUDAFLAGS) -o exec/$@ $<

# Versão OpenMP GPU
jogodavidaomp_gpu: src/jogodavidaomp_gpu.c | exec
	$(CC) $(CFLAGS) $(OMPGPUFLAGS) -o exec/$@ $<

# Limpeza
clean:
	rm -rf exec/ results/

# Regras para teste
test: all
	@echo "=== Versão sequencial ==="
	./jogodavida
	@echo "\n=== Versão OpenMP ==="
	export OMP_NUM_THREADS=4 && ./jogodavidaomp
	@echo "\n=== Versão MPI ==="
	mpirun -np 4 ./jogodavidampi
	@echo "\n=== Versão CUDA ==="
	./jogodavida_cuda
	@echo "\n=== Versão OpenMP GPU ==="
	./jogodavidaomp_gpu

# Regras para benchmark
benchmark: all
	@echo "Executando benchmark completo..."
	./run_benchmark.sh

.PHONY: all clean test benchmark