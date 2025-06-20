#!/bin/bash

# Script de Benchmark para Laboratório Jogo da Vida Paralelo
# Autor: Laboratório PSPD
# Data: $(date)

# Configurações
RESULTS_DIR="results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${RESULTS_DIR}/benchmark_${TIMESTAMP}.txt"

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Função para logging
log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1" | tee -a "$RESULTS_FILE"
}

error() {
    echo -e "${RED}[ERRO]${NC} $1" | tee -a "$RESULTS_FILE"
}

success() {
    echo -e "${GREEN}[OK]${NC} $1" | tee -a "$RESULTS_FILE"
}

warning() {
    echo -e "${YELLOW}[AVISO]${NC} $1" | tee -a "$RESULTS_FILE"
}

# Criar diretório de resultados
mkdir -p "$RESULTS_DIR"

# Cabeçalho do relatório
cat > "$RESULTS_FILE" << EOF
================================================================================
                    BENCHMARK JOGO DA VIDA PARALELO
================================================================================
Data/Hora: $(date)
Host: $(hostname)
Sistema: $(uname -a)
CPU: $(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
Memória: $(free -h | grep Mem | awk '{print $2}')
GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo "Não detectada")
================================================================================

EOF

log "Iniciando benchmark completo..."

# Verificar se os executáveis existem
EXECUTABLES=("jogodavida" "jogodavidaomp" "jogodavida_cuda" "jogodavidaomp_gpu")
MISSING_EXEC=()

for exec in "${EXECUTABLES[@]}"; do
    if [[ ! -f "./exec/$exec" ]]; then
        MISSING_EXEC+=("$exec")
    fi
done

if [[ ${#MISSING_EXEC[@]} -gt 0 ]]; then
    error "Executáveis não encontrados: ${MISSING_EXEC[*]}"
    log "Execute 'make all' antes de rodar o benchmark"
    exit 1
fi

# Configurações de teste
THREAD_COUNTS=(1 2 4 8 16)
MPI_PROCESSES=(1 2 4 8)
REPETITIONS=3

log "Configurações do benchmark:"
log "  - Threads OpenMP: ${THREAD_COUNTS[*]}"
log "  - Processos MPI: ${MPI_PROCESSES[*]}"
log "  - Repetições por teste: $REPETITIONS"

# Função para executar e coletar métricas
run_test() {
    local version="$1"
    local config="$2"
    local exec_cmd="$3"
    
    log "Executando: $version ($config)"
    
    echo "--- $version ($config) ---" >> "$RESULTS_FILE"
    
    local total_time=0
    local success_count=0
    
    for ((rep=1; rep<=REPETITIONS; rep++)); do
        log "  Repetição $rep/$REPETITIONS"
        
        # Executar comando e capturar saída
        if timeout 300s bash -c "$exec_cmd" >> "${RESULTS_FILE}_${version}_${config// /_}.log" 2>&1; then
            success_count=$((success_count + 1))
            log "    ✓ Sucesso"
        else
            warning "    ✗ Falha ou timeout"
        fi
    done
    
    echo "Sucessos: $success_count/$REPETITIONS" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    
    if [[ $success_count -eq $REPETITIONS ]]; then
        success "$version ($config): $success_count/$REPETITIONS sucessos"
    else
        warning "$version ($config): $success_count/$REPETITIONS sucessos"
    fi
}

# Teste 1: Versão Sequencial (baseline)
log "=== TESTE 1: Versão Sequencial ==="
run_test "Sequential" "baseline" "./exec/jogodavida"

# Teste 2: OpenMP com diferentes números de threads
log "=== TESTE 2: OpenMP ==="
for threads in "${THREAD_COUNTS[@]}"; do
    export OMP_NUM_THREADS=$threads
    run_test "OpenMP" "$threads threads" "./exec/jogodavidaomp"
done

# Teste 3: MPI com diferentes números de processos
log "=== TESTE 3: MPI ==="
for procs in "${MPI_PROCESSES[@]}"; do
    if command -v mpirun &> /dev/null; then
        run_test "MPI" "$procs processes" "mpirun -np $procs ./exec/jogodavidampi"
    else
        warning "MPI não disponível - pulando teste"
        break
    fi
done

# Teste 4: CUDA
log "=== TESTE 4: CUDA ==="
if command -v nvidia-smi &> /dev/null; then
    run_test "CUDA" "GPU" "./exec/jogodavida_cuda"
else
    warning "NVIDIA GPU não detectada - pulando teste CUDA"
fi

# Teste 5: OpenMP GPU
log "=== TESTE 5: OpenMP GPU ==="
run_test "OpenMP-GPU" "GPU offload" "./exec/jogodavidaomp_gpu"

# Análise de resultados
log "=== ANÁLISE DE RESULTADOS ==="

# Extrair tempos de execução e calcular speedups
python3 - << 'EOF' >> "$RESULTS_FILE" 2>/dev/null || log "Python3 não disponível para análise automática"
import re
import glob
import os

def extract_times(filename):
    times = []
    try:
        with open(filename, 'r') as f:
            content = f.read()
            # Buscar por padrão: tam=X; tempos: ... tot=Y.YYYYY
            matches = re.findall(r'tam=(\d+);.*?tot=([0-9.]+)', content)
            for tam, time in matches:
                times.append((int(tam), float(time)))
    except:
        pass
    return times

# Coletar todos os arquivos de log
log_files = glob.glob(f"${RESULTS_FILE}_*.log")
results = {}

for log_file in log_files:
    version = os.path.basename(log_file).replace(f"benchmark_${TIMESTAMP}.txt_", "").replace(".log", "")
    times = extract_times(log_file)
    if times:
        results[version] = times

print("\n=== RESUMO DE PERFORMANCE ===")
print(f"{'Versão':<20} {'Tam=8':<10} {'Tam=16':<10} {'Tam=32':<10} {'Tam=64':<10}")
print("-" * 70)

for version, times in results.items():
    row = f"{version:<20}"
    for tam_target in [8, 16, 32, 64]:
        time_found = None
        for tam, time in times:
            if tam == tam_target:
                time_found = f"{time:.4f}s"
                break
        row += f"{time_found or 'N/A':<10}"
    print(row)

# Calcular speedups se baseline disponível
if 'Sequential_baseline' in results:
    baseline_times = dict(results['Sequential_baseline'])
    print(f"\n=== SPEEDUPS (relativo ao sequencial) ===")
    print(f"{'Versão':<20} {'Tam=8':<10} {'Tam=16':<10} {'Tam=32':<10} {'Tam=64':<10}")
    print("-" * 70)
    
    for version, times in results.items():
        if version != 'Sequential_baseline':
            row = f"{version:<20}"
            for tam_target in [8, 16, 32, 64]:
                speedup = None
                if tam_target in baseline_times:
                    for tam, time in times:
                        if tam == tam_target and time > 0:
                            speedup = f"{baseline_times[tam_target]/time:.2f}x"
                            break
                row += f"{speedup or 'N/A':<10}"
            print(row)
EOF

# Resumo final
echo "" >> "$RESULTS_FILE"
echo "=================================================================================" >> "$RESULTS_FILE"
echo "BENCHMARK CONCLUÍDO" >> "$RESULTS_FILE"
echo "Arquivo completo: $RESULTS_FILE" >> "$RESULTS_FILE"
echo "Logs detalhados: ${RESULTS_FILE}_*.log" >> "$RESULTS_FILE"
echo "=================================================================================" >> "$RESULTS_FILE"

success "Benchmark concluído!"
log "Resultados salvos em: $RESULTS_FILE"
log "Logs detalhados em: ${RESULTS_FILE}_*.log"

# Mostrar resumo na tela
if [[ -f "$RESULTS_FILE" ]]; then
    log "=== RESUMO ==="
    tail -n 20 "$RESULTS_FILE"
fi