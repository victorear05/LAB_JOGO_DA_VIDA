#!/bin/bash

# Benchmark para as versões do Jogo da Vida

# Configurações
RESULTS_DIR="results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${RESULTS_DIR}/benchmark_${TIMESTAMP}.txt"
LOG_PREFIX="${RESULTS_DIR}/benchmark_${TIMESTAMP}"

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
    
    # Criar nome do arquivo de log correto (sem .txt no meio)
    local log_filename="${LOG_PREFIX}_${version}_${config// /_}.log"
    
    for ((rep=1; rep<=REPETITIONS; rep++)); do
        log "  Repetição $rep/$REPETITIONS"
        
        # Executar comando e capturar saída
        if timeout 300s bash -c "$exec_cmd" >> "$log_filename" 2>&1; then
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

# Versão Sequencial (baseline)
log "=== TESTE 1: Versão Sequencial ==="
run_test "Sequential" "baseline" "./exec/jogodavida"

# Versão OpenMP com diferentes números de threads
log "=== TESTE 2: OpenMP ==="
for threads in "${THREAD_COUNTS[@]}"; do
    export OMP_NUM_THREADS=$threads
    run_test "OpenMP" "$threads threads" "./exec/jogodavidaomp"
done

# Versão MPI com diferentes números de processos
log "=== TESTE 3: MPI ==="
for procs in "${MPI_PROCESSES[@]}"; do
    if command -v mpirun &> /dev/null; then
        run_test "MPI" "$procs processes" "mpirun -np $procs ./exec/jogodavidampi"
    else
        warning "MPI não disponível - pulando teste"
        break
    fi
done

# Versão CUDA
log "=== TESTE 4: CUDA ==="
if command -v nvidia-smi &> /dev/null; then
    run_test "CUDA" "GPU" "./exec/jogodavida_cuda"
else
    warning "NVIDIA GPU não detectada - pulando teste CUDA"
fi

# Versão OpenMP GPU
log "=== TESTE 5: OpenMP GPU ==="
run_test "OpenMP-GPU" "GPU offload" "./exec/jogodavidaomp_gpu"

# Resultado
echo "" >> "$RESULTS_FILE"
echo "=================================================================================" >> "$RESULTS_FILE"
echo "BENCHMARK CONCLUÍDO" >> "$RESULTS_FILE"
echo "Arquivo completo: $RESULTS_FILE" >> "$RESULTS_FILE"
echo "Logs detalhados: ${LOG_PREFIX}_*.log" >> "$RESULTS_FILE"
log "Execute o arquivo 'run_analiseResultados.sh' para a análise de resultados"
echo "=================================================================================" >> "$RESULTS_FILE"

success "Benchmark concluído!"
echo "📊 ANÁLISE DOS RESULTADOS DO BENCHMARK"
echo "======================================"

# Encontrar o arquivo de benchmark mais recente
LATEST_BENCHMARK=$(ls -t results/benchmark_*.txt 2>/dev/null | head -1)

if [[ ! -f "$LATEST_BENCHMARK" ]]; then
    echo "❌ Nenhum arquivo de benchmark encontrado!"
    echo "Execute './run_benchmark.sh' primeiro."
    exit 1
fi

echo "📂 Analisando: $LATEST_BENCHMARK"
echo ""

# Extrair timestamp do nome do arquivo
TIMESTAMP=$(basename "$LATEST_BENCHMARK" .txt | sed 's/benchmark_//')

# Verificar se existem arquivos de log (suporta ambos os formatos)
LOG_FILES=(results/benchmark_${TIMESTAMP}.txt_*.log)
if [[ ! -f "${LOG_FILES[0]}" ]]; then
    LOG_FILES=(results/benchmark_${TIMESTAMP}_*.log)
fi

if [[ ! -f "${LOG_FILES[0]}" ]]; then
    echo "❌ Arquivos de log não encontrados!"
    exit 1
fi

# Cores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Arrays para armazenar dados
declare -A times_8 times_16 times_32 times_64 times_128
declare -a versions

# Função para extrair tempo de um arquivo
extract_time() {
    local file="$1"
    local tam="$2"
    grep "tam=$tam.*tot=" "$file" 2>/dev/null | head -1 | grep -o "tot=[0-9.]*" | cut -d= -f2
}

# Processar cada arquivo de log
for log_file in "${LOG_FILES[@]}"; do
    if [[ -f "$log_file" ]]; then
        # Extrair nome da versão (funciona para ambos os formatos)
        version=$(basename "$log_file" .log | sed -e "s/benchmark_${TIMESTAMP}.txt_//" -e "s/benchmark_${TIMESTAMP}_//" | tr '_' ' ')
        versions+=("$version")
        
        # Extrair tempos para cada tamanho
        times_8["$version"]=$(extract_time "$log_file" 8)
        times_16["$version"]=$(extract_time "$log_file" 16)
        times_32["$version"]=$(extract_time "$log_file" 32)
        times_64["$version"]=$(extract_time "$log_file" 64)
        times_128["$version"]=$(extract_time "$log_file" 128)
    fi
done

# Ordenar versões para melhor apresentação
IFS=$'\n' sorted_versions=($(sort <<<"${versions[*]}"))
unset IFS

# Imprimir tabela de resultados
echo -e "${BLUE}TABELA COMPARATIVA DE TEMPOS DE EXECUÇÃO (em segundos)${NC}"
echo ""
printf "%-25s %12s %12s %12s %12s %12s\n" "Versão" "Tam=8" "Tam=16" "Tam=32" "Tam=64" "Tam=128"
echo "------------------------------------------------------------------------------------------------"

# Identificar baseline para cálculo de speedup
baseline_version=""
for version in "${sorted_versions[@]}"; do
    if [[ "$version" == *"Sequential"* ]] || [[ "$version" == *"baseline"* ]]; then
        baseline_version="$version"
        break
    fi
done

# Imprimir dados de cada versão
for version in "${sorted_versions[@]}"; do
    # Obter tempos
    t8="${times_8[$version]:-N/A}"
    t16="${times_16[$version]:-N/A}"
    t32="${times_32[$version]:-N/A}"
    t64="${times_64[$version]:-N/A}"
    t128="${times_128[$version]:-N/A}"
    
    # Formatar tempos com speedup se disponível
    if [[ -n "$baseline_version" && "$version" != "$baseline_version" ]]; then
        # Calcular e adicionar speedup entre parênteses
        base_8="${times_8[$baseline_version]}"
        base_16="${times_16[$baseline_version]}"
        base_32="${times_32[$baseline_version]}"
        base_64="${times_64[$baseline_version]}"
        base_128="${times_128[$baseline_version]}"
        
        # Função para formatar tempo com speedup
        format_with_speedup() {
            local curr_time=$1
            local base_time=$2
            if [[ "$curr_time" != "N/A" && -n "$base_time" && "$curr_time" != "0" ]]; then
                local speedup=$(awk -v b="$base_time" -v c="$curr_time" 'BEGIN {printf "%.1f", b/c}')
                echo "${curr_time}s (${speedup}x)"
            elif [[ "$curr_time" != "N/A" ]]; then
                echo "${curr_time}s"
            else
                echo "N/A"
            fi
        }
        
        t8_fmt=$(format_with_speedup "$t8" "$base_8")
        t16_fmt=$(format_with_speedup "$t16" "$base_16")
        t32_fmt=$(format_with_speedup "$t32" "$base_32")
        t64_fmt=$(format_with_speedup "$t64" "$base_64")
        t128_fmt=$(format_with_speedup "$t128" "$base_128")
    else
        # Versão baseline ou sem baseline disponível
        [[ "$t8" != "N/A" ]] && t8_fmt="${t8}s" || t8_fmt="N/A"
        [[ "$t16" != "N/A" ]] && t16_fmt="${t16}s" || t16_fmt="N/A"
        [[ "$t32" != "N/A" ]] && t32_fmt="${t32}s" || t32_fmt="N/A"
        [[ "$t64" != "N/A" ]] && t64_fmt="${t64}s" || t64_fmt="N/A"
        [[ "$t128" != "N/A" ]] && t128_fmt="${t128}s" || t128_fmt="N/A"
    fi
    
    # Destacar baseline
    if [[ "$version" == "$baseline_version" ]]; then
        echo -e "${GREEN}$(printf "%-25s %12s %12s %12s %12s %12s" "$version" "$t8_fmt" "$t16_fmt" "$t32_fmt" "$t64_fmt" "$t128_fmt")${NC}"
    else
        printf "%-25s %12s %12s %12s %12s %12s\n" "$version" "$t8_fmt" "$t16_fmt" "$t32_fmt" "$t64_fmt" "$t128_fmt"
    fi
done

echo "------------------------------------------------------------------------------------------------"
echo ""
echo "💡 Legenda: tempo em segundos (speedup em relação ao baseline)"
echo "🟢 Linha verde = versão baseline para comparação"

echo ""
echo "📁 Arquivos detalhados em: results/"