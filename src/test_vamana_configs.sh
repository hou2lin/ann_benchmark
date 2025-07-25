#!/bin/bash

# VAMANA Base Configuration Test Script
# Tests base group configs from cuvs_vamana.yaml

set -e

EXECUTABLE="./build/test_vamana_cuvs"
LOG_DIR="./test_logs"
RESULTS_FILE="./test_vamana_results.csv"

# Default parameters
N_SAMPLES=10000000
N_DIM=768
TOPK=10
N_QUERIES=1000
BATCH_SIZE=10

# VAMANA specific defaults
VISITED_SIZE=256
MAX_FRACTION=0.2
VAMANA_ITERS=1
ITOPK_SIZE=256
MAX_ITERATIONS=200
FORCE_REBUILD_INDEX=0
DISKANN_L_SEARCH=500

mkdir -p "$LOG_DIR"

# Function to run a test
run_test() {
    local test_name="$1"
    local graph_degree="$2"
    local visited_size="$3"
    local alpha="$4"
    local l_search="$5"
    local num_threads="$6"
    
    local log_file="$LOG_DIR/${test_name}.log"
    local cmd="$EXECUTABLE --num $N_SAMPLES --dim $N_DIM --topk $TOPK --queries $N_QUERIES --batch_size $BATCH_SIZE --graph_degree $graph_degree --visited_size $visited_size --max_fraction $alpha --vamana_iters $VAMANA_ITERS --search_width $l_search --itopk_size $ITOPK_SIZE --max_iterations $MAX_ITERATIONS --force_rebuild_index $FORCE_REBUILD_INDEX --diskann_l_search $DISKANN_L_SEARCH --force_rebuild_index 1"
    
    echo "[INFO] Running: $test_name"
    
    # Run test and capture output
    {
        echo "=== Test: $test_name ==="
        eval $cmd 2>&1
        echo "=== End Test ==="
    } | tee "$log_file"
    
    # Extract metrics
    local qps=$(grep -o "DiskANN QPS: [0-9.]*" "$log_file" | tail -1 | awk '{print $4}' || echo "N/A")
    local recall=$(grep -o "Recall: [0-9.]*" "$log_file" | tail -1 | awk '{print $4}' || echo "N/A")
    
    # Save to CSV
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$test_name,$graph_degree,$visited_size,$alpha,$l_search,$num_threads,$qps,$recall" >> "$RESULTS_FILE"
    
    echo "[SUCCESS] $test_name (QPS: $qps, Recall: $recall)"
}

# Initialize results file
echo "timestamp,test_name,graph_degree,visited_size,alpha,l_search,num_threads,qps,recall" > "$RESULTS_FILE"

# Check executable
if [[ ! -f "$EXECUTABLE" ]]; then
    echo "[ERROR] Executable not found: $EXECUTABLE"
    exit 1
fi

echo "[INFO] Running VAMANA base configurations..."

# Base configs from YAML: 
# build: graph_degree=[32,64], visited_size=[128,256], alpha=[1.2]
# search: L_search=[10,20,30,40,50,100,200,300]
# num_threads=[32]
# Total: 2 * 2 * 1 * 8 * 1 = 32 tests
for gd in 128; do #32 64
    for vs in 128 256; do
        for alpha in 0.4 0.6 1.0 1.2; do
            for l_search in 50 100 200 300; do
                for num_threads in 16 32; do
                    run_test "vamana_g${gd}_v${vs}_a${alpha}_l${l_search}_t${num_threads}" "$gd" "$vs" "$alpha" "$l_search" "$num_threads"
                done
            done
        done
    done
done

echo "[SUCCESS] All tests completed!"
echo "[INFO] Results: $RESULTS_FILE"
echo "[INFO] Logs: $LOG_DIR/" 