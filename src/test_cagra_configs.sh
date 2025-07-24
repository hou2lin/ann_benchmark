#!/bin/bash

# CAGRA Base Configuration Test Script
# Tests base group configs from cuvs_cagra.yaml

set -e

EXECUTABLE="./build/test_cagra_cuvs"
LOG_DIR="./test_logs"
RESULTS_FILE="./test_cagra_results.csv"

# Default parameters
N_SAMPLES=10000000
N_DIM=768
TOPK=10
N_QUERIES=1000
BATCH_SIZE=10

mkdir -p "$LOG_DIR"

# Function to run a test
run_test() {
    local test_name="$1"
    local intermediate_graph_degree="$2"
    local graph_degree="$3"
    local itopk_size="$4"
    local search_width="$5"
    
    local log_file="$LOG_DIR/${test_name}.log"
    local cmd="$EXECUTABLE --num $N_SAMPLES --dim $N_DIM --topk $TOPK --queries $N_QUERIES --batch_size $BATCH_SIZE --intermediate_graph_degree $intermediate_graph_degree --graph_degree $graph_degree --itopk_size $itopk_size --max_iterations 0 --search_width $search_width"
    
    echo "[INFO] Running: $test_name"
    
    # Run test and capture output
    {
        echo "=== Test: $test_name ==="
        eval $cmd 2>&1
        echo "=== End Test ==="
    } | tee "$log_file"
    
    # Extract metrics
    local qps=$(grep -o "qps is : [0-9.]*" "$log_file" | tail -1 | awk '{print $4}' || echo "N/A")
    local recall=$(grep -o "recall_ratio_ret is : [0-9.]*" "$log_file" | tail -1 | awk '{print $4}' || echo "N/A")
    
    # Save to CSV
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$test_name,$intermediate_graph_degree,$graph_degree,$itopk_size,$search_width,$qps,$recall" >> "$RESULTS_FILE"
    
    echo "[SUCCESS] $test_name (QPS: $qps, Recall: $recall)"
}

# Initialize results file
echo "timestamp,test_name,intermediate_graph_degree,graph_degree,itopk_size,search_width,qps,recall" > "$RESULTS_FILE"

# Check executable
if [[ ! -f "$EXECUTABLE" ]]; then
    echo "[ERROR] Executable not found: $EXECUTABLE"
    exit 1
fi

echo "[INFO] Running base configurations..."

# Base configs from YAML: graph_degree=[32,64,96,128], intermediate_graph_degree=[32,64,96,128], itopk=[32,64,128,256,512], search_width=[1,2,4,8,16,32,64]
for gd in 96 128; do #32 64 
    for igd in 96 128; do #32 64 
        for itopk in 256 512; do #32 64 128
            for sw in 64 128; do #1 2 4 8 16 32 64
                run_test "base_g${gd}_ig${igd}_i${itopk}_sw${sw}" "$igd" "$gd" "$itopk" "$sw"
            done
        done
    done
done

echo "[SUCCESS] All tests completed!"
echo "[INFO] Results: $RESULTS_FILE"
echo "[INFO] Logs: $LOG_DIR/" 