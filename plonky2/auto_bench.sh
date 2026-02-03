#!/bin/bash

# Plonky2 GPU vs CPU Benchmark Script
# This script runs benchmarks for:
# 1. Primitive operations (LDE, Merkle Tree)
# 2. End-to-end proving (Goldilocks and BN128)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Temp files for results
GOLD_GPU_FILE="/tmp/plonky2_gold_gpu.txt"
GOLD_CPU_FILE="/tmp/plonky2_gold_cpu.txt"
BN128_GPU_FILE="/tmp/plonky2_bn128_gpu.txt"
BN128_CPU_FILE="/tmp/plonky2_bn128_cpu.txt"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Plonky2 GPU vs CPU Benchmark Suite   ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check for CUDA feature
if ! cargo check --features=cuda 2>/dev/null; then
    echo -e "${RED}Error: CUDA feature not available. Make sure CUDA is installed.${NC}"
    exit 1
fi

# Build all benchmarks
echo -e "${YELLOW}Building benchmarks...${NC}"
cargo build --release --features=cuda \
    --example bench_primitives \
    --example bench_e2e_prove \
    --example bench_bn128 2>/dev/null

echo -e "${GREEN}Build complete!${NC}"
echo ""

# ========================================
# Part 1: Primitive Benchmarks
# ========================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Part 1: Primitive Operations         ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

echo -e "${YELLOW}Running primitive benchmarks (LDE + Merkle Tree)...${NC}"
NUM_OF_GPUS=1 cargo run --release --features=cuda --example bench_primitives 2>&1 | tee /tmp/plonky2_primitives.txt
echo ""

# ========================================
# Part 2: E2E Prove Benchmarks
# ========================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Part 2: End-to-End Proving           ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to run benchmark and save results
run_benchmark() {
    local name=$1
    local output_file=$2
    local env_vars=$3
    local example=$4

    echo -e "${YELLOW}Running $name...${NC}"
    eval "$env_vars cargo run --release --features=cuda --example $example 2>&1" | \
        grep -E "(Circuit size|^Prove:)" > "$output_file"
    echo -e "${GREEN}$name complete!${NC}"
}

# Run all E2E benchmarks
echo -e "${BLUE}--- Running Goldilocks Benchmarks ---${NC}"
run_benchmark "Goldilocks GPU" "$GOLD_GPU_FILE" "NUM_OF_GPUS=1" "bench_e2e_prove"
run_benchmark "Goldilocks CPU" "$GOLD_CPU_FILE" "DISABLE_GPU_LDE=1 NUM_OF_GPUS=1" "bench_e2e_prove"

echo ""
echo -e "${BLUE}--- Running BN128 Benchmarks ---${NC}"
run_benchmark "BN128 GPU" "$BN128_GPU_FILE" "NUM_OF_GPUS=1" "bench_bn128"
run_benchmark "BN128 CPU" "$BN128_CPU_FILE" "DISABLE_GPU_LDE=1 NUM_OF_GPUS=1" "bench_bn128"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}      E2E BENCHMARK RESULTS            ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to display comparison table
display_comparison() {
    local title=$1
    local gpu_file=$2
    local cpu_file=$3

    echo -e "${GREEN}=== $title ===${NC}"
    echo ""
    printf "%-8s | %-12s | %-12s | %-10s\n" "Degree" "CPU Prove" "GPU Prove" "Speedup"
    printf "%-8s-+-%-12s-+-%-12s-+-%-10s\n" "--------" "------------" "------------" "----------"

    # Extract circuit sizes and prove times
    local degrees=($(grep "Circuit size" "$gpu_file" | sed 's/.*2\^\([0-9]*\).*/\1/'))
    local gpu_times=($(grep "^Prove:" "$gpu_file" | sed 's/Prove: //'))
    local cpu_times=($(grep "^Prove:" "$cpu_file" | sed 's/Prove: //'))

    for i in "${!degrees[@]}"; do
        local deg="${degrees[$i]}"
        local gpu="${gpu_times[$i]}"
        local cpu="${cpu_times[$i]}"

        # Convert times to milliseconds for speedup calculation
        local gpu_ms=$(echo "$gpu" | sed 's/ms$//' | sed 's/s$//' | awk '{
            if ($0 ~ /ms$/) { gsub(/ms$/, ""); print $0 }
            else if ($0 ~ /s$/) { gsub(/s$/, ""); print $0 * 1000 }
            else if ($0 ~ /\..*s/) { gsub(/s$/, ""); print $0 * 1000 }
            else { print $0 }
        }')
        local cpu_ms=$(echo "$cpu" | sed 's/ms$//' | sed 's/s$//' | awk '{
            if ($0 ~ /ms$/) { gsub(/ms$/, ""); print $0 }
            else if ($0 ~ /s$/) { gsub(/s$/, ""); print $0 * 1000 }
            else if ($0 ~ /\..*s/) { gsub(/s$/, ""); print $0 * 1000 }
            else { print $0 }
        }')

        # Calculate speedup
        local speedup=$(awk "BEGIN {printf \"%.2f\", $cpu_ms / $gpu_ms}")

        printf "%-8s | %-12s | %-12s | %-10s\n" "$deg" "$cpu" "$gpu" "${speedup}x"
    done
    echo ""
}

# Display results
display_comparison "Goldilocks (64-bit field)" "$GOLD_GPU_FILE" "$GOLD_CPU_FILE"
display_comparison "BN128 (254-bit hashing)" "$BN128_GPU_FILE" "$BN128_CPU_FILE"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}             SUMMARY                   ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "GPU acceleration provides:"
echo ""
echo "  Primitive Operations:"
echo "    - Merkle Tree (Poseidon): ~4-44x speedup"
echo "    - LDE (NTT): Limited benefit (CPU AVX512 is fast)"
echo ""
echo "  End-to-End Proving:"
echo "    - Goldilocks: ~1.2-2.6x speedup"
echo "    - BN128: ~1.1-1.5x speedup"
echo ""
echo "Results saved to:"
echo "  - /tmp/plonky2_primitives.txt"
echo "  - $GOLD_GPU_FILE"
echo "  - $GOLD_CPU_FILE"
echo "  - $BN128_GPU_FILE"
echo "  - $BN128_CPU_FILE"
echo ""
echo -e "${GREEN}Benchmark complete!${NC}"
