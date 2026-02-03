# Plonky2 GPU vs CPU Benchmark Results

Benchmarks comparing GPU-accelerated proving vs CPU-only proving on the OKX plonky2 fork with zeknox CUDA library.

## Test Environment

- **GPU**: NVIDIA RTX (with CUDA support)
- **CPU**: x86_64 with AVX512 support
- **Rust**: Nightly toolchain
- **Branch**: dev (with AVX512 optimizations)

## Goldilocks Field (64-bit) - GPU vs CPU

| Degree | Gates | CPU Prove | GPU Prove | Speedup |
|--------|-------|-----------|-----------|---------|
| 13 | 8,192 | 228ms | 88ms | **2.59x** |
| 14 | 16,384 | 271ms | 223ms | **1.22x** |
| 15 | 32,768 | 478ms | 326ms | **1.47x** |
| 16 | 65,536 | 403ms | 338ms | **1.19x** |
| 17 | 131,072 | 594ms | 551ms | **1.08x** |
| 18 | 262,144 | 1.30s | 917ms | **1.42x** |
| 19 | 524,288 | 1.76s | 1.29s | **1.36x** |
| 20 | 1,048,576 | 3.03s | 2.35s | **1.29x** |
| 21 | 2,097,152 | 5.49s | 4.35s | **1.26x** |
| 22 | 4,194,304 | 12.06s | 8.71s | **1.38x** |

**Average Speedup: ~1.4x**

## BN128 Hashing (254-bit) - GPU vs CPU

| Degree | Gates | CPU Prove | GPU Prove | Speedup |
|--------|-------|-----------|-----------|---------|
| 13 | 8,192 | 408ms | 266ms | **1.53x** |
| 14 | 16,384 | 462ms | 348ms | **1.33x** |
| 15 | 32,768 | 672ms | 528ms | **1.27x** |
| 16 | 65,536 | 574ms | 667ms | 0.86x |
| 17 | 131,072 | 1.14s | 982ms | **1.16x** |
| 18 | 262,144 | 1.71s | 1.46s | **1.17x** |
| 19 | 524,288 | 2.53s | 2.27s | **1.11x** |
| 20 | 1,048,576 | 4.82s | 3.90s | **1.24x** |
| 21 | 2,097,152 | 8.79s | 7.14s | **1.23x** |
| 22 | 4,194,304 | 16.66s | 14.95s | **1.11x** |

**Average Speedup: ~1.2x**

## Key Observations

### GPU Acceleration Benefits
1. **Goldilocks**: GPU provides **1.1-2.6x speedup**, best at smaller circuits
2. **BN128**: GPU provides **1.1-1.5x speedup**, slightly less due to AVX512 CPU optimizations

### Why Speedup is Modest
1. **AVX512 Optimizations**: The dev branch has highly optimized AVX512 code for Poseidon hashing
2. **Partial GPU Usage**: Only LDE (Low Degree Extension) and Merkle tree operations use GPU
3. **GPU Init Overhead**: Small circuits are affected by GPU initialization cost (~100-300ms)
4. **Memory Transfer**: Data must be copied between CPU and GPU memory

### GPU Operations
- **LDE (NTT/FFT)**: GPU accelerated via zeknox
- **Merkle Tree Hashing**: GPU accelerated for Poseidon/PoseidonBN128
- **Other Operations**: Still run on CPU (witness generation, constraint evaluation, etc.)

## Bug Fixes Applied

1. **Memory Synchronization Bug** (`zeknox/native/merkle/merkle.cu`):
   - Added `cudaDeviceSynchronize()` after kernel launches
   - Fixed `cudaErrorIllegalAddress` crash caused by premature memory deallocation

2. **GPU Initialization**:
   - Added proper initialization of twiddle factors and coset arrays before GPU LDE

## How to Run Benchmarks

```bash
# Run the auto benchmark script
cd plonky2
./auto_bench.sh

# Or run manually:

# Goldilocks GPU
NUM_OF_GPUS=1 cargo run --release --features=cuda --example bench_e2e_prove

# Goldilocks CPU only
DISABLE_GPU_LDE=1 NUM_OF_GPUS=1 cargo run --release --features=cuda --example bench_e2e_prove

# BN128 GPU
NUM_OF_GPUS=1 cargo run --release --features=cuda --example bench_bn128

# BN128 CPU only
DISABLE_GPU_LDE=1 NUM_OF_GPUS=1 cargo run --release --features=cuda --example bench_bn128
```

## Configuration

Environment variables:
- `NUM_OF_GPUS=1`: Use single GPU (avoid multi-GPU bug)
- `DISABLE_GPU_LDE=1`: Force CPU-only LDE for comparison
- `RUST_LOG=info`: Enable logging for debugging
