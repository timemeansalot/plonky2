/// Benchmark individual GPU vs CPU primitives: NTT/LDE, Merkle Tree
///
/// This benchmark measures raw performance of:
/// 1. NTT/LDE (Number Theoretic Transform / Low Degree Extension)
/// 2. Merkle Tree construction with Poseidon hashing
///
/// Run with: NUM_OF_GPUS=1 cargo run --release --features=cuda --example bench_primitives

use anyhow::Result;
use core::mem::MaybeUninit;
use core::slice;
use plonky2::hash::hash_types::NUM_HASH_OUT_ELTS;
use plonky2::hash::merkle_tree::MerkleTree;
use plonky2::hash::poseidon::PoseidonHash;
use plonky2::plonk::config::{Hasher, PoseidonGoldilocksConfig};
use plonky2_field::goldilocks_field::GoldilocksField;
use plonky2_field::polynomial::PolynomialCoeffs;
use plonky2_field::types::{Field, PrimeField64};
use rand::random;
use std::time::Instant;

#[cfg(feature = "cuda")]
use zeknox::{
    device::{memory::HostOrDeviceSlice, stream::CudaStream},
    fill_digests_buf_linear_gpu_with_gpu_ptr, get_number_of_gpus_rs, init_coset_rs,
    init_twiddle_factors_rs, lde_batch, types::NTTConfig,
};

#[cfg(not(target_os = "macos"))]
use jemallocator::Jemalloc;

#[cfg(not(target_os = "macos"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

type F = GoldilocksField;
type C = PoseidonGoldilocksConfig;

fn random_fr() -> u64 {
    let fr: u64 = random();
    fr % 0xffffffff00000001
}

fn random_field_vec(n: usize) -> Vec<F> {
    (0..n).map(|_| F::from_canonical_u64(random_fr())).collect()
}

#[cfg(feature = "cuda")]
fn init_gpu() {
    let num_of_gpus = match std::env::var("NUM_OF_GPUS") {
        Ok(val) => val.parse().unwrap_or_else(|_| get_number_of_gpus_rs()),
        Err(_) => {
            let detected = get_number_of_gpus_rs();
            std::env::set_var("NUM_OF_GPUS", detected.to_string());
            detected
        }
    };

    println!("Initializing {} GPU(s)...", num_of_gpus);

    let log_ns: Vec<usize> = (2..27).collect();

    for device_id in 0..num_of_gpus {
        init_coset_rs(
            device_id,
            26,
            GoldilocksField::coset_shift().to_canonical_u64(),
        );

        for log_n in &log_ns {
            init_twiddle_factors_rs(device_id, *log_n);
        }
    }
    println!("GPU initialization complete!\n");
}

// ==================== LDE Benchmarks ====================

fn bench_lde_cpu(log_n: usize, rate_bits: usize, batches: usize) -> std::time::Duration {
    let n = 1 << log_n;

    // Create random polynomials
    let polys: Vec<PolynomialCoeffs<F>> = (0..batches)
        .map(|_| PolynomialCoeffs::new(random_field_vec(n)))
        .collect();

    let start = Instant::now();
    for poly in polys {
        let _ = poly.lde(rate_bits);
    }
    start.elapsed()
}

#[cfg(feature = "cuda")]
fn bench_lde_gpu(log_n: usize, rate_bits: usize, batches: usize) -> std::time::Duration {
    let input_domain_size = 1 << log_n;
    let output_domain_size = 1 << (log_n + rate_bits);
    let total_output = output_domain_size * batches;

    // Create random input data
    let inputs: Vec<u64> = (0..input_domain_size * batches).map(|_| random_fr()).collect();

    // Configure LDE
    let mut cfg = NTTConfig::default();
    cfg.batches = batches as u32;
    cfg.extension_rate_bits = rate_bits as u32;
    cfg.with_coset = true;
    cfg.are_inputs_on_device = false;
    cfg.are_outputs_on_device = false;

    // Allocate output
    let mut outputs: Vec<u64> = vec![0u64; total_output];

    let start = Instant::now();
    lde_batch(
        0, // device_id
        outputs.as_mut_ptr(),
        inputs.as_ptr(),
        log_n,
        cfg,
    );
    start.elapsed()
}

// ==================== Merkle Tree Benchmarks ====================

fn capacity_up_to_mut<T>(v: &mut Vec<T>, len: usize) -> &mut [MaybeUninit<T>] {
    assert!(v.capacity() >= len);
    let v_ptr = v.as_mut_ptr().cast::<MaybeUninit<T>>();
    unsafe { slice::from_raw_parts_mut(v_ptr, len) }
}

fn bench_merkle_cpu(log_leaves: usize, leaf_size: usize, cap_height: usize) -> std::time::Duration {
    let num_leaves = 1 << log_leaves;
    let leaves: Vec<Vec<F>> = (0..num_leaves)
        .map(|_| random_field_vec(leaf_size))
        .collect();

    let start = Instant::now();
    let _mt = MerkleTree::<F, PoseidonHash>::new_from_2d(leaves, cap_height);
    start.elapsed()
}

#[cfg(feature = "cuda")]
fn bench_merkle_gpu(log_leaves: usize, leaf_size: usize, cap_height: usize) -> std::time::Duration {
    let num_leaves = 1 << log_leaves;
    let gpu_id = 0i32;

    // Create flat leaves array
    let leaves_1d: Vec<F> = (0..num_leaves * leaf_size)
        .map(|_| F::from_canonical_u64(random_fr()))
        .collect();

    // Calculate buffer sizes
    let num_digests = 2 * (num_leaves - (1 << cap_height));
    let len_cap = 1 << cap_height;
    let digests_size = if num_digests == 0 { NUM_HASH_OUT_ELTS } else { num_digests * NUM_HASH_OUT_ELTS };
    let caps_size = if len_cap == 0 { NUM_HASH_OUT_ELTS } else { len_cap * NUM_HASH_OUT_ELTS };

    let start = Instant::now();

    // Allocate GPU memory
    let mut gpu_leaves: HostOrDeviceSlice<'_, F> =
        HostOrDeviceSlice::cuda_malloc(gpu_id, leaves_1d.len()).unwrap();
    let mut gpu_digests: HostOrDeviceSlice<'_, F> =
        HostOrDeviceSlice::cuda_malloc(gpu_id, digests_size).unwrap();
    let mut gpu_caps: HostOrDeviceSlice<'_, F> =
        HostOrDeviceSlice::cuda_malloc(gpu_id, caps_size).unwrap();

    // Copy leaves to GPU
    gpu_leaves.copy_from_host(&leaves_1d).unwrap();

    // Run GPU Merkle tree construction
    unsafe {
        fill_digests_buf_linear_gpu_with_gpu_ptr(
            gpu_digests.as_mut_ptr() as *mut core::ffi::c_void,
            gpu_caps.as_mut_ptr() as *mut core::ffi::c_void,
            gpu_leaves.as_ptr() as *mut core::ffi::c_void,
            num_digests as u64,
            len_cap as u64,
            num_leaves as u64,
            leaf_size as u64,
            cap_height as u64,
            0, // HasherType::Poseidon
            gpu_id as u64,
        );
    }

    // Copy results back (include in timing as it's part of real usage)
    let mut digests: Vec<<PoseidonHash as Hasher<F>>::Hash> = Vec::with_capacity(num_digests);
    let mut cap: Vec<<PoseidonHash as Hasher<F>>::Hash> = Vec::with_capacity(len_cap);

    let stream1 = CudaStream::create().unwrap();
    let stream2 = CudaStream::create().unwrap();

    if num_digests > 0 {
        let digests_buf = capacity_up_to_mut(&mut digests, num_digests);
        gpu_digests
            .copy_to_host_ptr_async(
                digests_buf.as_mut_ptr() as *mut core::ffi::c_void,
                digests_size,
                &stream1,
            )
            .unwrap();
    }

    if len_cap > 0 {
        let cap_buf = capacity_up_to_mut(&mut cap, len_cap);
        gpu_caps
            .copy_to_host_ptr_async(
                cap_buf.as_mut_ptr() as *mut core::ffi::c_void,
                caps_size,
                &stream2,
            )
            .unwrap();
    }

    stream1.synchronize().unwrap();
    stream2.synchronize().unwrap();
    stream1.destroy().unwrap();
    stream2.destroy().unwrap();

    start.elapsed()
}

fn format_duration(d: std::time::Duration) -> String {
    let ms = d.as_secs_f64() * 1000.0;
    if ms >= 1000.0 {
        format!("{:.2}s", ms / 1000.0)
    } else {
        format!("{:.1}ms", ms)
    }
}

fn main() -> Result<()> {
    let _ = env_logger::builder().format_timestamp(None).try_init();

    #[cfg(feature = "cuda")]
    init_gpu();

    println!("{}", "=".repeat(70));
    println!("  Plonky2 GPU vs CPU Primitives Benchmark");
    println!("{}", "=".repeat(70));
    println!();

    // ==================== LDE Benchmark ====================
    println!("{}", "=".repeat(70));
    println!("  1. LDE (Low Degree Extension) - NTT-based polynomial extension");
    println!("{}", "=".repeat(70));
    println!();

    let rate_bits = 3; // 8x extension
    let batches = 100;

    println!("Configuration: rate_bits={} ({}x extension), batches={}", rate_bits, 1 << rate_bits, batches);
    println!();

    println!("{:<10} | {:<12} | {:<12} | {:<10}", "Log Size", "CPU Time", "GPU Time", "Speedup");
    println!("{}", "-".repeat(52));

    for log_n in [12, 14, 16, 18, 20].iter() {
        let cpu_time = bench_lde_cpu(*log_n, rate_bits, batches);

        #[cfg(feature = "cuda")]
        let gpu_time = bench_lde_gpu(*log_n, rate_bits, batches);

        #[cfg(not(feature = "cuda"))]
        let gpu_time = cpu_time;

        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

        println!(
            "{:<10} | {:<12} | {:<12} | {:.2}x",
            format!("2^{}", log_n),
            format_duration(cpu_time),
            format_duration(gpu_time),
            speedup
        );
    }

    println!();

    // ==================== Merkle Tree Benchmark ====================
    println!("{}", "=".repeat(70));
    println!("  2. Merkle Tree Construction (Poseidon Hash)");
    println!("{}", "=".repeat(70));
    println!();

    let leaf_size = 135; // Typical leaf size in plonky2
    let cap_height = 4;

    println!("Configuration: leaf_size={}, cap_height={}", leaf_size, cap_height);
    println!();

    println!("{:<12} | {:<12} | {:<12} | {:<12} | {:<10}", "Leaves", "Log Size", "CPU Time", "GPU Time", "Speedup");
    println!("{}", "-".repeat(66));

    for log_leaves in [12, 14, 16, 18, 20].iter() {
        let cpu_time = bench_merkle_cpu(*log_leaves, leaf_size, cap_height);

        #[cfg(feature = "cuda")]
        let gpu_time = bench_merkle_gpu(*log_leaves, leaf_size, cap_height);

        #[cfg(not(feature = "cuda"))]
        let gpu_time = cpu_time;

        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

        println!(
            "{:<12} | {:<12} | {:<12} | {:<12} | {:.2}x",
            format!("{}", 1 << log_leaves),
            format!("2^{}", log_leaves),
            format_duration(cpu_time),
            format_duration(gpu_time),
            speedup
        );
    }

    println!();
    println!("{}", "=".repeat(70));
    println!("  Benchmark Complete!");
    println!("{}", "=".repeat(70));

    Ok(())
}
