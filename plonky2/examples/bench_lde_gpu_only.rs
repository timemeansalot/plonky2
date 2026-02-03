/// LDE benchmark - GPU with output on device (matching actual E2E flow)
/// In E2E proving, GPU LDE output stays on GPU for Merkle tree
///
/// Run with: NUM_OF_GPUS=1 cargo run --release --features=cuda --example bench_lde_gpu_only

use anyhow::Result;
use plonky2_field::goldilocks_field::GoldilocksField;
use plonky2_field::polynomial::PolynomialCoeffs;
use plonky2_field::types::{Field, PrimeField64};
use rand::random;
use std::time::Instant;

#[cfg(feature = "cuda")]
use zeknox::{
    device::memory::HostOrDeviceSlice,
    get_number_of_gpus_rs, init_coset_rs, init_twiddle_factors_rs,
    lde_batch, types::NTTConfig,
};

#[cfg(not(target_os = "macos"))]
use jemallocator::Jemalloc;

#[cfg(not(target_os = "macos"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

type F = GoldilocksField;

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

fn bench_lde_cpu(log_n: usize, rate_bits: usize, batches: usize) -> std::time::Duration {
    let n = 1 << log_n;

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
fn bench_lde_gpu_output_on_device(log_n: usize, rate_bits: usize, batches: usize) -> std::time::Duration {
    let input_domain_size = 1 << log_n;
    let output_domain_size = 1 << (log_n + rate_bits);
    let total_output = output_domain_size * batches;

    // Create random input data on CPU
    let inputs: Vec<u64> = (0..input_domain_size * batches).map(|_| random_fr()).collect();

    // Configure LDE - output stays on GPU (matching E2E flow)
    let mut cfg = NTTConfig::default();
    cfg.batches = batches as u32;
    cfg.extension_rate_bits = rate_bits as u32;
    cfg.with_coset = true;
    cfg.are_inputs_on_device = false;  // Input from CPU
    cfg.are_outputs_on_device = true;  // Output stays on GPU

    // Allocate GPU output buffer
    let mut device_output: HostOrDeviceSlice<'_, u64> =
        HostOrDeviceSlice::cuda_malloc(0, total_output).unwrap();

    let start = Instant::now();
    lde_batch(
        0, // device_id
        device_output.as_mut_ptr(),
        inputs.as_ptr(),
        log_n,
        cfg,
    );
    let elapsed = start.elapsed();

    // GPU memory will be freed when device_output is dropped
    elapsed
}

#[cfg(feature = "cuda")]
fn bench_lde_gpu_full_roundtrip(log_n: usize, rate_bits: usize, batches: usize) -> std::time::Duration {
    let input_domain_size = 1 << log_n;
    let output_domain_size = 1 << (log_n + rate_bits);
    let total_output = output_domain_size * batches;

    let inputs: Vec<u64> = (0..input_domain_size * batches).map(|_| random_fr()).collect();

    // Full roundtrip: CPU → GPU → CPU
    let mut cfg = NTTConfig::default();
    cfg.batches = batches as u32;
    cfg.extension_rate_bits = rate_bits as u32;
    cfg.with_coset = true;
    cfg.are_inputs_on_device = false;
    cfg.are_outputs_on_device = false;  // Copy output back to CPU

    let mut outputs: Vec<u64> = vec![0u64; total_output];

    let start = Instant::now();
    lde_batch(
        0,
        outputs.as_mut_ptr(),
        inputs.as_ptr(),
        log_n,
        cfg,
    );
    start.elapsed()
}

fn format_duration(d: std::time::Duration) -> String {
    let us = d.as_micros();
    if us >= 1_000_000 {
        format!("{:.2}s", us as f64 / 1_000_000.0)
    } else if us >= 1000 {
        format!("{:.2}ms", us as f64 / 1000.0)
    } else {
        format!("{}us", us)
    }
}

fn main() -> Result<()> {
    let _ = env_logger::builder().format_timestamp(None).try_init();

    #[cfg(feature = "cuda")]
    init_gpu();

    println!("{}", "=".repeat(90));
    println!("  LDE Benchmark: CPU vs GPU (output on device) vs GPU (full roundtrip)");
    println!("{}", "=".repeat(90));
    println!();

    let rate_bits = 3;

    // Test key configurations
    let test_configs = [
        (13, 2, "2^13 × 2"),
        (13, 10, "2^13 × 10"),
        (15, 2, "2^15 × 2"),
        (15, 10, "2^15 × 10"),
        (17, 2, "2^17 × 2"),
        (17, 10, "2^17 × 10"),
        (19, 2, "2^19 × 2"),
        (19, 10, "2^19 × 10"),
        (20, 2, "2^20 × 2"),
    ];

    println!("{:<15} | {:<12} | {:<15} | {:<15} | {:<12} | {:<12}",
             "Config", "CPU", "GPU (on device)", "GPU (roundtrip)", "Speedup(dev)", "Speedup(rt)");
    println!("{}", "-".repeat(90));

    for (log_n, batches, desc) in test_configs.iter() {
        let cpu_time = bench_lde_cpu(*log_n, rate_bits, *batches);

        #[cfg(feature = "cuda")]
        let gpu_on_device = bench_lde_gpu_output_on_device(*log_n, rate_bits, *batches);

        #[cfg(feature = "cuda")]
        let gpu_roundtrip = bench_lde_gpu_full_roundtrip(*log_n, rate_bits, *batches);

        #[cfg(not(feature = "cuda"))]
        let gpu_on_device = cpu_time;
        #[cfg(not(feature = "cuda"))]
        let gpu_roundtrip = cpu_time;

        let speedup_dev = cpu_time.as_secs_f64() / gpu_on_device.as_secs_f64();
        let speedup_rt = cpu_time.as_secs_f64() / gpu_roundtrip.as_secs_f64();

        println!(
            "{:<15} | {:<12} | {:<15} | {:<15} | {:<12.2}x | {:<12.2}x",
            desc,
            format_duration(cpu_time),
            format_duration(gpu_on_device),
            format_duration(gpu_roundtrip),
            speedup_dev,
            speedup_rt
        );
    }

    println!();
    println!("{}", "=".repeat(90));
    println!("  Analysis");
    println!("{}", "=".repeat(90));
    println!();
    println!("'GPU (on device)' = LDE output stays on GPU (actual E2E flow)");
    println!("'GPU (roundtrip)' = LDE output copied back to CPU");
    println!();
    println!("In E2E proving:");
    println!("  - GPU LDE output feeds directly into GPU Merkle tree");
    println!("  - No GPU→CPU transfer for LDE output");
    println!("  - CPU path requires CPU→GPU transfer for Merkle tree input");
    println!();

    Ok(())
}
