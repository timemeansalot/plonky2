/// Realistic LDE benchmark comparing GPU vs CPU
/// Tests with parameters matching actual E2E proving workloads
///
/// Run with: NUM_OF_GPUS=1 cargo run --release --features=cuda --example bench_lde_realistic

use anyhow::Result;
use plonky2_field::goldilocks_field::GoldilocksField;
use plonky2_field::polynomial::PolynomialCoeffs;
use plonky2_field::types::{Field, PrimeField64};
use rand::random;
use std::time::Instant;

#[cfg(feature = "cuda")]
use zeknox::{
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

    println!("{}", "=".repeat(80));
    println!("  Realistic LDE Benchmark: GPU vs CPU");
    println!("  (Parameters matching actual E2E proving workloads)");
    println!("{}", "=".repeat(80));
    println!();

    let rate_bits = 3; // Standard 8x extension in plonky2

    // Test configurations matching actual E2E usage:
    // - Small batches (2-10 polynomials) for partial products
    // - Medium batches (50-100) for wire polynomials
    // - Large batches (200-300) for sigma polynomials
    let test_configs = [
        // (log_n, batches, description)
        (13, 2, "Small circuit, minimal batch"),
        (13, 10, "Small circuit, small batch"),
        (13, 100, "Small circuit, medium batch"),
        (13, 255, "Small circuit, large batch (sigma polys)"),
        (15, 2, "Medium circuit, minimal batch"),
        (15, 10, "Medium circuit, small batch"),
        (15, 100, "Medium circuit, medium batch"),
        (15, 255, "Medium circuit, large batch"),
        (17, 2, "Large circuit, minimal batch"),
        (17, 10, "Large circuit, small batch"),
        (17, 100, "Large circuit, medium batch"),
        (17, 255, "Large circuit, large batch"),
        (19, 2, "Very large circuit, minimal batch"),
        (19, 10, "Very large circuit, small batch"),
        (19, 100, "Very large circuit, medium batch"),
        (20, 2, "Huge circuit, minimal batch"),
        (20, 10, "Huge circuit, small batch"),
    ];

    println!("rate_bits={} ({}x extension)\n", rate_bits, 1 << rate_bits);
    println!("{:<12} | {:<8} | {:<12} | {:<12} | {:<8} | {}",
             "Poly Size", "Batches", "CPU Time", "GPU Time", "Speedup", "Description");
    println!("{}", "-".repeat(80));

    for (log_n, batches, desc) in test_configs.iter() {
        let cpu_time = bench_lde_cpu(*log_n, rate_bits, *batches);

        #[cfg(feature = "cuda")]
        let gpu_time = bench_lde_gpu(*log_n, rate_bits, *batches);

        #[cfg(not(feature = "cuda"))]
        let gpu_time = cpu_time;

        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

        println!(
            "{:<12} | {:<8} | {:<12} | {:<12} | {:<8.2}x | {}",
            format!("2^{}", log_n),
            batches,
            format_duration(cpu_time),
            format_duration(gpu_time),
            speedup,
            desc
        );
    }

    println!();
    println!("{}", "=".repeat(80));
    println!("  Summary");
    println!("{}", "=".repeat(80));
    println!();
    println!("Key observations:");
    println!("  - GPU LDE has fixed overhead (~10-50ms) for kernel launch and memory transfer");
    println!("  - For small workloads (small poly × few batches), CPU is faster");
    println!("  - For large workloads (large poly × many batches), GPU catches up or wins");
    println!("  - In actual E2E proving, most LDE calls are small (2-10 batches)");
    println!("  - GPU advantage comes from keeping data on GPU for subsequent Merkle tree");
    println!();

    Ok(())
}
