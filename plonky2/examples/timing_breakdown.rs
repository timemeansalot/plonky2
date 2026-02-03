/// Detailed timing breakdown for a single circuit
use anyhow::Result;
use plonky2::field::types::Field;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

#[cfg(not(target_os = "macos"))]
use jemallocator::Jemalloc;

#[cfg(not(target_os = "macos"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main() -> Result<()> {
    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    let log_size = 20; // 2^20 = 1M gates
    println!("Circuit size: 2^{} = {} gates\n", log_size, 1 << log_size);

    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let num_ops = 1 << log_size;
    let initial = builder.add_virtual_target();
    let mut cur = initial;

    for i in 0..num_ops {
        let constant = builder.constant(F::from_canonical_u64((i % 1000) as u64 + 1));
        if i % 2 == 0 {
            cur = builder.mul(cur, constant);
        } else {
            cur = builder.add(cur, constant);
        }
    }

    builder.register_public_input(initial);
    builder.register_public_input(cur);

    println!("=== BUILD PHASE ===");
    let data = builder.build::<C>();

    println!("\n=== PROVE PHASE ===");
    let mut pw = PartialWitness::new();
    pw.set_target(initial, F::from_canonical_u64(12345));
    let proof = data.prove(pw)?;

    println!("\n=== VERIFY PHASE ===");
    data.verify(proof)?;

    Ok(())
}
