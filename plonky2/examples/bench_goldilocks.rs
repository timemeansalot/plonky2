/// E2E benchmark for Goldilocks field
use anyhow::Result;
use plonky2::field::types::Field;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use std::time::Instant;

fn main() -> Result<()> {
    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    println!("=== PoseidonGoldilocksConfig (64-bit field) ===\n");

    for log_size in 13..=20 {
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

        let build_start = Instant::now();
        let data = builder.build::<C>();
        let build_time = build_start.elapsed();

        let mut pw = PartialWitness::new();
        pw.set_target(initial, F::from_canonical_u64(12345));

        let prove_start = Instant::now();
        let proof = data.prove(pw)?;
        let prove_time = prove_start.elapsed();

        let verify_start = Instant::now();
        data.verify(proof)?;
        let verify_time = verify_start.elapsed();

        println!(
            "2^{:2} ({:7} gates): Build {:>8.2?} | Prove {:>8.2?} | Verify {:>8.2?} | Total {:>8.2?}",
            log_size,
            1 << log_size,
            build_time,
            prove_time,
            verify_time,
            build_time + prove_time + verify_time
        );
    }

    Ok(())
}
