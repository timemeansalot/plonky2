# Metal Migration Draft

Migration plan for porting CUDA GPU acceleration to Apple Metal.

## Migration Strategy

**Order**: Hash functions first, then NTT/LDE

---

## Phase 1: Hash Functions

### 1.1 Poseidon Hash (Goldilocks field)
- [ ] Create Metal shader for Goldilocks field arithmetic (add, mul, reduce)
- [ ] Implement Poseidon permutation kernel (full rounds + partial rounds)
- [ ] Implement MDS matrix multiplication
- [ ] Add batch hashing support (multiple inputs in parallel)
- [ ] Integrate with `plonky2/src/hash/poseidon.rs`

### 1.2 Poseidon2 Hash
- [ ] Implement Poseidon2 permutation (different round constants/structure)
- [ ] Integrate with `plonky2/src/hash/poseidon2.rs`

### 1.3 PoseidonBN128 Hash (254-bit field)
- [ ] Create Metal shader for BN128 field arithmetic (multi-limb)
- [ ] Implement Poseidon permutation for BN128
- [ ] Integrate with `plonky2/src/hash/poseidon_bn128.rs`

### 1.4 Merkle Tree Building
- [ ] Implement parallel leaf hashing kernel
- [ ] Implement tree reduction kernel (bottom-up hashing)
- [ ] Integrate with `plonky2/src/hash/merkle_tree.rs`

---

## Phase 2: NTT/LDE

### 2.1 NTT (Number Theoretic Transform)
- [ ] Implement butterfly operations for Goldilocks field
- [ ] Implement radix-2 NTT kernel
- [ ] Add twiddle factor precomputation
- [ ] Support forward and inverse NTT

### 2.2 LDE (Low Degree Extension)
- [ ] Implement coset FFT
- [ ] Implement batch LDE (multiple polynomials)
- [ ] Keep output on GPU option (for Merkle tree input)
- [ ] Integrate with `plonky2/src/fri/oracle.rs`

---

## Phase 3: Integration

- [ ] Create `--features=metal` flag in Cargo.toml
- [ ] Add Metal device initialization
- [ ] Create Rust bindings (metal-rs or objc crate)
- [ ] Add runtime GPU/CPU selection
- [ ] Benchmarking and optimization

---

## Reference Files

| Component | CUDA Reference | Metal Target |
|-----------|---------------|--------------|
| Field ops | `zeknox/native/goldilocks/` | New Metal shaders |
| Poseidon | `zeknox/native/poseidon/` | New Metal shaders |
| NTT | `zeknox/native/ntt/` | New Metal shaders |
| Merkle | `zeknox/native/merkle/` | New Metal shaders |
| Existing demo | - | `plonky2-metal-demo/` |

---

## Estimated Complexity

| Component | Lines of CUDA | Difficulty |
|-----------|---------------|------------|
| Goldilocks field | ~500 | Low |
| Poseidon hash | ~1,500 | Medium |
| Poseidon2 | ~800 | Medium |
| PoseidonBN128 | ~2,000 | High (254-bit) |
| NTT | ~3,000 | High |
| Merkle tree | ~1,000 | Medium |

---

## Technical Notes

### Goldilocks Field
- Prime: `p = 2^64 - 2^32 + 1`
- Fits in 64-bit, but reduction needs careful handling
- Metal supports 64-bit integers natively

### Poseidon Parameters
- State width: 12 elements
- Full rounds: 8
- Partial rounds: 22
- MDS matrix: 12x12

### BN128 Field (for PoseidonBN128)
- 254-bit prime field
- Requires 4x 64-bit limbs
- More complex reduction logic

### NTT Considerations
- Radix-2 Cooley-Tukey algorithm
- Twiddle factors can be precomputed
- Batch processing for multiple polynomials
- Consider threadgroup memory for intermediate values

---

## Rust Crates for Metal

- `metal` - Official Apple Metal bindings
- `objc` - Objective-C runtime bindings
- `block` - Objective-C blocks support

Example setup:
```toml
[dependencies]
metal = "0.27"
objc = "0.2"
```

---

## Expected Performance

Based on CUDA benchmarks and Metal characteristics:

| Operation | CUDA Speedup | Expected Metal Speedup |
|-----------|--------------|------------------------|
| Merkle Tree | 4-44x | 2-20x |
| LDE (on GPU) | 10-34x | 5-15x |
| E2E Proving | 1.2-2.6x | 1.0-1.5x |

Note: Metal has fewer GPU cores than NVIDIA GPUs, so absolute speedup will be lower.
