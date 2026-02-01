# Ryzz Gate: Quantum-Accelerated Low Autocorrelation Binary Sequence Optimization

**Team Planck Scale** — Martin, Trent, Sanjeev, Benjamin, Joseph *(presented via huzz.vc)*

---

## 🎯 The Problem: Why This Matters

**Ryzz Gate** is our attempt to accelerate solving the **Low Autocorrelation Binary Sequence (LABS)** problem—an optimization task that seeks sequences with *minimal autocorrelation*, motivated by practical radar waveform considerations.

Brute force search scales exponentially (~2ⁿ), quickly becoming intractable. Our project explores whether a *quantum-inspired / quantum-enhanced* approach can improve the effective scaling relative to strong classical baselines, while still producing high-quality candidate sequences.

**Goal**: Find a binary sequence `s = [s₁, s₂, ..., sₙ]` where `sᵢ ∈ {-1, +1}` that **minimizes**:

```
E(s) = Σ (Cₖ)²  where  Cₖ = Σ sᵢ · sᵢ₊ₖ
```

---

## 🏗️ What We Built

### Phase 1 — Algorithmic Approach (Quantum-Enhanced Framing)

We model the LABS objective and explore a **CDQO / annealing-inspired** approach with an "impulse-regime" intuition, explicitly reasoning about the interplay between **Counter-Diabatic (CD)** and **Adiabatic (AD)** components.

We introduce a tunable **λ(t)** schedule designed to enter a CD-dominant regime early, then transition to a slower, AD-dominant finish.

### Phase 2 — Engineering for Throughput (Classical + GPU Acceleration)

We optimized the implementation to make large experiment sweeps feasible:
- Aggressive compilation choices
- Custom CUDA-Q kernels
- Deliberate thread allocation
- Profiling-driven tuning (Nsight-style workflow) until performance gains saturated

### Extending Beyond 1 Qubit : 1 Bit

To push past qubit-count limits, we also explored **Pauli Coefficient Encoding (PCE)**: instead of representing each bit directly, we encode information through the sign structure of selected Pauli-string observables (sampling from {X,Y,Z} to reduce redundancy), aiming for a higher effective bits-per-qubit ratio.

---

## 📁 Repository Structure

```
team-submissions/
├── labs_gpu/
│   ├── one_shot.py          # Quantum sampling (CUDA-Q kernels)
│   ├── one_shot.cu          # GPU-accelerated classical optimization (CUDA)
│   ├── tests.py             # Comprehensive test suite
│   └── warm_starts/         # Pre-computed warm start files
├── PCE encoding.ipynb       # VQE + PCE gradient descent approach
├── PRD.md                   # Product Requirements Document
├── AI_REPORT.md             # AI usage report
└── README.md                # ← You are here
```

---

## 🚀 Pipeline: Quantum → Bin → GPU

Our solution uses a **two-stage hybrid pipeline**:

### Stage 1: Quantum Sampling ([`one_shot.py`](labs_gpu/one_shot.py))

We use **CUDA-Q** to sample high-quality candidate sequences from quantum circuits implementing CD and AD schedules.

### Stage 2: Classical Refinement ([`one_shot.cu`](labs_gpu/one_shot.cu))

Quantum samples are exported to a **binary file (`.bin`)** which is loaded by a GPU-accelerated **Tabu Search** algorithm for refinement.

---

## 🔬 Quantum Approaches in `one_shot.py`

### Kernel Variants

| Variant | Description |
|---------|-------------|
| `kernel_default` | Baseline with full 2-body and 4-body ZZ interactions |
| `kernel_jenga` | Optimized default with cleaner gate stacking |
| `kernel_dna` | Hybrid kernel **interleaving** CD and AD steps via toggle arrays |
| `kernel_beyblade` | AD steps come **before** CD steps (alternating pattern) |
| `kernel_tensor_heavy` | Designed for tensor network simulation (`tensornet` target) |

### Lambda Schedules λ(t)

| Method | Formula | Regime |
|--------|---------|--------|
| `linear` | λ = t/T | Constant rate |
| `sqrt` | λ = √(t/T) | CD-dominant early, AD-dominant late |
| `cuberoot` | λ = ∛(t/T) | Even stronger early push |
| `trig` | λ = sin²(πt/2T) | Smooth S-curve transition |

---

## 📄 Why Binary Files (`.bin`)?

1. **Memory Efficiency**: `int8` storage (1 byte/spin vs 8 bytes/float)
2. **GPU-Optimized Layout**: Fixed stride `MAX_N = 512` for coalesced access
3. **Decoupled Pipeline**: Run quantum once, refine many times
4. **Warm Start Injection**: Partial loading with `load_warm_start_partial()`

---

## ⚡ GPU Optimizations in `one_shot.cu`

| Optimization | Impact |
|--------------|--------|
| **Shared Memory** | Entire sequence + autocorrelations in fast local memory |
| **Incremental ΔE** | O(N) per flip instead of O(N²) recomputation |
| **Warp Shuffles** | `__shfl_down_sync` for lock-free reduction |
| **Memetic Elitism** | Top 128 sequences preserved across generations |

---

## 📓 PCE/VQE Approach ([`PCE encoding.ipynb`](PCE%20encoding.ipynb))

Alternative approach using **Pauli Coefficient Encoding**:

- **Parameterized circuit** with trainable RY/RZ rotations + entangling layers
- **Tanh relaxation**: `x̃ = tanh(α · ⟨P⟩)` for differentiable optimization
- **Trade-off**: More flexible but requires iterative training vs. single-shot CD/AD

---

## 🏃 How to Run

```bash
# 1. Quantum Sampling
cd labs_gpu
python one_shot.py --n 64 --shots 10000 --variant dna --lambda_method trig --output warm_start.bin

# 2. Compile CUDA
nvcc -O3 -arch=sm_80 one_shot.cu -o labs_solver -lcurand

# 3. GPU Optimization
./labs_solver 64 8192 1000000 warm_start.bin
```

---

## 📊 Evaluation & Outcomes

We compare three approaches:
1. **Brute force** scaling (exponential baseline)
2. **Strong classical heuristics** (MTS-style scaling)
3. **Our quantum-enhanced variants** (CD/AD + PCE)

Metrics tracked: runtime, solution quality (energy), and throughput. Some regimes are bottlenecked by practical size limits (e.g., timeouts at larger N), which we report transparently.

**Key Results:**
- **Throughput**: >100M moves/second on A100 GPU
- **Scalability**: Tested up to N=512 with shared memory
- **Quantum Advantage**: Warm-started populations converge faster than random initialization

---

## 🤖 Responsible AI Note

We used AI tools primarily to accelerate documentation onboarding and conceptual lookup (e.g., CUDA-Q idioms), while keeping the *core implementation and correctness checks* human-verified. Where AI-produced edits or derivations failed validation, we reverted to verified formulations.

See [`AI_REPORT.md`](AI_REPORT.md) for detailed documentation.

---

## 📚 Further Reading

- **[PRD.md](PRD.md)** — Full product requirements and architecture decisions
- **[AI_REPORT.md](AI_REPORT.md)** — Documentation of AI-assisted development
- **[TESTS.md](labs_gpu/TESTS.md)** — Test coverage and verification methodology
