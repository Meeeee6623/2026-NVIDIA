# LABS GPU: Quantum-Accelerated Low Autocorrelation Binary Sequence Optimization

This directory contains our hybrid quantum-classical solution for the **Low Autocorrelation Binary Sequence (LABS)** problem, accelerated using CUDA-Q and CUDA.

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

## 🧠 The Problem: LABS

The goal is to find a binary sequence `s = [s₁, s₂, ..., sₙ]` where `sᵢ ∈ {-1, +1}` that **minimizes** the energy function:

```
E(s) = Σ (Cₖ)²  where  Cₖ = Σ sᵢ · sᵢ₊ₖ
```

This is an NP-hard combinatorial optimization problem with applications in signal processing, radar systems, and communications.

---

## 🚀 Our Approach: Quantum → Bin → GPU

Our solution uses a **two-stage hybrid pipeline**:

### Stage 1: Quantum Sampling ([`one_shot.py`](labs_gpu/one_shot.py))

We use **CUDA-Q** to sample high-quality candidate sequences from quantum circuits. The quantum circuits implement **Counter-Diabatic (CD)** and **Adiabatic (AD)** schedules to guide the system toward low-energy states.

### Stage 2: Classical Refinement ([`one_shot.cu`](labs_gpu/one_shot.cu))

The quantum samples are exported to a **binary file (`.bin`)** which is then loaded by a GPU-accelerated **Tabu Search** algorithm. The GPU solver uses massive parallelism to refine the quantum-generated candidates into optimal solutions.

---

## 🔬 Quantum Approaches in `one_shot.py`

The quantum code implements **5 different kernel variants**, each with unique trade-offs:

### 1. `kernel_default`
- **Description**: Baseline implementation with full 2-body and 4-body ZZ interaction terms.
- **Use Case**: Reference implementation for correctness validation.

### 2. `kernel_jenga`
- **Description**: Optimized version of default with identical physics but cleaner code structure.
- **Named for**: The careful "stacking" of gate operations like Jenga blocks.

### 3. `kernel_dna`
- **Description**: Hybrid kernel that **interleaves** Counter-Diabatic (CD) and Adiabatic (AD) steps.
- **Key Feature**: Uses boolean toggle arrays (`CD[]`, `AD[]`) to dynamically switch between evolution modes.
- **Physics**: Combines the precision of CD driving with the robustness of AD evolution.

### 4. `kernel_beyblade`
- **Description**: Similar to DNA but with a different interleaving pattern - AD steps come **before** CD steps within each iteration.
- **Named for**: The "spinning" alternation between evolution types.

### 5. `kernel_tensor_heavy`
- **Description**: Designed for **tensor network simulation** (`cudaq.set_target("tensornet")`).
- **Use Case**: Scales to larger N by exploiting tensor contraction instead of state vector simulation.
- **Trade-off**: Requires more GPU memory but can handle deeper circuits.

### Lambda Scheduling Methods

The adiabatic schedule `λ(t)` controls how the Hamiltonian evolves from the mixer to the problem Hamiltonian:

| Method | Formula | Characteristic |
|--------|---------|----------------|
| `linear` | `λ = t/T` | Constant rate |
| `sqrt` | `λ = √(t/T)` | Fast start, slow finish |
| `cuberoot` | `λ = ∛(t/T)` | Even faster start |
| `trig` | `λ = sin²(πt/2T)` | Smooth S-curve |

---

## 📄 Why Binary Files (`.bin`)?

The quantum sampler exports candidates to a binary file for several key reasons:

1. **Memory Efficiency**: Binary format stores sequences as `int8` (-1/+1 values), using only 1 byte per spin instead of 8 bytes for floats.

2. **GPU-Optimized Layout**: The data is pre-formatted with:
   - **Fixed row stride** of `MAX_N = 512` for coalesced memory access
   - **Consistent population size** for predictable kernel launches

3. **Decoupled Pipeline**: Allows the quantum and classical stages to run independently:
   - Run quantum sampling once, refine many times
   - Experiment with different classical solvers on the same quantum data

4. **Warm Start Injection**: The classical solver can **partially load** quantum candidates and fill remaining slots with random sequences. This is the `load_warm_start_partial()` function in the CUDA code.

---

## ⚡ GPU Optimizations in `one_shot.cu`

The CUDA code is heavily optimized for A100/A6000 GPUs:

### Shared Memory Architecture

```cpp
__shared__ int8_t s_seq[MAX_N];      // Current sequence
__shared__ int8_t s_best_seq[MAX_N]; // Best found
__shared__ int s_corr[MAX_N];        // Autocorrelations
__shared__ int s_tabu[MAX_N];        // Tabu tenure array
```

**Why?** Shared memory is ~100x faster than global memory. The entire sequence and its autocorrelations fit in shared memory, enabling in-place modification without global memory round-trips.

### Incremental Energy Updates

Instead of recomputing `E(s)` from scratch after each flip:

```cpp
for (int k = 1; k < N; k++) {
    int nb = 0;
    if (i + k < N) nb += s_seq[i + k];
    if (i - k >= 0) nb += s_seq[i - k];
    int dCk = -2 * s_p * nb;
    dE += (long long)dCk * (2 * s_corr[k] + dCk);
}
```

This computes the **delta energy** for flipping position `i` using only O(N) operations instead of O(N²).

### Warp-Level Reduction

```cpp
__device__ __forceinline__ void warpReduceMin(long long &val, int &idx) {
    for (int offset = 16; offset > 0; offset /= 2) {
        long long o_val = __shfl_down_sync(0xFFFFFFFF, val, offset);
        int o_idx = __shfl_down_sync(0xFFFFFFFF, idx, offset);
        if (o_val < val) { val = o_val; idx = o_idx; }
    }
}
```

Uses **warp shuffle intrinsics** to find the best flip position across all threads without expensive atomic operations.

### Memetic Algorithm with Elitism

Every `MEMETIC_FREQ` iterations:
1. Sort population by energy
2. Keep top `ELITE_COUNT` sequences
3. Generate new candidates via crossover + mutation
4. Inject diversity while preserving best solutions

---

## 📓 VQE + PCE Approach ([`PCE encoding.ipynb`](PCE%20encoding.ipynb))

This notebook explores an **alternative quantum approach** using:

### Parameterized CE (PCE) Encoding

Instead of directly encoding the LABS problem, we use a **variational quantum circuit** with trainable parameters:

```python
for i in range(n):
    ry(params[i], qubits[i])        # Single-qubit rotations
    rz(params[i+n], qubits[i])
cx(qubits[i], qubits[i+1])          # Entangling layer
rz(params[...], qubits[i+1])        # Parameterized ZZ interaction
```

### Gradient Descent Optimization

The loss function uses a **tanh relaxation** to convert continuous expectation values to approximate binary values:

```python
x_tilde = [np.tanh(alpha * exp) for exp in expectation_values]
```

This makes the energy landscape differentiable for gradient-based optimization.

### Trade-offs

| Aspect | PCE/VQE | Counter-Diabatic |
|--------|---------|------------------|
| Flexibility | High (many parameters) | Fixed schedule |
| Training | Requires many iterations | Single shot |
| Noise Resilience | Better (shallow circuits) | Worse (deep circuits) |
| Implementation | More complex | Simpler |

---

## 🏃 How to Run

### 1. Quantum Sampling (Generate Warm Start)

```bash
cd labs_gpu
python one_shot.py --n 64 --shots 10000 --variant dna --lambda_method trig --output warm_start.bin
```

**Arguments:**
- `--n`: Sequence length
- `--shots`: Number of quantum samples
- `--variant`: Kernel choice (`jenga`, `dna`, `beyblade`, `tensor_heavy`)
- `--lambda_method`: Adiabatic schedule (`linear`, `sqrt`, `cuberoot`, `trig`)
- `--output`: Binary output file

### 2. CUDA Compilation

```bash
nvcc -O3 -arch=sm_80 one_shot.cu -o labs_solver -lcurand
```

### 3. GPU Optimization

```bash
./labs_solver 64 8192 1000000 warm_start.bin
```

**Arguments:**
1. `N`: Sequence length
2. `POP_SIZE`: Population size
3. `MAX_ITERS`: Maximum iterations
4. `WARM_START_FILE`: (Optional) Binary file from quantum sampling

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
cd labs_gpu
python tests.py
```

See [`TESTS.md`](labs_gpu/TESTS.md) for detailed test documentation.

---

## 📊 Key Results

- **Throughput**: >100M moves/second on A100 GPU
- **Scalability**: Tested up to N=512 with shared memory
- **Quantum Advantage**: Warm-started populations converge faster than random initialization

---

## 📚 Further Reading

- **PRD.md**: Full product requirements and architecture decisions
- **AI_REPORT.md**: Documentation of AI-assisted development
- **TESTS.md**: Test coverage and verification methodology
