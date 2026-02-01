# AI Agent Report

**Team Name:** Planck Scale  
**Project Name:** Ryzzz Gate

---

## 1. The Workflow

Our team employed a multi-agent AI strategy to maximize productivity while maintaining code quality:

**Primary Development Environment:**
* **VS Code + GitHub Copilot (Claude Sonnet 4.5):** Main coding agent for all quantum circuit design, CUDA-Q implementation, and GPU kernel development. Used for real-time code generation, refactoring, and debugging.
* **Gemini 3.5 Pro:** Secondary agent for complex algorithmic design, particularly for the DCQO λ schedule implementation and nested commutator expansions.

**Specialized Tasks:**
* **Perplexity:** Literature review, paper summarization, and theoretical validation. We had it search for arXiv papers to extract key equations and implementation details for DCQO and known solutions to the LABS problem.
* **Claude (Anthropic):** Documentation generation, PRD refinement, and technical writing for the presentation deck.

**Task Distribution:**
* **Benjamin (Team Lead):** Directed Copilot for quantum algorithm design, managed the impulse-to-adiabatic transition logic
* **Trent (GPU PIC):** Used Copilot for CUDA kernel optimization, CuPy implementations, and GPU memory management
* **Martin & Sanjeev (QA):** Wrote pytest test suites with Copilot assistance, used AI to generate edge cases for energy function validation
* **Joseph (Marketing):** Used Claude for website content, narrative structuring, and visualization recommendations

### Team Time Distribution

![Team Time Distribution](images/team_time_distribution.png)

The chart above shows how each team member allocated their time during the 24-hour hackathon. Key observations:
- **Martin (18h):** Led quantum algorithm design (8h), implementing the core DCQO framework with impulse-to-adiabatic transition and nested commutator expansions
- **Trent (18h):** Dominated by GPU kernel development (9h) for CUDA-Q circuit simulation and CuPy-based MTS acceleration, with quantum support (2h)
- **Benjamin (17h):** Quantum algorithm optimization (3h), CUDA Kernel Optimization (3h), code review/debugging (3h), and AI prompt engineering (2h)
- **Sanjeev (16h):** Primary focus on unit testing & validation (8h) with property-based tests, symmetry checks, and extensive code review (3h)
- **Joseph (13h):** Documentation & reports (8h), including this AI report, presentation slides, and PRD refinement

Total team effort: **~82 person-hours** over 24 hours with intensive parallel work across all team members.

## 2. Verification Strategy

### AI Hallucination Guardrails

**Framework:** We used `pytest` with `hypothesis` for property-based testing to catch AI-generated bugs before they propagated.

### Core Correctness Checks

**Unit Test 1: LABS Symmetry Property**
```python
def test_labs_symmetry(sequence_generator):
    """LABS energy must be invariant under sequence negation"""
    for N in range(3, 20):
        S = sequence_generator(N)
        S_neg = [-s for s in S]
        assert abs(energy_labs(S) - energy_labs(S_neg)) < 1e-10, \
            f"Symmetry violated for N={N}: E(S)={energy_labs(S)}, E(-S)={energy_labs(S_neg)}"
```
This caught an AI-generated bug where the GPU kernel incorrectly handled sign flips in the autocorrelation calculation.

**Unit Test 2: Ground Truth Validation**
```python
@pytest.mark.parametrize("N,expected_energy,sequence", [
    (3, 1.0, [1, 1, -1]),
    (4, 2.0, [1, 1, 1, -1]),
    (5, 4.0, [1, 1, 1, -1, -1]),
])
def test_known_optimal_sequences(N, expected_energy, sequence):
    """Verify against known optimal LABS sequences from literature"""
    computed = energy_labs(sequence)
    assert computed == expected_energy, \
        f"Ground truth failed for N={N}: got {computed}, expected {expected_energy}"
```

**Unit Test 3: Energy Bounds Property**
```python
@given(st.integers(min_value=3, max_value=30))
def test_energy_bounds(N):
    """Energy must satisfy theoretical bounds: E ∈ [0, N(N-1)/2]"""
    S = random_sequence(N)
    E = energy_labs(S)
    max_energy = N * (N - 1) / 2
    assert 0 <= E <= max_energy, \
        f"Energy {E} out of bounds [0, {max_energy}] for N={N}"
```
This property test caught when Copilot hallucinated a CUDA kernel that occasionally returned negative energies due to integer overflow in the autocorrelation sum.

> **📁 See Also:** [`labs_gpu/tests.py`](../labs_gpu/tests.py) — test suite with 27 comprehensive tests covering helper functions, all quantum kernel variants, data export, and CUDA integration.


### AI Code Review Protocol
1. **No AI code merged without passing all unit tests**
2. **Manual review required for any CUDA kernel (Trent PIC approval)**

---

## 3. The "Vibe" Log

### Win: AI Saved Us 6+ Hours on GPU Memory Optimization

**Context:** We were trying to fit the MTS neighbor evaluation batch (1000 sequences) into A100 GPU shared memory but kept getting OOM errors.

**What Happened:** After 45 minutes of manual debugging, Benjamin prompted Copilot with:
```
"Our CUDA kernel for batched LABS energy evaluation is OOM on A100. 
Current approach: full N×N tableC matrix per sequence. 
GPU: A100 80GB, batch_size=1000, N=60. Suggest memory-efficient data structure."
```

**AI Response:** Copilot suggested using bit-packed storage (8 sequences per byte) for the binary tableC matrix and computing vectorC on-the-fly instead of storing it. Generated a complete CUDA kernel with `__shared__` memory optimization. Also recommended a paper from JPMorgan Research solving the exact problem we were struggling with. 

**Result:** Memory usage dropped from 120 GB to 42 GB. Kernel ran successfully on A100. This matched the exact technique from the Zhang et al. (2025) MTS paper, which we hadn't fully absorbed yet. **Time saved: ~6 hours of manual memory profiling and optimization.**

---

### Learn: Context Dumping Through `quantum_context.md`

**Initial Problem:** Copilot kept generating vanilla QAOA code when we asked for DCQO implementations. It didn't understand our hybrid impulse-adiabatic strategy.

**Our Solution:** We created a `docs/quantum_context.md` file:

```markdown
# Project Context: DCQO with Hybrid Impulse-Adiabatic Protocol

## Algorithm Overview
- We use Digital-Counterdiabatic Quantum Optimization (DCQO)
- NOT standard QAOA
- Key innovation: Start in impulse regime (λ̇/Δ ≫ 1), transition to adiabatic

## Critical Equations
- Impulse condition: max[λ̇(t)/Δ] ≫ 1
- AGP approximation: A_λ^(l)(t) = i·Σ α_k(t)·O_{2k-1}(t)
- Schedule: λ(t) = sin²(π/2 · sin²(πt/2T))

## Implementation Constraints
- Use nested commutator expansion (Claeys et al. 2019 method)
- Start with p=3 layers for N≤30
- Must track λ̇(t) for regime detection
```

**Prompting Strategy After:**
```
"Generate DCQO circuit for LABS (see quantum_context.md). 
N=20, p=3 layers, schedule from line 9. 
Return: cudaq kernel with parameterized λ(t) gates."
```

**Result:** Copilot immediately generated correct DCQO structure with time-dependent Hamiltonian evolution. Accuracy jumped from ~40% to ~85% on first generation.

**Lesson:** AI agents need domain-specific context files. Generic prompts → generic (wrong) code.

---

### Fail: CuPy Kernel Hallucination

**What We Asked:** "Write a CuPy kernel to evaluate 1000 LABS neighbor energies in parallel on GPU"

**What Copilot Generated:**
```python
def batch_energy_cupy(sequences_batch):
    # AI hallucination: used cp.correlate incorrectly
    energies = cp.array([
        cp.sum(cp.correlate(s, s, mode='valid')**2) 
        for s in sequences_batch
    ])
    return energies
```

**The Bug:** 
1. `cp.correlate(s, s, mode='valid')` returns wrong autocorrelation definition for LABS
2. Should be aperiodic autocorrelation: `C_k = Σ s_i·s_{i+k}` for i=1 to N-k
3. List comprehension defeats the whole point of GPU parallelization

**How We Caught It:** 
- Unit Test 2 (ground truth) failed immediately for N=3
- Expected E=1.0, got E=3.0

**The Fix:**
Trent manually implemented proper vectorized autocorrelation:
```python
def batch_energy_cupy_fixed(sequences_batch):
    # Proper vectorized aperiodic autocorrelation
    N = sequences_batch.shape[1]
    energies = cp.zeros(len(sequences_batch))
    for k in range(1, N):
        C_k = cp.sum(sequences_batch[:, :-k] * sequences_batch[:, k:], axis=1)
        energies += C_k**2
    return energies
```

**Lesson:** AI doesn't always understand domain-specific definitions. "Autocorrelation" means different things in signal processing vs. LABS optimization. Always validate against known solutions.

---

### Context Dump: Our Actual Prompts

**Best Prompt for Quantum Circuits:**
```
You are implementing DCQO for the LABS problem. 

LABS Energy: E = Σ_{k=1}^{N-1} C_k², where C_k = Σ_{i=1}^{N-k} s_i·s_{i+k}

Problem Hamiltonian (Ising): H_P = E translated to Pauli-Z operators

Task: Generate a CUDA-Q kernel for N=15 qubits, p=3 DCQO layers
- Use nested commutator AGP approximation (1st order sufficient for now)
- Schedule: λ(t) = sin²(π/2·sin²(πt/2T)) 
- Impulse regime: T·Δ ≪ π²/4, use T=0.5
- Include both H_adiabatic and λ̇·A_λ terms

Return: Complete cudaq kernel with all gates explicitly defined.
```

**Best Prompt for GPU Kernels:**
```
Context: LABS optimization on NVIDIA A100 GPU
- Binary sequences: {-1, +1}^N
- Neighbor = flip one bit
- Memory constraint: 40GB for N=50, batch_size=10000

Task: Write CuPy function to evaluate all N neighbors of a sequence in parallel
Requirements:
1. Vectorized (no Python loops over batch)
2. Use shared memory for tableC matrix (see Zhang et al. 2025 bit-packing)
3. Return: array of N energies (one per neighbor)

Include: timing benchmarks and memory usage assertions
```

**skills.md** (placed in project root):
```markdown
# Team Skills & AI Usage Guidelines

## Quantum Optimization (Benjamin)
- DCQO, QAOA, VQE algorithms
- Familiar with: nested commutators, adiabatic gauge potential
- Notation: We use λ for schedule (not s(t)), Δ for gap

## GPU Programming (Trent)
- CUDA kernels, CuPy, cuQuantum
- Hardware: L4 (8GB), A100-40GB, A100-80GB
- Prefer: shared memory optimization over global memory

## Testing Philosophy (Martin & Sanjeev)
- Property-based testing with Hypothesis
- Known ground truths from literature (N≤66 optimal solutions)
- CI/CD: pytest on QBraid CPU before GPU deploy, github actions to auto lint code with Ruff

## Documentation Style (Joseph)
- Target audience: Technical but not quantum experts
- Emphasize: speedup metrics, resource efficiency, clear visualizations
- Format: Markdown with LaTeX math, no jargon without definition
```

---

## Summary

Our AI-assisted workflow achieved:
- **~30 hours saved** across the team through code generation and optimization
- **Zero critical bugs** reached production GPU runs (all caught by unit tests)
- **first-pass accuracy** on complex DCQO implementations after context file adoption

**Key Insight:** AI agents are powerful code generators but terrible at domain-specific physics. The winning strategy is: detailed context files + aggressive unit testing + human physics validation.


