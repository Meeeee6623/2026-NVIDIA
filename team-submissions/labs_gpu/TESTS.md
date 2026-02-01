# LABS GPU Test Suite

**Production-grade, comprehensive unit and integration tests** for the quantum-classical LABS solver.

## Requirements
- **CUDA-Q** (`cudaq`) with simulator backend
- **NumPy**
- **nvcc** (optional, for CUDA compilation tests)

## Run
```bash
python3 tests.py
```

## Test Coverage (27 Tests)

| Category | Tests | Coverage |
|----------|:-----:|----------|
| **Helper Functions** | 9 | `get_interactions` topology, `compute_adiabatic_schedule` bounds/monotonicity |
| **Quantum Kernels** | 7 | All 3 variants (jenga, dna, beyblade), multi-step Trotter, spin validation |
| **Data Export** | 4 | Binary format, file size, data integrity, GPU buffer padding |
| **CUDA Integration** | 3 | Source verification, `nvcc` compilation, warm start compatibility |
| **Edge Cases** | 4 | Boundary conditions (N=1,2), single-step schedule, stress test (N=32) |

## Design Principles

- **Isolation**: Each test is independent with proper setup/teardown
- **Deterministic**: Verifiable assertions on known mathematical properties
- **CI/CD Ready**: Structured output, clear pass/fail summary
- **Real Execution**: No mocking — tests run actual quantum circuits via CUDA-Q

## Expected Output
```
----------------------------------------------------------------------
Ran 27 tests in X.XXXs

OK

======================================================================
TEST SUMMARY
======================================================================
Tests Run:    27
Failures:     0
Errors:       0
Skipped:      0
Status:       PASSED
======================================================================
```
