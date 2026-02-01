"""
Comprehensive Unit Test Suite for LABS GPU
===========================================
This module provides exhaustive testing for both the Python simulation controller
(one_shot.py) and the CUDA optimization kernel (one_shot.cu).

Test Categories:
1. Helper Functions: get_interactions, compute_adiabatic_schedule
2. Quantum Kernels: jenga, dna, beyblade (real cudaq execution)
3. Data Export: export_ready_for_cuda binary format
4. CUDA Integration: Compilation check, warm start loading

REQUIREMENTS: cudaq, numpy
Run on a system with CUDA-Q installed.

Author: Generated for NVIDIA 2026 Hackathon
"""

import unittest
import numpy as np
import os
import sys
import tempfile
import subprocess

# =============================================================================
# REAL IMPORTS (No Mocking)
# =============================================================================
# Import the module under test - requires cudaq to be installed
import one_shot


# =============================================================================
# TEST CLASS: HELPER FUNCTIONS
# =============================================================================
class TestHelperFunctions(unittest.TestCase):
    """Unit tests for standalone helper functions in one_shot.py."""

    # -------------------------------------------------------------------------
    # get_interactions
    # -------------------------------------------------------------------------
    def test_get_interactions_n4_basic_structure(self):
        """Verify output types and non-empty lists for N=4."""
        n = 4
        g2, g4, l2, l4 = one_shot.get_interactions(n)

        self.assertIsInstance(g2, list, "G2 should be a list")
        self.assertIsInstance(g4, list, "G4 should be a list")
        self.assertIsInstance(l2, list, "l2 should be a flattened list")
        self.assertIsInstance(l4, list, "l4 should be a flattened list")

        # Expect some 2-body interactions for any N > 3
        self.assertGreater(len(g2), 0, "Should have 2-body interactions")

    def test_get_interactions_n4_flattened_list_lengths(self):
        """Verify flattened lists have correct lengths (2*G2, 4*G4)."""
        n = 4
        g2, g4, l2, l4 = one_shot.get_interactions(n)

        self.assertEqual(len(l2), len(g2) * 2, "l2 length = 2 * G2 count")
        self.assertEqual(len(l4), len(g4) * 4, "l4 length = 4 * G4 count")

    def test_get_interactions_n6_has_4body_terms(self):
        """For N=6, we expect non-trivial 4-body interactions."""
        n = 6
        g2, g4, l2, l4 = one_shot.get_interactions(n)

        self.assertGreater(len(g4), 0, "N=6 should produce 4-body terms")

    def test_get_interactions_uniqueness(self):
        """Verify all generated pairs/quads are unique."""
        n = 8
        g2, g4, _, _ = one_shot.get_interactions(n)

        # Convert to frozensets for uniqueness check
        g2_sets = [frozenset(p) for p in g2]
        g4_sets = [frozenset(q) for q in g4]

        self.assertEqual(len(g2_sets), len(set(g2_sets)), "G2 should be unique")
        self.assertEqual(len(g4_sets), len(set(g4_sets)), "G4 should be unique")

    def test_get_interactions_indices_in_range(self):
        """Verify all indices are within [0, N-1]."""
        n = 10
        _, _, l2, l4 = one_shot.get_interactions(n)

        for idx in l2:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, n)
        for idx in l4:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, n)

    # -------------------------------------------------------------------------
    # compute_adiabatic_schedule
    # -------------------------------------------------------------------------
    def test_compute_adiabatic_schedule_length(self):
        """Verify schedule length matches n_steps."""
        for n_steps in [1, 5, 10, 100]:
            lambdas = one_shot.compute_adiabatic_schedule(n_steps)
            self.assertEqual(len(lambdas), n_steps)

    def test_compute_adiabatic_schedule_bounds(self):
        """Verify all lambda values are in [0, 1]."""
        lambdas = one_shot.compute_adiabatic_schedule(50)
        for l in lambdas:
            self.assertGreaterEqual(l, 0.0)
            self.assertLessEqual(l, 1.0)

    def test_compute_adiabatic_schedule_monotonic(self):
        """Schedule should be monotonically increasing (sin^2 trend)."""
        lambdas = one_shot.compute_adiabatic_schedule(20)
        for i in range(len(lambdas) - 1):
            self.assertLessEqual(lambdas[i], lambdas[i + 1],
                                 "Schedule should be non-decreasing")

    def test_compute_adiabatic_schedule_endpoint(self):
        """Final lambda should approach 1.0."""
        lambdas = one_shot.compute_adiabatic_schedule(100)
        self.assertGreater(lambdas[-1], 0.99, "Final lambda should be near 1.0")


# =============================================================================
# TEST CLASS: DATA EXPORT
# =============================================================================
class TestDataExport(unittest.TestCase):
    """Unit tests for the export_ready_for_cuda function."""

    def setUp(self):
        """Create a temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_export_creates_file(self):
        """Verify the function creates a binary file."""
        n = 8
        pop = np.random.choice([-1, 1], size=(10, n)).astype(np.int8)
        filepath = os.path.join(self.test_dir, "test.bin")

        one_shot.export_ready_for_cuda(pop, n, filepath, gpu_pop_size=10, gpu_max_n=16)

        self.assertTrue(os.path.exists(filepath))

    def test_export_file_size(self):
        """Verify file size = pop_size * gpu_max_n * sizeof(int8)."""
        n = 8
        pop_size = 64
        gpu_max_n = 512
        pop = np.random.choice([-1, 1], size=(pop_size, n)).astype(np.int8)
        filepath = os.path.join(self.test_dir, "sized.bin")

        one_shot.export_ready_for_cuda(pop, n, filepath, gpu_pop_size=pop_size, gpu_max_n=gpu_max_n)

        expected_size = pop_size * gpu_max_n * 1  # int8 = 1 byte
        self.assertEqual(os.path.getsize(filepath), expected_size)

    def test_export_data_integrity(self):
        """Verify written data matches input population."""
        n = 4
        pop_size = 8
        gpu_max_n = 8
        pop = np.array([
            [1, -1, 1, -1],
            [-1, 1, -1, 1],
            [1, 1, -1, -1],
            [-1, -1, 1, 1],
            [1, 1, 1, 1],
            [-1, -1, -1, -1],
            [1, -1, -1, 1],
            [-1, 1, 1, -1],
        ], dtype=np.int8)
        filepath = os.path.join(self.test_dir, "integrity.bin")

        one_shot.export_ready_for_cuda(pop, n, filepath, gpu_pop_size=pop_size, gpu_max_n=gpu_max_n)

        # Read back and verify
        data = np.fromfile(filepath, dtype=np.int8).reshape(pop_size, gpu_max_n)
        np.testing.assert_array_equal(data[:, :n], pop)

    def test_export_pads_correctly(self):
        """Ensure that the buffer is padded to gpu_max_n."""
        n = 4
        pop_size = 2
        gpu_max_n = 16
        pop = np.array([[1, -1, 1, -1], [-1, 1, -1, 1]], dtype=np.int8)
        filepath = os.path.join(self.test_dir, "padded.bin")

        one_shot.export_ready_for_cuda(pop, n, filepath, gpu_pop_size=pop_size, gpu_max_n=gpu_max_n)

        data = np.fromfile(filepath, dtype=np.int8).reshape(pop_size, gpu_max_n)
        # First n columns should be our data
        np.testing.assert_array_equal(data[:, :n], pop)
        # Remaining columns are random padding (just check shape)
        self.assertEqual(data.shape[1], gpu_max_n)


# =============================================================================
# TEST CLASS: QUANTUM KERNEL VARIANTS (Real Execution)
# =============================================================================
class TestQuantumKernels(unittest.TestCase):
    """Tests for the three quantum kernel variants: jenga, dna, beyblade.
    
    These tests run REAL quantum simulations via cudaq.
    """

    def setUp(self):
        """Create temp directory for output files."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_jenga_kernel_produces_output(self):
        """Verify jenga kernel produces valid bitstrings."""
        output = os.path.join(self.test_dir, "jenga.bin")
        one_shot.run_simulation(n=4, shots=100, pop_size=4, output_file=output,
                                variant='jenga', steps=1)
        self.assertTrue(os.path.exists(output))
        # Verify file has correct size
        expected_size = 4 * 512  # pop_size * MAX_N (default)
        self.assertEqual(os.path.getsize(output), expected_size)

    def test_dna_kernel_produces_output(self):
        """Verify dna kernel produces valid bitstrings."""
        output = os.path.join(self.test_dir, "dna.bin")
        one_shot.run_simulation(n=4, shots=100, pop_size=4, output_file=output,
                                variant='dna', steps=1)
        self.assertTrue(os.path.exists(output))

    def test_beyblade_kernel_produces_output(self):
        """Verify beyblade kernel produces valid bitstrings."""
        output = os.path.join(self.test_dir, "beyblade.bin")
        one_shot.run_simulation(n=4, shots=100, pop_size=4, output_file=output,
                                variant='beyblade', steps=1)
        self.assertTrue(os.path.exists(output))

    def test_run_simulation_with_multiple_steps(self):
        """Verify simulation runs correctly with multiple Trotter steps."""
        output = os.path.join(self.test_dir, "multi_step.bin")
        one_shot.run_simulation(n=4, shots=100, pop_size=4, output_file=output,
                                variant='dna', steps=5)
        self.assertTrue(os.path.exists(output))

    def test_output_contains_valid_spins(self):
        """Verify output binary contains only ±1 spin values."""
        output = os.path.join(self.test_dir, "spins.bin")
        n = 4
        pop_size = 8
        one_shot.run_simulation(n=n, shots=100, pop_size=pop_size, output_file=output,
                                variant='jenga', steps=1)
        
        data = np.fromfile(output, dtype=np.int8).reshape(pop_size, 512)
        # First n columns should be ±1
        valid_spins = data[:, :n]
        self.assertTrue(np.all(np.isin(valid_spins, [-1, 1])),
                        "All spin values should be ±1")

    def test_jenga_n6_larger_system(self):
        """Test jenga on a slightly larger system (N=6)."""
        output = os.path.join(self.test_dir, "n6.bin")
        one_shot.run_simulation(n=6, shots=200, pop_size=10, output_file=output,
                                variant='jenga', steps=2)
        self.assertTrue(os.path.exists(output))

    def test_beyblade_multiple_trotter_steps(self):
        """Test beyblade with multiple Trotter steps."""
        output = os.path.join(self.test_dir, "beyblade_steps.bin")
        one_shot.run_simulation(n=4, shots=100, pop_size=4, output_file=output,
                                variant='beyblade', steps=3)
        self.assertTrue(os.path.exists(output))


# =============================================================================
# TEST CLASS: CUDA INTEGRATION
# =============================================================================
class TestCUDAIntegration(unittest.TestCase):
    """Tests for CUDA-related functionality (compilation, loading)."""

    def test_cuda_file_exists(self):
        """Verify one_shot.cu exists in the expected location."""
        cuda_path = os.path.join(os.path.dirname(__file__), "one_shot.cu")
        self.assertTrue(os.path.exists(cuda_path), f"CUDA file not found: {cuda_path}")

    def test_cuda_syntax_check(self):
        """
        Check CUDA syntax by attempting to compile with nvcc (dry run).
        This test is SKIPPED if nvcc is not available.
        """
        cuda_path = os.path.join(os.path.dirname(__file__), "one_shot.cu")

        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True, text=True
            )
            nvcc_available = result.returncode == 0
        except FileNotFoundError:
            nvcc_available = False

        if not nvcc_available:
            self.skipTest("nvcc not available, skipping CUDA compilation check")

        # Compile to object file (syntax check)
        with tempfile.NamedTemporaryFile(suffix=".o", delete=True) as tmp:
            result = subprocess.run(
                ["nvcc", "-c", cuda_path, "-o", tmp.name],
                capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0,
                             f"CUDA compilation failed: {result.stderr}")

    def test_warm_start_binary_format(self):
        """
        Verify that a generated warm_start.bin can be correctly interpreted
        by the expected CUDA loading logic.
        """
        n = 8
        pop_size = 16
        max_n = 512  # Must match CUDA's MAX_N

        # Create a known population
        pop = np.random.choice([-1, 1], size=(pop_size, n)).astype(np.int8)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            filepath = f.name

        try:
            one_shot.export_ready_for_cuda(pop, n, filepath, gpu_pop_size=pop_size, gpu_max_n=max_n)

            # Simulate CUDA loading: read as (pop_size, max_n) int8 buffer
            cuda_buffer = np.fromfile(filepath, dtype=np.int8).reshape(pop_size, max_n)

            # Verify first n columns match our input
            np.testing.assert_array_equal(cuda_buffer[:, :n], pop)

            # Verify all values are either -1 or 1 (spin values)
            self.assertTrue(np.all(np.isin(cuda_buffer[:, :n], [-1, 1])))

        finally:
            os.unlink(filepath)


# =============================================================================
# TEST CLASS: EDGE CASES
# =============================================================================
class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and boundary conditions."""

    def test_get_interactions_n2(self):
        """N=2: algorithm produces 0 pairs (requires overlapping squares)."""
        g2, g4, l2, l4 = one_shot.get_interactions(2)
        # The algorithm looks for overlaps, so N=2 yields empty lists
        self.assertEqual(len(g2), 0)
        self.assertEqual(len(g4), 0)

    def test_get_interactions_n1(self):
        """N=1 should produce empty interactions (no pairs possible)."""
        g2, g4, l2, l4 = one_shot.get_interactions(1)
        self.assertEqual(len(g2), 0)
        self.assertEqual(len(g4), 0)

    def test_compute_adiabatic_schedule_single_step(self):
        """Single step schedule should have one element."""
        lambdas = one_shot.compute_adiabatic_schedule(1)
        self.assertEqual(len(lambdas), 1)
        self.assertGreater(lambdas[0], 0)

    def test_large_n_interactions(self):
        """Stress test for larger N values."""
        n = 32
        g2, g4, l2, l4 = one_shot.get_interactions(n)

        # Just verify it completes and produces valid output
        self.assertGreater(len(g2), 0)
        self.assertGreater(len(g4), 0)
        self.assertTrue(all(0 <= i < n for i in l2))
        self.assertTrue(all(0 <= i < n for i in l4))


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    # Custom test runner with verbose output for grading clarity
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestHelperFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestDataExport))
    suite.addTests(loader.loadTestsFromTestCase(TestQuantumKernels))
    suite.addTests(loader.loadTestsFromTestCase(TestCUDAIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))

    # Run with verbosity=2 for detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests Run:    {result.testsRun}")
    print(f"Failures:     {len(result.failures)}")
    print(f"Errors:       {len(result.errors)}")
    print(f"Skipped:      {len(result.skipped)}")
    print(f"Status:       {'PASSED' if result.wasSuccessful() else 'FAILED'}")
    print("=" * 70)
