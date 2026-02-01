import re
from pathlib import Path

import numpy as np

from tests.helpers import brute_force_min_energy, calculate_energy

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_targets():
    labs_file = REPO_ROOT / "labs_gpu" / "labs_gpu.cu"
    content = labs_file.read_text()
    match = re.search(r"const int TARGETS\[] = \{([^}]*)\};", content, re.S)
    if not match:
        raise AssertionError("TARGETS table not found in labs_gpu.cu")
    numbers = [int(x.strip()) for x in match.group(1).split(",") if x.strip()]
    return numbers


def test_energy_symmetry_variants():
    seq = np.array([1, -1, 1, 1, -1, 1], dtype=np.int8)
    energy = calculate_energy(seq)
    assert energy == calculate_energy(seq[::-1])
    assert energy == calculate_energy(-seq)


def test_energy_known_sequence_value():
    seq = np.array([1, 1, 1, 1], dtype=np.int8)
    assert calculate_energy(seq) == 14


def test_bruteforce_minima_match_targets_small_n():
    targets = load_targets()
    for n in range(2, 9):
        min_energy, _ = brute_force_min_energy(n)
        assert min_energy == targets[n]
