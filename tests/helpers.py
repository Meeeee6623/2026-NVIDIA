import itertools
import numpy as np


def calculate_energy(sequence: np.ndarray) -> int:
    """Compute LABS energy for a +/-1 sequence."""
    n = sequence.size
    energy = 0
    for k in range(1, n):
        ck = 0
        for i in range(n - k):
            ck += int(sequence[i]) * int(sequence[i + k])
        energy += ck * ck
    return int(energy)


def brute_force_min_energy(n: int) -> tuple[int, np.ndarray]:
    """Return minimum LABS energy and an example sequence achieving it."""
    best_energy = None
    best_seq = None
    for bits in itertools.product([-1, 1], repeat=n):
        seq = np.array(bits, dtype=np.int8)
        energy = calculate_energy(seq)
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_seq = seq
    return int(best_energy), best_seq
