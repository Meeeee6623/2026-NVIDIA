import pytest

cudaq = pytest.importorskip("cudaq")

from labs_gpu.one_shot import (  # noqa: E402
    compute_adiabatic_schedule,
    get_interactions,
    kernel_beyblade,
    kernel_dna,
    kernel_jenga,
)
from labs_gpu.auxiliary_files import labs_utils  # noqa: E402


def _sample_kernel(kernel, n, list_2, list_4, steps):
    t = 1.0 / steps
    thetas = [
        labs_utils.compute_theta(t, t, 1.0, n, list_2, list_4) for _ in range(steps)
    ]
    lambdas = compute_adiabatic_schedule(steps)
    toggles = [True] * steps
    if kernel is kernel_jenga:
        result = cudaq.sample(kernel, n, list_2, list_4, thetas, shots_count=10)
    else:
        result = cudaq.sample(
            kernel,
            n,
            t,
            list_2,
            list_4,
            thetas,
            lambdas,
            toggles,
            toggles,
            shots_count=10,
        )
    return result


def test_cudaq_kernels_return_bitstrings():
    n = 4
    _, _, list_2, list_4 = get_interactions(n)
    for kernel in (kernel_jenga, kernel_dna, kernel_beyblade):
        result = _sample_kernel(kernel, n, list_2, list_4, steps=1)
        assert result
        for bitstring in result.keys():
            assert len(bitstring) == n
