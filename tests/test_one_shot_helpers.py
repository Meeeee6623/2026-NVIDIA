import sys
import types

import numpy as np

try:
    import cudaq  # noqa: F401
except ImportError:
    dummy = types.ModuleType("cudaq")
    dummy.kernel = lambda fn=None, **_: fn  # type: ignore[assignment]
    sys.modules["cudaq"] = dummy

from labs_gpu.one_shot import (
    compute_adiabatic_schedule,
    export_ready_for_cuda,
    get_interactions,
)


def test_get_interactions_shapes():
    g2, g4, list_2, list_4 = get_interactions(6)
    assert len(list_2) % 2 == 0
    assert len(list_4) % 4 == 0
    assert len(g2) * 2 == len(list_2)
    assert len(g4) * 4 == len(list_4)


def test_compute_adiabatic_schedule_bounds():
    schedule = compute_adiabatic_schedule(4)
    assert len(schedule) == 4
    assert all(0.0 <= val <= 1.0 for val in schedule)
    assert schedule[-1] > 0.9


def test_export_ready_for_cuda_writes_file(tmp_path):
    population = np.array([[1, -1, 1, -1], [-1, -1, 1, 1]], dtype=np.int8)
    output = tmp_path / "warm_start.bin"
    export_ready_for_cuda(
        population, n=4, filename=str(output), gpu_pop_size=4, gpu_max_n=8
    )
    data = np.fromfile(output, dtype=np.int8)
    assert data.size == 4 * 8
    loaded = data.reshape(4, 8)
    assert np.array_equal(loaded[:2, :4], population)
