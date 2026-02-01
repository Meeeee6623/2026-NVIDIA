import numpy as np

from labs_gpu.auxiliary_files import labs_utils


def test_compute_topology_overlaps_self_counts():
    g2 = [[0, 1], [1, 2]]
    g4 = [[0, 1, 2, 3]]
    overlaps = labs_utils.compute_topology_overlaps(g2, g4)
    assert overlaps == {"22": 2, "44": 1, "24": 0}


def test_compute_theta_zero_total_time_returns_zero():
    theta = labs_utils.compute_theta(t=0.5, dt=0.1, total_time=0.0, N=4, G2=[], G4=[])
    assert theta == 0.0


def test_compute_theta_finite_for_small_system():
    g2 = [[0, 1]]
    g4 = []
    theta = labs_utils.compute_theta(t=0.25, dt=0.1, total_time=1.0, N=4, G2=g2, G4=g4)
    assert np.isfinite(theta)
