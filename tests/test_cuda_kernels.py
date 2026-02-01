import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _skip_if_no_cuda():
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc not available")
    try:
        subprocess.run(["nvidia-smi"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pytest.skip("CUDA GPU not available")


def test_cuda_kernels_smoke(tmp_path):
    _skip_if_no_cuda()

    binary = tmp_path / "labs_solver"
    source = REPO_ROOT / "labs_gpu" / "labs_gpu.cu"
    compile_cmd = [
        "nvcc",
        "-O2",
        "-std=c++17",
        str(source),
        "-o",
        str(binary),
    ]
    subprocess.run(compile_cmd, check=True)

    pop_size = 8
    max_n = 512
    n = 8
    warm_start = tmp_path / "warm_start.bin"
    rng = np.random.default_rng(123)
    data = rng.choice([-1, 1], size=(pop_size, max_n)).astype(np.int8)
    data.tofile(warm_start)

    run_cmd = [str(binary), str(n), str(pop_size), str(warm_start), "1000"]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    result = subprocess.run(run_cmd, check=True, capture_output=True, text=True, env=env)
    assert "PASS" in result.stdout or "FAIL" in result.stdout
