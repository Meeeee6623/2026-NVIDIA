#!/usr/bin/env python3
import cudaq
from cudaq import spin
import numpy as np
import random
import time
import pandas as pd
import os
from scipy.optimize import minimize

# --- CONFIGURATION ---
cudaq.set_target("nvidia", option="mqpu,fp32")  # or "mqpu"

target = cudaq.get_target()
print("Visible GPUs:", cudaq.num_available_gpus())
print("MQPU QPUs:", target.num_qpus())

NUM_QUBITS = 5
N_LAYERS = 15
NUM_PARAMS = N_LAYERS * 15  # 225
CSV_FILE = "optimization_log.csv"


# --- KERNEL DEFINITION ---
@cudaq.kernel
def labs_ansatz(params: list[float], n: int, n_layers: int):
    qubits = cudaq.qvector(n)
    for i in range(n):
        h(qubits[i])

    for l in range(n_layers):
        offset = 15 * l
        for i in range(n):
            ry(params[offset + i], qubits[i])
            rz(params[offset + 5 + i], qubits[i])
            h(qubits[i])

        for i in range(n):
            nxt = (i + 1) % n
            cx(qubits[i], qubits[nxt])
            rz(params[offset + 10 + i], qubits[nxt])
            cx(qubits[i], qubits[nxt])


# --- HELPERS ---
def generate_pauli_list(n, n_qubits):
    pauli_list = []
    while len(pauli_list) < n:
        res = "".join(random.choice(["X", "Y", "Z"]) for _ in range(n_qubits))
        if res not in pauli_list:
            pauli_list.append(res)

    cudaq_ops = []
    for p_str in pauli_list:
        op = spin.identity(n_qubits)
        for i, char in enumerate(p_str):
            if char == "X":
                op *= spin.x(i)
            elif char == "Y":
                op *= spin.y(i)
            elif char == "Z":
                op *= spin.z(i)
        cudaq_ops.append(op)
    return cudaq_ops


def export_to_bin(top_bitstrings, n_qubits, n_paulis):
    """Saves Top 128 strings into 8192x512 int8 format (4MB binary)."""
    filename = f"warm_start_pauli_{n_paulis}.bin"
    host_buffer = np.random.choice([-1, 1], size=(8192, 512)).astype(np.int8)
    for i, s in enumerate(top_bitstrings[:128]):
        arr = np.array([int(b) for b in s])
        host_buffer[i, :n_qubits] = (2 * arr - 1).astype(np.int8)
    host_buffer.tofile(filename)


class JengaOptimizer:
    def __init__(self, paulis, n_qubits, n_layers):
        self.paulis = paulis
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.alpha, self.beta = 6.0, 15.0
        self.step_count = 0

    def compute_loss(self, params):
        # Distributed across 8 GPUs
        futures = [
            cudaq.observe_async(labs_ansatz, p_op, params, self.n_qubits, self.n_layers)
            for p_op in self.paulis
        ]
        E_list = np.array([f.get().expectation() for f in futures])
        x_tilde = np.tanh(self.alpha * E_list)

        N = len(x_tilde)
        loss = 0.0
        for ell in range(1, N):
            for i in range(N - ell):
                loss += (x_tilde[i] * x_tilde[i + ell]) ** 2
        loss -= self.beta * np.sum(x_tilde**2)
        return loss

    def callback(self, x):
        self.step_count += 1
        print(f"    [Step {self.step_count}] Logic update...", flush=True)


# --- MAIN REVERSE LOOP ---

if not os.path.exists(CSV_FILE):
    headers = ["n_paulis", "duration", "best_loss"] + [
        f"p{i}" for i in range(NUM_PARAMS)
    ]
    pd.DataFrame(columns=headers).to_csv(CSV_FILE, index=False)

# Start at 60, stop at 3 (inclusive)
for n_p in range(59, 2, -1):
    print(f"\n>>> PROCESSING N_PAULIS = {n_p}", flush=True)
    pauli_ops = generate_pauli_list(n_p, NUM_QUBITS)
    opt = JengaOptimizer(pauli_ops, NUM_QUBITS, N_LAYERS)

    start_time = time.time()

    # AGGRESSIVE STOPPING FOR SPEED
    # For 225 params, 1 step = 226 evaluations.
    # maxiter=8 means roughly 1800-2000 evaluations total.
    res = minimize(
        opt.compute_loss,
        np.random.uniform(0, 2 * np.pi, NUM_PARAMS),
        method="L-BFGS-B",
        callback=opt.callback,
        options={
            "maxiter": 8,  # Hard limit: Only 8 major gradient steps
            "maxfun": 2500,  # Safety ceiling on total evaluations
            "maxls": 5,  # Limit search depth (prevents hanging)
            "ftol": 1e-3,  # Coarser tolerance for faster exit
        },
    )

    elapsed = time.time() - start_time

    # Final Extraction
    print(f"  Complete. Loss: {res.fun:.4f}. Sampling...", flush=True)
    sample_result = cudaq.sample(
        labs_ansatz, res.x, NUM_QUBITS, N_LAYERS, shots_count=5000
    )
    sorted_strings = [
        bs
        for bs, count in sorted(sample_result.items(), key=lambda x: x[1], reverse=True)
    ]

    export_to_bin(sorted_strings, NUM_QUBITS, n_p)

    # Log to CSV
    log_data = [n_p, elapsed, res.fun] + list(res.x)
    pd.DataFrame([log_data]).to_csv(CSV_FILE, mode="a", header=False, index=False)

print("\nSweep Complete.", flush=True)
