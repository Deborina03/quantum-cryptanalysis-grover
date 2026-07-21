"""

Dependencies: qiskit, qiskit-aer, numpy, matplotlib.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from qiskit import transpile
from qiskit_aer import AerSimulator

# Reuse every cipher / oracle / benchmark routine from the sibling module
# instead of re-deriving them -- this keeps the scaling analysis backed by
# the exact same real cipher and the exact same real Grover circuit
# construction used elsewhere in the repository.
from simulation_bht_vs_grover import (
    BLOCK_SIZE_BITS,
    build_grover_key_recovery_circuit,
    bht_theoretical_query_complexity,
    classical_candidate_keys,
    generate_known_plaintext_pairs,
    optimal_grover_iterations,
    run_classical_brute_force,
)
import random


# ===========================================================================
# 1. Real circuit-resource scaling (qubits / depth / two-qubit gate count)
# ===========================================================================

def measure_circuit_resources(key_bits: int, num_marked: int = 1) -> Dict:
    """
    Build an actual Grover circuit for a key space of this size and read
    real resource counts off Qiskit's transpiler output -- not a formula.
    `num_marked` marked states are placed at arbitrary positions (0, 1, ...)
    since resource cost (qubits/depth/gate count) depends on the NUMBER of
    marked states and the iteration count, not on which specific keys are
    marked.
    """
    marked_states = list(range(num_marked))
    iterations = optimal_grover_iterations(1 << key_bits, num_marked)
    circuit = build_grover_key_recovery_circuit(marked_states, key_bits, iterations)

    backend = AerSimulator()
    transpiled = transpile(circuit, backend, basis_gates=["id", "rz", "sx", "x", "cx"],
                            optimization_level=1)
    ops = transpiled.count_ops()

    return {
        "key_bits": key_bits,
        "iterations": iterations,
        "qubits": circuit.num_qubits,
        "logical_depth": circuit.depth(),
        "transpiled_depth": transpiled.depth(),
        "two_qubit_gate_count": ops.get("cx", 0),
        "total_gate_count": sum(ops.values()),
    }


# ===========================================================================
# 2. Query-cost scaling: actually measured classical + Grover, fitted
# ===========================================================================

def run_scaling_sweep(
    key_bit_range: List[int] = None,
    rounds: int = 4,
    classical_trials: int = 100,
    grover_shots: int = 2048,
    seed: int = 2026,
) -> Dict:
    if key_bit_range is None:
        key_bit_range = list(range(4, 12))  # 4..11 bits

    rng = random.Random(seed)
    classical_queries: List[float] = []
    grover_iterations: List[int] = []
    grover_success: List[float] = []
    bht_theoretical: List[float] = []
    resources: List[Dict] = []

    for key_bits in key_bit_range:
        secret_key = rng.randrange(1 << key_bits)
        pairs = generate_known_plaintext_pairs(secret_key, key_bits, rounds, num_pairs=2)
        while classical_candidate_keys(pairs, key_bits, rounds) != [secret_key]:
            pairs = generate_known_plaintext_pairs(secret_key, key_bits, rounds, len(pairs) + 1)

        mean_queries, _ = run_classical_brute_force(
            secret_key, key_bits, rounds, pairs, classical_trials, rng
        )
        classical_queries.append(mean_queries)

        iterations = optimal_grover_iterations(1 << key_bits, 1)
        marked = classical_candidate_keys(pairs, key_bits, rounds)
        circuit = build_grover_key_recovery_circuit(marked, key_bits, iterations)
        backend = AerSimulator(seed_simulator=seed)
        transpiled = transpile(circuit, backend, optimization_level=1)
        result = backend.run(transpiled, shots=grover_shots).result()
        counts = result.get_counts()
        true_bitstring = format(secret_key, f"0{key_bits}b")
        grover_success.append(counts.get(true_bitstring, 0) / grover_shots)
        grover_iterations.append(iterations)

        bht_theoretical.append(bht_theoretical_query_complexity(1 << key_bits))
        resources.append(measure_circuit_resources(key_bits))

    return {
        "key_bit_range": key_bit_range,
        "classical_queries": classical_queries,
        "grover_iterations": grover_iterations,
        "grover_success": grover_success,
        "bht_theoretical": bht_theoretical,
        "resources": resources,
    }


def fit_power_law_exponent(sizes: List[int], values: List[float]) -> Tuple[float, float]:
    """
    Fit y = a * N^b via log-log linear regression (polyfit degree 1 on
    log(N) vs log(y)). Returns (exponent b, prefactor a). Used to turn
    "the classical search cost grows roughly linearly with N" from an
    eyeballed impression into an actual fitted number.
    """
    log_n = np.log(sizes)
    log_y = np.log(values)
    b, log_a = np.polyfit(log_n, log_y, 1)
    return float(b), float(math.exp(log_a))


def print_report(sweep: Dict) -> None:
    key_bits = sweep["key_bit_range"]
    ns = [1 << k for k in key_bits]

    classical_exp, classical_prefactor = fit_power_law_exponent(ns, sweep["classical_queries"])
    grover_exp, grover_prefactor = fit_power_law_exponent(ns, sweep["grover_iterations"])

    print("=" * 100)
    print("SCALING ANALYSIS -- MEASURED QUERY COST vs KEY SIZE")
    print("=" * 100)
    header = (f"{'bits':>4} | {'N':>6} | {'classical (measured)':>21} | "
              f"{'grover iters (measured)':>24} | {'grover P(success)':>17} | "
              f"{'BHT theory N^(1/3)':>19}")
    print(header)
    print("-" * len(header))
    for i, kb in enumerate(key_bits):
        print(f"{kb:>4} | {ns[i]:>6} | {sweep['classical_queries'][i]:>21.1f} | "
              f"{sweep['grover_iterations'][i]:>24} | {sweep['grover_success'][i]:>17.4f} | "
              f"{sweep['bht_theoretical'][i]:>19.2f}")
    print("-" * len(header))
    print(f"Fitted classical scaling exponent : {classical_exp:.3f}  "
          f"(theoretical exponent for exhaustive search: 1.0)")
    print(f"Fitted Grover scaling exponent     : {grover_exp:.3f}  "
          f"(theoretical exponent for Grover search: 0.5)")
    print("=" * 100)

    print("REAL CIRCUIT-RESOURCE SCALING (from actual Qiskit transpilation)")
    print("=" * 100)
    rheader = (f"{'bits':>4} | {'qubits':>6} | {'logical depth':>13} | "
               f"{'transpiled depth':>16} | {'2-qubit gates':>13} | {'total gates':>11}")
    print(rheader)
    print("-" * len(rheader))
    for r in sweep["resources"]:
        print(f"{r['key_bits']:>4} | {r['qubits']:>6} | {r['logical_depth']:>13} | "
              f"{r['transpiled_depth']:>16} | {r['two_qubit_gate_count']:>13} | "
              f"{r['total_gate_count']:>11}")
    depth_exp, _ = fit_power_law_exponent(ns, [r["transpiled_depth"] for r in sweep["resources"]])
    print(f"Fitted transpiled-depth scaling exponent vs N : {depth_exp:.3f}")
    print("=" * 100)


def plot_scaling_analysis(sweep: Dict) -> None:
    key_bits = sweep["key_bit_range"]
    ns = np.array([1 << k for k in key_bits], dtype=float)

    classical = np.array(sweep["classical_queries"])
    grover = np.array(sweep["grover_iterations"], dtype=float)
    bht = np.array(sweep["bht_theoretical"])

    classical_exp, classical_a = fit_power_law_exponent(ns.tolist(), classical.tolist())
    grover_exp, grover_a = fit_power_law_exponent(ns.tolist(), grover.tolist())

    fine_n = np.linspace(ns.min(), ns.max(), 200)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Panel 1: query-cost scaling ---
    ax1.scatter(ns, classical, color="#4c72b0", zorder=5, label="Classical (measured)")
    ax1.plot(fine_n, classical_a * fine_n ** classical_exp, color="#4c72b0", linestyle="--",
              label=f"Classical fit: $N^{{{classical_exp:.2f}}}$")

    ax1.scatter(ns, grover, color="#d62728", marker="s", zorder=5, label="Grover (measured)")
    ax1.plot(fine_n, grover_a * fine_n ** grover_exp, color="#d62728", linestyle="--",
              label=f"Grover fit: $N^{{{grover_exp:.2f}}}$")

    ax1.plot(ns, bht, color="#55a868", marker="^", linestyle=":",
              label="BHT (theoretical, $N^{1/3}$)")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Key space size $N = 2^{\\mathrm{bits}}$", fontsize=11)
    ax1.set_ylabel("Query count (log scale)", fontsize=11)
    ax1.set_title("Measured Query-Cost Scaling with Fitted Exponents", fontsize=11)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3, which="both")

    # --- Panel 2: real circuit-resource scaling ---
    depths = [r["transpiled_depth"] for r in sweep["resources"]]
    two_q_gates = [r["two_qubit_gate_count"] for r in sweep["resources"]]
    qubits = [r["qubits"] for r in sweep["resources"]]

    ax2b = ax2.twinx()
    l1 = ax2.plot(key_bits, depths, color="#8172b2", marker="o", label="Transpiled circuit depth")
    l2 = ax2.plot(key_bits, two_q_gates, color="#c44e52", marker="s", label="Two-qubit (CX) gate count")
    l3 = ax2b.plot(key_bits, qubits, color="#937860", marker="^", linestyle="--", label="Qubit count")

    ax2.set_xlabel("Key size (bits)", fontsize=11)
    ax2.set_ylabel("Depth / gate count", fontsize=11)
    ax2b.set_ylabel("Qubit count", fontsize=11)
    ax2.set_title("Real Circuit-Resource Scaling (Transpiled)", fontsize=11)
    lines = l1 + l2 + l3
    ax2.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper left")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    plt.show()


def main() -> None:
    sweep = run_scaling_sweep(key_bit_range=list(range(4, 12)), classical_trials=100, grover_shots=2048)
    print_report(sweep)
    plot_scaling_analysis(sweep)


if __name__ == "__main__":
    main()
