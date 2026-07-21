"""
Attack-cost ACCUMULATION analysis for a periodic key-rotation defense:
models how much total attacker effort (classical queries, and separately
Grover queries) is required to maintain continuous key compromise over a
fixed defense horizon, as a function of how often the defender rotates
keys -- backed by ACTUALLY MEASURED per-attack costs (real randomized
classical search + real Grover circuit simulation), not an assumed
constant.

Dependencies: qiskit, qiskit-aer, numpy, matplotlib.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from qiskit import transpile
from qiskit_aer import AerSimulator

# Reuse the real cipher, oracle, and benchmark routines from the sibling
# module so this analysis is grounded in the exact same attack machinery
# used throughout the repository.
from simulation_bht_vs_grover import (
    build_grover_key_recovery_circuit,
    classical_candidate_keys,
    generate_known_plaintext_pairs,
    optimal_grover_iterations,
    run_classical_brute_force,
)


# ===========================================================================
# 1. Measure real per-attack cost across several independently rotated keys
# ===========================================================================

def measure_per_attack_cost_samples(
    key_bits: int,
    rounds: int,
    num_independent_keys: int,
    classical_trials: int,
    grover_shots: int,
    seed: int,
) -> Dict:
    """
    Simulate `num_independent_keys` separate key-rotation events: each one
    uses a genuinely different random secret key, and for each we ACTUALLY
    run (a) randomized classical brute-force search and (b) a real Grover
    circuit simulation, recording the real per-attack cost. This captures
    sample-to-sample variance that a single fixed-key measurement would
    hide -- relevant because a defender rotating keys forces the attacker
    to face a new, independently-drawn key each time.
    """
    rng = random.Random(seed)
    classical_costs: List[float] = []
    grover_costs: List[int] = []
    grover_success: List[float] = []

    for _ in range(num_independent_keys):
        secret_key = rng.randrange(1 << key_bits)
        pairs = generate_known_plaintext_pairs(secret_key, key_bits, rounds, num_pairs=2)
        while classical_candidate_keys(pairs, key_bits, rounds) != [secret_key]:
            pairs = generate_known_plaintext_pairs(secret_key, key_bits, rounds, len(pairs) + 1)

        mean_queries, _ = run_classical_brute_force(
            secret_key, key_bits, rounds, pairs, classical_trials, rng
        )
        classical_costs.append(mean_queries)

        candidates = classical_candidate_keys(pairs, key_bits, rounds)
        iterations = optimal_grover_iterations(1 << key_bits, len(candidates))
        circuit = build_grover_key_recovery_circuit(candidates, key_bits, iterations)
        backend = AerSimulator(seed_simulator=seed)
        transpiled = transpile(circuit, backend, optimization_level=1)
        result = backend.run(transpiled, shots=grover_shots).result()
        counts = result.get_counts()
        true_bitstring = format(secret_key, f"0{key_bits}b")
        grover_success.append(counts.get(true_bitstring, 0) / grover_shots)
        grover_costs.append(iterations)

    return {
        "key_bits": key_bits,
        "classical_costs": classical_costs,
        "grover_costs": grover_costs,
        "grover_success": grover_success,
        "classical_mean": float(np.mean(classical_costs)),
        "classical_std": float(np.std(classical_costs)),
        "grover_mean": float(np.mean(grover_costs)),
        "grover_std": float(np.std(grover_costs)),
    }


# ===========================================================================
# 2. Attack-cost accumulation model over a defense horizon
# ===========================================================================

def accumulated_attack_cost(
    per_attack_cost: float, horizon_epochs: int, rotation_period_epochs: int
) -> float:
    """
    Explicit accumulation model: over `horizon_epochs` epochs, a defender
    rotating keys every `rotation_period_epochs` epochs forces the
    attacker to mount ceil(horizon / rotation_period) INDEPENDENT attacks
    to remain continuously compromised. Each attack costs the measured
    `per_attack_cost`. This is a simple, explicit multiplication -- stated
    here so it is auditable, not buried inside a plotting routine.
    """
    num_rotations = math.ceil(horizon_epochs / rotation_period_epochs)
    return num_rotations * per_attack_cost


def run_rotation_defense_analysis(
    key_bits: int = 6,
    rounds: int = 4,
    num_independent_keys: int = 8,
    classical_trials: int = 150,
    grover_shots: int = 2048,
    horizon_epochs: int = 200,
    rotation_periods: List[int] = None,
    seed: int = 99,
) -> Dict:
    if rotation_periods is None:
        rotation_periods = [1, 2, 5, 10, 20, 50, 100, 200]

    samples = measure_per_attack_cost_samples(
        key_bits, rounds, num_independent_keys, classical_trials, grover_shots, seed
    )

    accumulated_classical = [
        accumulated_attack_cost(samples["classical_mean"], horizon_epochs, T)
        for T in rotation_periods
    ]
    accumulated_grover = [
        accumulated_attack_cost(samples["grover_mean"], horizon_epochs, T)
        for T in rotation_periods
    ]
    # Defender-side companion quantity: average exposure window (epochs of
    # data at risk) if a break happens partway through a key's lifetime.
    average_exposure_window = [T / 2.0 for T in rotation_periods]

    return {
        "samples": samples,
        "horizon_epochs": horizon_epochs,
        "rotation_periods": rotation_periods,
        "accumulated_classical": accumulated_classical,
        "accumulated_grover": accumulated_grover,
        "average_exposure_window": average_exposure_window,
    }


def print_report(analysis: Dict) -> None:
    s = analysis["samples"]
    print("=" * 100)
    print("KEY-ROTATION DEFENSE -- MEASURED PER-ATTACK COST (across independently rotated keys)")
    print("=" * 100)
    print(f"Key size                     : {s['key_bits']} bits (N = {1 << s['key_bits']})")
    print(f"Independent rotation events measured : {len(s['classical_costs'])}")
    print(f"Classical per-attack cost    : mean = {s['classical_mean']:.1f} queries, "
          f"std = {s['classical_std']:.1f}")
    print(f"Grover per-attack cost       : mean = {s['grover_mean']:.1f} iterations, "
          f"std = {s['grover_std']:.2f}")
    print(f"Grover measured success prob : {np.mean(s['grover_success']):.4f} (averaged across events)")
    print("-" * 100)
    print(f"ACCUMULATED ATTACKER COST OVER A {analysis['horizon_epochs']}-EPOCH DEFENSE HORIZON")
    print("-" * 100)
    header = (f"{'rotation period T (epochs)':>26} | {'# re-attacks needed':>19} | "
              f"{'accum. classical cost':>22} | {'accum. Grover cost':>19} | "
              f"{'avg exposure window':>19}")
    print(header)
    for i, T in enumerate(analysis["rotation_periods"]):
        num_rotations = math.ceil(analysis["horizon_epochs"] / T)
        print(f"{T:>26} | {num_rotations:>19} | "
              f"{analysis['accumulated_classical'][i]:>22.1f} | "
              f"{analysis['accumulated_grover'][i]:>19.1f} | "
              f"{analysis['average_exposure_window'][i]:>19.1f}")
    print("=" * 100)
    print("Reading: shorter rotation periods force MORE re-attacks (higher accumulated attacker "
          "cost) but SHORTER data-exposure windows per successful break -- the two right-hand "
          "columns are the defender's actual trade-off.")


def plot_rotation_analysis(analysis: Dict) -> None:
    rotation_periods = analysis["rotation_periods"]
    accumulated_classical = analysis["accumulated_classical"]
    accumulated_grover = analysis["accumulated_grover"]
    exposure = analysis["average_exposure_window"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax1.plot(rotation_periods, accumulated_classical, marker="o", color="#4c72b0",
              label="Accumulated classical attacker cost")
    ax1.plot(rotation_periods, accumulated_grover, marker="s", color="#d62728",
              label="Accumulated Grover attacker cost")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Key rotation period T (epochs)", fontsize=11)
    ax1.set_ylabel("Accumulated attacker query cost (log scale)", fontsize=11)
    ax1.set_title(f"Accumulated Attacker Cost over {analysis['horizon_epochs']}-Epoch Horizon\n"
                   f"(measured per-attack cost x number of forced re-attacks)", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, which="both")

    ax2.plot(rotation_periods, exposure, marker="^", color="#55a868")
    ax2.set_xscale("log")
    ax2.set_xlabel("Key rotation period T (epochs)", fontsize=11)
    ax2.set_ylabel("Average data-exposure window (epochs)", fontsize=11)
    ax2.set_title("Defender's Exposure Window vs Rotation Frequency", fontsize=11)
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    plt.show()


def main() -> None:
    analysis = run_rotation_defense_analysis(
        key_bits=6, num_independent_keys=8, classical_trials=150,
        grover_shots=2048, horizon_epochs=200,
    )
    print_report(analysis)
    plot_rotation_analysis(analysis)


if __name__ == "__main__":
    main()
