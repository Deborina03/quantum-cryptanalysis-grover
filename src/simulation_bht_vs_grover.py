"""
simulation_bht_vs_grover.py
Benchmarking FRAMEWORK comparing three key-search strategies against the
reduced lightweight cipher:

  1. Classical brute-force search       -- ACTUALLY EXECUTED, randomized
                                            query order, real measured
                                            query counts (averaged over
                                            multiple trials).
  2. Grover's algorithm                 -- ACTUALLY SIMULATED on Qiskit
                                            Aer, real oracle built from
                                            known-plaintext pairs, real
                                            measured success probability
                                            and iteration count.
  3. BHT collision search               -- THEORETICAL reference curve
                                            

Dependencies: qiskit, qiskit-aer, numpy, matplotlib.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

PRESENT_SBOX: Tuple[int, ...] = (
    0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD,
    0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2,
)
BLOCK_SIZE_BITS = 8
NUM_NIBBLES = BLOCK_SIZE_BITS // 4
BLOCK_MASK = (1 << BLOCK_SIZE_BITS) - 1
PERMUTATION: Tuple[int, ...] = (0, 2, 4, 6, 1, 3, 5, 7)


def _bits_of(value: int, width: int) -> List[int]:
    return [(value >> i) & 1 for i in range(width)]


def sbox_layer(state: int, sbox: Tuple[int, ...] = PRESENT_SBOX) -> int:
    out = 0
    for n in range(NUM_NIBBLES):
        nibble = (state >> (4 * n)) & 0xF
        out |= sbox[nibble] << (4 * n)
    return out


def permutation_layer(state: int, perm: Tuple[int, ...] = PERMUTATION) -> int:
    bits = _bits_of(state, BLOCK_SIZE_BITS)
    out_bits = [0] * BLOCK_SIZE_BITS
    for src_pos, dst_pos in enumerate(perm):
        out_bits[dst_pos] = bits[src_pos]
    out = 0
    for i, b in enumerate(out_bits):
        out |= (b & 1) << i
    return out


def key_schedule(master_key: int, key_bits: int, rounds: int) -> List[int]:
    key_mask = (1 << key_bits) - 1
    state = master_key & key_mask
    subkeys: List[int] = []
    for round_idx in range(1, rounds + 1):
        expanded = 0
        shift = 0
        while shift < BLOCK_SIZE_BITS:
            expanded |= state << shift
            shift += key_bits
        expanded &= BLOCK_MASK
        subkeys.append(sbox_layer(expanded))
        if key_bits > 3:
            state = ((state << 3) | (state >> (key_bits - 3))) & key_mask
        state ^= round_idx & key_mask
    return subkeys


def encrypt(plaintext: int, key: int, key_bits: int, rounds: int) -> int:
    state = plaintext & BLOCK_MASK
    subkeys = key_schedule(key, key_bits, rounds)
    for rk in subkeys:
        state ^= rk
        state = sbox_layer(state)
        state = permutation_layer(state)
    state ^= subkeys[-1] if subkeys else 0
    return state & BLOCK_MASK


def generate_known_plaintext_pairs(
    secret_key: int, key_bits: int, rounds: int, num_pairs: int = 2
) -> List[Tuple[int, int]]:
    plaintexts = [(0x11 * i) & BLOCK_MASK for i in range(num_pairs)]
    return [(pt, encrypt(pt, secret_key, key_bits, rounds)) for pt in plaintexts]


def classical_candidate_keys(
    pairs: List[Tuple[int, int]], key_bits: int, rounds: int
) -> List[int]:
    candidates = []
    for k in range(1 << key_bits):
        if all(encrypt(pt, k, key_bits, rounds) == ct for pt, ct in pairs):
            candidates.append(k)
    return candidates



def build_marking_oracle(marked_states: List[int], num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Oracle")
    for state in marked_states:
        bits = _bits_of(state, num_qubits)
        flip_qubits = [i for i, b in enumerate(bits) if b == 0]
        if flip_qubits:
            qc.x(flip_qubits)
        if num_qubits == 1:
            qc.z(0)
        else:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        if flip_qubits:
            qc.x(flip_qubits)
    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.z(0)
    else:
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def optimal_grover_iterations(search_space_size: int, num_marked: int) -> int:
    if num_marked <= 0:
        return 0
    return max(1, round((math.pi / 4.0) * math.sqrt(search_space_size / num_marked)))


def build_grover_key_recovery_circuit(
    marked_states: List[int], num_qubits: int, iterations: int
) -> QuantumCircuit:
    qr = QuantumRegister(num_qubits, "key")
    cr = ClassicalRegister(num_qubits, "meas")
    qc = QuantumCircuit(qr, cr, name="GroverKeyRecovery")
    qc.h(qr)
    oracle_gate = build_marking_oracle(marked_states, num_qubits).to_gate(label="Oracle")
    diffuser_gate = build_diffuser(num_qubits).to_gate(label="Diffuser")
    for _ in range(iterations):
        qc.append(oracle_gate, qr)
        qc.append(diffuser_gate, qr)
    qc.measure(qr, cr)
    return qc




@dataclass
class BenchmarkPoint:
    key_bits: int
    search_space_size: int
    classical_measured_queries: float           # averaged over trials, ACTUALLY executed
    classical_trials: int
    grover_iterations: int                       # ACTUALLY used in a real circuit
    grover_measured_success_probability: float    # ACTUALLY measured on Aer
    grover_qubits: int
    grover_transpiled_depth: int
    bht_theoretical_queries: float                # THEORETICAL reference only, see module docstring
    wall_clock_seconds_classical: float
    wall_clock_seconds_grover: float


def run_classical_brute_force(
    secret_key: int, key_bits: int, rounds: int, pairs: List[Tuple[int, int]],
    trials: int, rng: random.Random,
) -> Tuple[float, float]:
    """
    ACTUALLY execute randomized brute-force search: shuffle the key order
    (so we are not biased by always trying key 0 first) and count how many
    `encrypt` evaluations are needed before the first key consistent with
    every known-plaintext pair is found. Averaged over `trials` random
    orderings to get a stable estimate of real search cost, plus wall-clock
    timing of the actual Python execution.
    """
    key_space = list(range(1 << key_bits))
    query_counts = []

    t0 = time.perf_counter()
    for _ in range(trials):
        rng.shuffle(key_space)
        queries = 0
        for k in key_space:
            queries += 1
            if all(encrypt(pt, k, key_bits, rounds) == ct for pt, ct in pairs):
                break
        query_counts.append(queries)
    elapsed = time.perf_counter() - t0

    return float(np.mean(query_counts)), elapsed


def run_grover_benchmark(
    secret_key: int, key_bits: int, rounds: int, pairs: List[Tuple[int, int]],
    shots: int, seed: int,
) -> Dict:
    """ACTUALLY build and simulate the Grover circuit for this key size."""
    candidates = classical_candidate_keys(pairs, key_bits, rounds)
    if candidates != [secret_key]:
        raise RuntimeError(f"Non-unique or incorrect candidate set: {candidates}")

    search_space_size = 1 << key_bits
    iterations = optimal_grover_iterations(search_space_size, len(candidates))
    circuit = build_grover_key_recovery_circuit(candidates, key_bits, iterations)

    backend = AerSimulator(seed_simulator=seed)

    t0 = time.perf_counter()
    transpiled = transpile(circuit, backend, optimization_level=1)
    result = backend.run(transpiled, shots=shots).result()
    elapsed = time.perf_counter() - t0

    counts = result.get_counts()
    true_key_bitstring = format(secret_key, f"0{key_bits}b")
    success_probability = counts.get(true_key_bitstring, 0) / shots

    return {
        "iterations": iterations,
        "success_probability": success_probability,
        "qubits": circuit.num_qubits,
        "depth": transpiled.depth(),
        "elapsed": elapsed,
    }


def bht_theoretical_query_complexity(search_space_size: int) -> float:
    """
    THEORETICAL reference value only : the
    well-known asymptotic quantum query complexity of the BHT algorithm
    for 2-to-1 collision finding is O(N^(1/3)). 
    """
    return search_space_size ** (1.0 / 3.0)


# ===========================================================================
# 4. Full benchmark sweep
# ===========================================================================

def run_benchmark_suite(
    key_bit_range: List[int] = None,
    rounds: int = 4,
    classical_trials: int = 200,
    grover_shots: int = 4096,
    seed: int = 123,
) -> List[BenchmarkPoint]:
    if key_bit_range is None:
        key_bit_range = [4, 5, 6, 7, 8]

    rng = random.Random(seed)
    results: List[BenchmarkPoint] = []

    for key_bits in key_bit_range:
        secret_key = rng.randrange(1 << key_bits)
        pairs = generate_known_plaintext_pairs(secret_key, key_bits, rounds, num_pairs=2)

        # Make sure the pairs actually pin down a unique key at this size;
        # if not, add pairs until they do (keeps the benchmark honest).
        num_pairs = 2
        while classical_candidate_keys(pairs, key_bits, rounds) != [secret_key]:
            num_pairs += 1
            pairs = generate_known_plaintext_pairs(secret_key, key_bits, rounds, num_pairs)

        classical_mean_queries, classical_time = run_classical_brute_force(
            secret_key, key_bits, rounds, pairs, classical_trials, rng
        )
        grover_stats = run_grover_benchmark(secret_key, key_bits, rounds, pairs, grover_shots, seed)

        results.append(BenchmarkPoint(
            key_bits=key_bits,
            search_space_size=1 << key_bits,
            classical_measured_queries=classical_mean_queries,
            classical_trials=classical_trials,
            grover_iterations=grover_stats["iterations"],
            grover_measured_success_probability=grover_stats["success_probability"],
            grover_qubits=grover_stats["qubits"],
            grover_transpiled_depth=grover_stats["depth"],
            bht_theoretical_queries=bht_theoretical_query_complexity(1 << key_bits),
            wall_clock_seconds_classical=classical_time,
            wall_clock_seconds_grover=grover_stats["elapsed"],
        ))

    return results


def print_report(results: List[BenchmarkPoint]) -> None:
    print("=" * 100)
    print("BENCHMARK: CLASSICAL BRUTE FORCE (measured) vs GROVER (simulated) vs BHT (theoretical)")
    print("=" * 100)
    header = (f"{'bits':>4} | {'N':>6} | {'classical (measured avg queries)':>32} | "
              f"{'grover iters (measured)':>24} | {'grover P(success)':>17} | "
              f"{'BHT N^(1/3) [theory]':>20}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.key_bits:>4} | {r.search_space_size:>6} | "
              f"{r.classical_measured_queries:>32.1f} | "
              f"{r.grover_iterations:>24} | "
              f"{r.grover_measured_success_probability:>17.4f} | "
              f"{r.bht_theoretical_queries:>20.2f}")
    print("=" * 100)
    print("Note: classical and Grover columns are ACTUALLY EXECUTED (randomized search / Aer "
          "simulation). BHT is a THEORETICAL reference for a different problem (collision "
          "finding), included for order-of-growth comparison only -- see module docstring.")


def plot_benchmark(results: List[BenchmarkPoint]) -> None:
    key_bits = [r.key_bits for r in results]
    classical = [r.classical_measured_queries for r in results]
    grover = [r.grover_iterations for r in results]
    bht = [r.bht_theoretical_queries for r in results]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(key_bits, classical, marker="o", color="#4c72b0",
            label="Classical brute force (measured, avg queries)")
    ax.plot(key_bits, grover, marker="s", color="#d62728",
            label="Grover search (measured, iterations used)")
    ax.plot(key_bits, bht, marker="^", color="#55a868", linestyle="--",
            label="BHT collision search (theoretical reference, $N^{1/3}$)")

    ax.set_yscale("log")
    ax.set_xlabel("Key size (bits)", fontsize=11)
    ax.set_ylabel("Query count (log scale)", fontsize=11)
    ax.set_title("Key-Search Cost: Measured Classical & Grover vs. Theoretical BHT", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    plt.show()


def main() -> None:
    results = run_benchmark_suite(key_bit_range=[4, 5, 6, 7, 8], classical_trials=200)
    print_report(results)
    plot_benchmark(results)


if __name__ == "__main__":
    main()
