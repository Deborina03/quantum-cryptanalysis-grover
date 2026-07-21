"""

Noise-aware Grover key-recovery attack: compares IDEAL vs NOISY execution
of the same live Grover circuit from `simulation_live_grover_key_recovery.py`
under a Qiskit Aer depolarizing noise model, across a sweep of physical
error rates.

Dependencies: qiskit, qiskit-aer, numpy, matplotlib.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# ===========================================================================
# 1. Reduced lightweight cipher (identical construction to
#    simulation_live_grover_key_recovery.py, duplicated here so this file
#    remains independently runnable without cross-file imports).
# ===========================================================================

PRESENT_SBOX: Tuple[int, ...] = (
    0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD,
    0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2,
)
PRESENT_SBOX_INV: Tuple[int, ...] = tuple(PRESENT_SBOX.index(i) for i in range(16))

BLOCK_SIZE_BITS = 8
NUM_NIBBLES = BLOCK_SIZE_BITS // 4
BLOCK_MASK = (1 << BLOCK_SIZE_BITS) - 1
PERMUTATION: Tuple[int, ...] = (0, 2, 4, 6, 1, 3, 5, 7)
PERMUTATION_INV: Tuple[int, ...] = tuple(PERMUTATION.index(i) for i in range(BLOCK_SIZE_BITS))


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


# ===========================================================================
# 2. Grover oracle + diffuser (identical construction to the live-attack file)
# ===========================================================================

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


# ===========================================================================
# 3. Depolarizing noise model construction
# ===========================================================================

def build_depolarizing_noise_model(p1: float, p2: float) -> NoiseModel:
    """
    Build a Qiskit Aer NoiseModel with depolarizing error applied to
    single-qubit gates (rate p1) and two-qubit gates (rate p2). Two-qubit gate error rates are conventionally set higher
    than single-qubit rates, matching typical superconducting-qubit
    hardware characterization reports.

    IMPORTANT: the error is attached to gate names in `BASIS_GATES` below.
    The circuit MUST be transpiled to this exact basis before simulation,
    otherwise Aer's default transpilation may keep multi-controlled gates
    (e.g. `mcx`) as single native instructions that never match a "cx"
    noise entry -- silently making the "noisy" run identical to the ideal
    one. `run_noise_sweep` enforces this by transpiling with
    `basis_gates=BASIS_GATES` explicitly.
    """
    noise_model = NoiseModel()

    error_1q = depolarizing_error(p1, 1)
    noise_model.add_all_qubit_quantum_error(error_1q, ["id", "rz", "sx", "x"])

    error_2q = depolarizing_error(p2, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])

    return noise_model


# Fixed basis so the circuit ALWAYS gets decomposed down to single- and
# two-qubit gates the noise model can actually attach errors to -- Aer's
# default transpilation would otherwise keep `mcx` as one native
# instruction, bypassing per-gate depolarizing noise entirely.
BASIS_GATES = ["id", "rz", "sx", "x", "cx"]


# ===========================================================================
# 4. Ideal vs noisy execution sweep
# ===========================================================================

def run_noise_sweep(
    secret_key: int = 0b10110,
    key_bits: int = 5,
    rounds: int = 4,
    num_known_pairs: int = 2,
    shots: int = 4096,
    p1_sweep: List[float] = None,
    two_qubit_to_single_qubit_ratio: float = 10.0,
    seed: int = 7,
) -> Dict:
    """
    Run the SAME Grover key-recovery circuit under a sweep of physical
    single-qubit depolarizing error rates (with the two-qubit rate set at
    `two_qubit_to_single_qubit_ratio` times higher, matching the usual
    real-hardware pattern of noisier two-qubit gates), and report the
    measured success probability of recovering the true key at each point.
    """
    if p1_sweep is None:
        p1_sweep = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]

    pairs = generate_known_plaintext_pairs(secret_key, key_bits, rounds, num_known_pairs)
    candidates = classical_candidate_keys(pairs, key_bits, rounds)
    if candidates != [secret_key]:
        raise RuntimeError(f"Known-plaintext pairs do not uniquely determine the key: {candidates}")

    search_space_size = 1 << key_bits
    iterations = optimal_grover_iterations(search_space_size, len(candidates))
    circuit = build_grover_key_recovery_circuit(candidates, key_bits, iterations)

    # Transpile ONCE to the fixed basis so both the ideal and every noisy
    # run share identical gate structure -- the noise model's error rates
    # are then the only thing that differs between runs.
    ideal_backend = AerSimulator(seed_simulator=seed)
    common_transpiled = transpile(circuit, ideal_backend, basis_gates=BASIS_GATES,
                                   optimization_level=1)
    ideal_result = ideal_backend.run(common_transpiled, shots=shots).result()
    ideal_counts = ideal_result.get_counts()
    true_key_bitstring = format(secret_key, f"0{key_bits}b")
    ideal_success = ideal_counts.get(true_key_bitstring, 0) / shots

    two_qubit_gate_count = common_transpiled.count_ops().get("cx", 0)
    circuit_depth = common_transpiled.depth()

    noisy_success_probs: List[float] = []
    for p1 in p1_sweep:
        p2 = min(0.75, p1 * two_qubit_to_single_qubit_ratio)  # depolarizing error must stay < 1
        noise_model = build_depolarizing_noise_model(p1, p2)
        noisy_backend = AerSimulator(noise_model=noise_model, seed_simulator=seed)
        noisy_result = noisy_backend.run(common_transpiled, shots=shots).result()
        noisy_counts = noisy_result.get_counts()
        noisy_success_probs.append(noisy_counts.get(true_key_bitstring, 0) / shots)

    return {
        "secret_key": secret_key,
        "key_bits": key_bits,
        "search_space_size": search_space_size,
        "iterations": iterations,
        "shots": shots,
        "p1_sweep": p1_sweep,
        "ideal_success_probability": ideal_success,
        "noisy_success_probabilities": noisy_success_probs,
        "two_qubit_gate_count": two_qubit_gate_count,
        "circuit_depth": circuit_depth,
    }


def print_report(sweep: Dict) -> None:
    print("=" * 70)
    print("NOISE-AWARE GROVER KEY-RECOVERY -- IDEAL VS NOISY EXECUTION")
    print("=" * 70)
    print(f"Reduced cipher key size      : {sweep['key_bits']} bits "
          f"(search space N = {sweep['search_space_size']})")
    print(f"Grover iterations            : {sweep['iterations']}")
    print(f"Two-qubit (CX) gate count    : {sweep['two_qubit_gate_count']}  "
          f"<- noise accumulates through this many noisy 2-qubit gates")
    print(f"Transpiled circuit depth     : {sweep['circuit_depth']}")
    print(f"Shots per data point         : {sweep['shots']}")
    print("-" * 70)
    print(f"IDEAL (noiseless) success probability : {sweep['ideal_success_probability']:.4f}")
    print("-" * 70)
    print(f"{'1-qubit error rate p1':>24} | {'noisy success probability':>26}")
    for p1, prob in zip(sweep["p1_sweep"], sweep["noisy_success_probabilities"]):
        print(f"{p1:>24.4f} | {prob:>26.4f}")
    print("=" * 70)


def plot_noise_curve(sweep: Dict) -> None:
    """Publication-quality ideal-vs-noisy success probability curve."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.axhline(sweep["ideal_success_probability"], color="#2ca02c", linestyle="--",
               linewidth=1.5, label="Ideal (noiseless) success probability")
    ax.plot(sweep["p1_sweep"], sweep["noisy_success_probabilities"],
            marker="o", color="#d62728", linewidth=2, markersize=6,
            label="Noisy execution (depolarizing model)")

    uniform_baseline = 1.0 / sweep["search_space_size"]
    ax.axhline(uniform_baseline, color="gray", linestyle=":", linewidth=1,
               label="Uniform-guessing baseline (1/N)")

    ax.set_xlabel("Single-qubit depolarizing error rate $p_1$", fontsize=11)
    ax.set_ylabel("Measured success probability", fontsize=11)
    ax.set_title(
        f"Grover Key-Recovery Success Under Depolarizing Noise\n"
        f"{sweep['key_bits']}-bit reduced cipher, {sweep['iterations']} Grover iterations, "
        f"{sweep['shots']} shots/point",
        fontsize=11,
    )
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="center left", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    plt.show()


def main() -> None:
    sweep = run_noise_sweep()
    print_report(sweep)
    plot_noise_curve(sweep)


if __name__ == "__main__":
    main()
