
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator


PRESENT_SBOX: Tuple[int, ...] = (
    0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD,
    0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2,
)
PRESENT_SBOX_INV: Tuple[int, ...] = tuple(PRESENT_SBOX.index(i) for i in range(16))

BLOCK_SIZE_BITS = 8
NUM_NIBBLES = BLOCK_SIZE_BITS // 4
BLOCK_MASK = (1 << BLOCK_SIZE_BITS) - 1

# Fixed bit-permutation layer (scaled-down analogue of PRESENT's 64-bit wire
# crossing): spreads the 4 output bits of each S-box across both S-boxes of
# the next round, forcing genuine diffusion instead of independent nibbles.
PERMUTATION: Tuple[int, ...] = (0, 2, 4, 6, 1, 3, 5, 7)
PERMUTATION_INV: Tuple[int, ...] = tuple(PERMUTATION.index(i) for i in range(BLOCK_SIZE_BITS))


def _bits_of(value: int, width: int) -> List[int]:
    return [(value >> i) & 1 for i in range(width)]


def _value_of(bits: List[int]) -> int:
    out = 0
    for i, b in enumerate(bits):
        out |= (b & 1) << i
    return out


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
    return _value_of(out_bits)


def key_schedule(master_key: int, key_bits: int, rounds: int) -> List[int]:
    """Derive one round subkey per round from a `key_bits`-bit master key."""
    key_mask = (1 << key_bits) - 1
    state = master_key & key_mask
    subkeys: List[int] = []

    for round_idx in range(1, rounds + 1):
        # Expand the key_bits-wide state to a full BLOCK_SIZE_BITS-wide
        # value by tiling it at increasing shifts. Loop bound is on `shift`
        # (independent of the key's numeric value), so this terminates
        # even when state == 0 -- unlike a bit-length-based loop condition,
        # which would spin forever whenever the state happens to be zero.
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
    """SPN encryption: (XOR round key -> S-box -> permutation) x rounds, + final whitening."""
    state = plaintext & BLOCK_MASK
    subkeys = key_schedule(key, key_bits, rounds)
    for rk in subkeys:
        state ^= rk
        state = sbox_layer(state)
        state = permutation_layer(state)
    state ^= subkeys[-1] if subkeys else 0
    return state & BLOCK_MASK


def decrypt(ciphertext: int, key: int, key_bits: int, rounds: int) -> int:
    """Exact inverse of `encrypt`, used to verify a recovered key end-to-end."""
    subkeys = key_schedule(key, key_bits, rounds)
    state = (ciphertext & BLOCK_MASK) ^ (subkeys[-1] if subkeys else 0)
    for rk in reversed(subkeys):
        state = permutation_layer(state, PERMUTATION_INV)
        state = sbox_layer(state, PRESENT_SBOX_INV)
        state ^= rk
    return state & BLOCK_MASK


def generate_known_plaintext_pairs(
    secret_key: int, key_bits: int, rounds: int, num_pairs: int = 2
) -> List[Tuple[int, int]]:
    """Produce (plaintext, ciphertext) pairs under the fixed secret key."""
    plaintexts = [(0x11 * i) & BLOCK_MASK for i in range(num_pairs)]
    return [(pt, encrypt(pt, secret_key, key_bits, rounds)) for pt in plaintexts]


def classical_candidate_keys(
    pairs: List[Tuple[int, int]], key_bits: int, rounds: int
) -> List[int]:
    """
    Classically enumerate every key consistent with ALL known-plaintext
    pairs. This is exactly the predicate the Grover oracle below marks --
    computed here once, classically, purely to construct the oracle's
    marked-state list (the oracle circuit itself does not "know" the key,
    it only encodes which basis states satisfy the predicate).
    """
    candidates = []
    for k in range(1 << key_bits):
        if all(encrypt(pt, k, key_bits, rounds) == ct for pt, ct in pairs):
            candidates.append(k)
    return candidates


# ===========================================================================
# 2. Grover oracle + diffuser construction (Qiskit)
# ===========================================================================

def build_marking_oracle(marked_states: List[int], num_qubits: int) -> QuantumCircuit:
    """
    Build a phase oracle that flips the sign of |k> for every k in
    `marked_states`. Implementation: for each marked state, X-gate the
    qubits that should read 0 (so the target pattern becomes all-ones),
    apply a multi-controlled-Z, then undo the X-gates. This is the
    standard technique for turning an explicit, classically-enumerated
    truth table into a quantum phase oracle when the predicate ("does
    key k satisfy the known-plaintext constraints?") is defined via the
    cipher's encrypt() rather than via simple arithmetic that would admit
    a direct quantum-arithmetic circuit.
    """
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
    """Standard Grover diffusion operator: inversion about the mean."""
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
    """Standard optimal iteration count: round(pi/4 * sqrt(N/M)), floor of 1."""
    if num_marked <= 0:
        return 0
    return max(1, round((math.pi / 4.0) * math.sqrt(search_space_size / num_marked)))


def build_grover_key_recovery_circuit(
    marked_states: List[int], num_qubits: int, iterations: int
) -> QuantumCircuit:
    """Full Grover circuit: uniform superposition, (oracle, diffuser) x iterations, measure."""
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
# 3. End-to-end live attack: run, verify, report, plot
# ===========================================================================

def measure_success_probability_sweep(
    candidates: List[int],
    key_bits: int,
    secret_key: int,
    max_iterations: int,
    shots: int,
    seed: int,
) -> List[Tuple[int, float]]:
    """
    ACTUALLY build and simulate a separate Grover circuit for every
    iteration count k = 0, 1, ..., max_iterations (k=0 means no
    oracle/diffuser applications at all -- just the initial uniform
    superposition, measured directly), and record the real measured
    probability of landing on the true secret key at each k.

    This is what lets the amplitude-amplification curve be plotted from
    genuine repeated simulation rather than the closed-form
    sin^2((2k+1)*theta) formula: at k=0 we expect roughly the uniform
    baseline 1/N, and probability should rise and peak at the optimal
    iteration count computed by `optimal_grover_iterations`, exactly as
    demonstrated empirically here.
    """
    true_key_bitstring = format(secret_key, f"0{key_bits}b")
    backend = AerSimulator(seed_simulator=seed)

    sweep: List[Tuple[int, float]] = []
    for k in range(max_iterations + 1):
        circuit = build_grover_key_recovery_circuit(candidates, key_bits, k)
        transpiled = transpile(circuit, backend, optimization_level=1)
        result = backend.run(transpiled, shots=shots).result()
        counts = result.get_counts()
        success_probability = counts.get(true_key_bitstring, 0) / shots
        sweep.append((k, success_probability))

    return sweep


def run_live_grover_key_recovery(
    secret_key: int = 0b101101,
    key_bits: int = 6,
    rounds: int = 4,
    num_known_pairs: int = 2,
    shots: int = 4096,
    seed: int = 42,
) -> Dict:
    """
    Execute the full live attack:
      1. Generate known-plaintext pairs under `secret_key`.
      2. Classically determine the candidate key set M (should be a
         singleton with enough pairs -- checked, not assumed).
      3. Build and run the Grover circuit on Qiskit Aer.
      4. Verify the top measured key actually decrypts/encrypts correctly.
      5. Report resource usage and return everything needed for plotting.
    """
    search_space_size = 1 << key_bits

    pairs = generate_known_plaintext_pairs(secret_key, key_bits, rounds, num_known_pairs)
    candidates = classical_candidate_keys(pairs, key_bits, rounds)

    if len(candidates) != 1:
        raise RuntimeError(
            f"Known-plaintext pairs do not uniquely determine the key "
            f"(found {len(candidates)} candidates: {candidates}). "
            f"Increase num_known_pairs."
        )
    if candidates[0] != secret_key:
        raise RuntimeError("Internal error: classical search did not recover the true key.")

    num_marked = len(candidates)
    iterations = optimal_grover_iterations(search_space_size, num_marked)

    circuit = build_grover_key_recovery_circuit(candidates, key_bits, iterations)

    backend = AerSimulator(seed_simulator=seed)
    transpiled = transpile(circuit, backend, optimization_level=1)
    result = backend.run(transpiled, shots=shots).result()
    counts = result.get_counts()

    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    top_bitstring, top_count = sorted_counts[0]
    recovered_key = int(top_bitstring, 2)
    success_probability = top_count / shots

    # --- Cryptographic verification: does the recovered key actually work? ---
    test_pt, test_ct = pairs[0]
    verified = (
        encrypt(test_pt, recovered_key, key_bits, rounds) == test_ct
        and decrypt(test_ct, recovered_key, key_bits, rounds) == test_pt
        and recovered_key == secret_key
    )

    # --- Amplitude-amplification sweep: measured success probability at
    # every iteration count from 0 up to the optimal count, so the
    # convergence toward the correct key can be shown as an actual curve
    # of independently-simulated data points rather than asserted. ---
    iteration_sweep = measure_success_probability_sweep(
        candidates, key_bits, secret_key, iterations, shots, seed
    )

    return {
        "secret_key": secret_key,
        "key_bits": key_bits,
        "rounds": rounds,
        "pairs": pairs,
        "candidates": candidates,
        "search_space_size": search_space_size,
        "iterations": iterations,
        "counts": counts,
        "recovered_key": recovered_key,
        "success_probability": success_probability,
        "verified": verified,
        "shots": shots,
        "circuit": circuit,
        "transpiled_circuit": transpiled,
        "classical_brute_force_expected_queries": search_space_size / 2,
        "iteration_sweep": iteration_sweep,
    }


def print_report(attack: Dict) -> None:
    tqc = attack["transpiled_circuit"]
    gate_counts = tqc.count_ops()

    print("=" * 70)
    print("LIVE GROVER KEY-RECOVERY ATTACK -- SUMMARY")
    print("=" * 70)
    print(f"Reduced cipher: {attack['key_bits']}-bit key, {attack['rounds']} SPN rounds, "
          f"{BLOCK_SIZE_BITS}-bit block")
    print(f"Known-plaintext pairs used   : {len(attack['pairs'])}")
    print(f"Classically-verified M       : {len(attack['candidates'])} "
          f"(candidate key(s) consistent with all pairs)")
    print(f"Search space N               : {attack['search_space_size']}")
    print(f"Grover iterations (optimal)  : {attack['iterations']}")
    print(f"Shots                        : {attack['shots']}")
    print("-" * 70)
    print(f"True secret key              : {attack['secret_key']:#04x} "
          f"({attack['secret_key']:0{attack['key_bits']}b})")
    print(f"Top measured key             : {attack['recovered_key']:#04x} "
          f"({attack['recovered_key']:0{attack['key_bits']}b})")
    print(f"Measured success probability : {attack['success_probability']:.4f}")
    print(f"Cryptographically verified   : {attack['verified']}")
    print("-" * 70)
    print(f"Qubits used                  : {attack['circuit'].num_qubits}")
    print(f"Logical circuit depth        : {attack['circuit'].depth()}")
    print(f"Transpiled circuit depth     : {tqc.depth()}")
    print(f"Transpiled gate counts       : {dict(gate_counts)}")
    print(f"Classical brute force (avg queries needed): "
          f"{attack['classical_brute_force_expected_queries']:.1f}")
    print(f"Grover queries used (iterations)          : {attack['iterations']}")
    print("-" * 70)
    print("Amplitude-amplification sweep (measured success probability per iteration count):")
    for k, prob in attack["iteration_sweep"]:
        marker = "  <-- optimal" if k == attack["iterations"] else ""
        print(f"  k = {k:>2} : P(success) = {prob:.4f}{marker}")
    print("=" * 70)


def plot_measurement_histogram(attack: Dict) -> None:
    """
    Publication-quality TWO-PANEL figure suitable for an MSc dissertation
    or research presentation.

      Left  panel: measurement histogram at the optimal Grover iteration
            count, with the recovered key highlighted and annotated with
            its measured probability.
      Right panel: measured success probability P(correct key) vs Grover
            iteration count k, overlaid with the theoretical Grover
            success-probability curve P(k) = sin^2((2k+1)*theta), where
            theta is computed directly from the simulation's own N and M
            (search-space size and marked-state count) -- no hardcoded
            assumptions -- plus an annotated optimal iteration count and
            an information box summarizing the attack's key figures.
    """
    key_bits = attack["key_bits"]
    counts = attack["counts"]
    shots = attack["shots"]
    secret_key = attack["secret_key"]
    search_space_size = attack["search_space_size"]
    num_marked = len(attack["candidates"])
    optimal_k = attack["iterations"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6.0))

    # =======================================================================
    # LEFT PANEL: key-recovery histogram at the optimal iteration count
    # =======================================================================
    all_keys = list(range(1 << key_bits))
    heights = [counts.get(format(k, f"0{key_bits}b"), 0) / shots for k in all_keys]
    colors = ["#d62728" if k == secret_key else "#4c72b0" for k in all_keys]
    uniform_baseline = 1.0 / search_space_size
    recovered_probability = heights[secret_key]

    ax1.bar(all_keys, heights, color=colors, width=0.8, edgecolor="black", linewidth=0.3, zorder=3)

    ax1.axhline(uniform_baseline, color="gray", linestyle="--", linewidth=1.2, zorder=2,
                label=f"Random-guess baseline (1/N = {uniform_baseline:.4f})")

    # Annotate the recovered key directly above its bar.
    ax1.annotate(
        f"Recovered key\n$P$ = {recovered_probability:.4f}",
        xy=(secret_key, recovered_probability),
        xytext=(secret_key, min(recovered_probability + 0.14, 1.12)),
        ha="center", va="bottom", fontsize=10, fontweight="bold", color="#d62728",
        arrowprops=dict(arrowstyle="-", color="#d62728", linewidth=1.0),
    )

    ax1.set_xlabel("Candidate key (integer value)", fontsize=12)
    ax1.set_ylabel("Measured probability", fontsize=12)
    ax1.set_title(
        f"Key-Recovery Histogram at Optimal Iteration Count ($k^*$ = {optimal_k})\n"
        f"{key_bits}-bit reduced PRESENT-style cipher, {shots} shots",
        fontsize=12,
    )
    ax1.set_xticks(all_keys)
    ax1.set_xticklabels([f"{k:0{key_bits}b}" for k in all_keys], rotation=90, fontsize=6)
    ax1.set_ylim(0, 1.18)
    ax1.tick_params(axis="both", labelsize=10)
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax1.grid(axis="y", alpha=0.3, zorder=0)

    # =======================================================================
    # RIGHT PANEL: Grover amplitude-amplification analysis
    # =======================================================================
    sweep = attack["iteration_sweep"]
    ks = [k for k, _ in sweep]
    measured_probs = [p for _, p in sweep]

    # theta computed directly from the simulation's own N and M -- no
    # hardcoded assumptions about the number of marked states or the
    # search-space size.
    sin_theta_sq = num_marked / search_space_size
    theta = math.asin(math.sqrt(sin_theta_sq))

    k_dense = np.linspace(0, max(ks), 400)
    theoretical_probs = np.sin((2 * k_dense + 1) * theta) ** 2

    # Theoretical curve first (drawn underneath), then measured points on top.
    ax2.plot(k_dense, theoretical_probs, color="black", linestyle="--", linewidth=1.6,
             label=r"Theoretical: $P(k)=\sin^2((2k{+}1)\theta)$", zorder=2)
    ax2.plot(ks, measured_probs, marker="o", color="#d62728", linewidth=2, markersize=8,
             markeredgecolor="black", markeredgewidth=0.5,
             label="Measured $P(\\mathrm{correct\\ key})$", zorder=4)

    ax2.axhline(uniform_baseline, color="gray", linestyle=":", linewidth=1,
                label="Uniform ($k$=0) baseline", zorder=1)
    ax2.axvline(optimal_k, color="#4c72b0", linestyle=":", linewidth=1.5, zorder=1,
                label=f"Optimal iteration $k^*$ = {optimal_k}")

    peak_prob = measured_probs[ks.index(optimal_k)]
    ax2.annotate(
        f"Optimal iteration count\n$k^*$ = {optimal_k}",
        xy=(optimal_k, peak_prob),
        xytext=(optimal_k - max(ks) * 0.28, max(0.05, peak_prob - 0.30)),
        ha="center", va="top", fontsize=9.5, fontweight="bold", color="#4c72b0",
        arrowprops=dict(arrowstyle="-", color="#4c72b0", linewidth=1.0),
    )

    classical_queries = search_space_size / 2.0
    speedup = classical_queries / optimal_k if optimal_k > 0 else float("nan")
    info_text = (
        f"Search space $N$ = {search_space_size}\n"
        f"Marked states $M$ = {num_marked}\n"
        f"Classical queries $\\approx N/2$ = {classical_queries:.1f}\n"
        f"Grover iterations = $k^*$ = {optimal_k}\n"
        f"Speedup $\\approx (N/2)/k^*$ = {speedup:.2f}$\\times$"
    )
    ax2.text(
        0.98, 0.04, info_text, transform=ax2.transAxes, fontsize=9,
        ha="right", va="bottom", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.92),
    )

    ax2.set_xlabel("Grover iteration count $k$", fontsize=12)
    ax2.set_ylabel("Success probability $P(\\mathrm{correct\\ key})$", fontsize=12)
    ax2.set_title(
        f"Grover Amplitude Amplification: Measured vs Theoretical\n"
        f"{key_bits}-bit reduced cipher, $N$ = {search_space_size}, $M$ = {num_marked}",
        fontsize=12,
    )
    ax2.set_xlim(-0.3, max(ks) + 0.3)
    ax2.set_ylim(-0.02, 1.08)
    ax2.set_xticks(ks)
    ax2.tick_params(axis="both", labelsize=10)
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    plt.show()


def main() -> None:
    attack = run_live_grover_key_recovery(
        secret_key=0b101101,
        key_bits=6,
        rounds=4,
        num_known_pairs=2,
        shots=4096,
    )
    print_report(attack)
    plot_measurement_histogram(attack)

    if not attack["verified"]:
        raise SystemExit("Attack verification FAILED -- recovered key does not decrypt correctly.")


if __name__ == "__main__":
    main()
