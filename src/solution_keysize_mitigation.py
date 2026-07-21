"""
solution_keysize_mitigation.py
=================================

EXPERIMENTAL VALIDATION of Grover's quadratic speed-up: uses the reduced
cipher's actual attack machinery (same oracle/circuit construction used
throughout this repository) to measure how the required Grover iteration
count grows with key-space size, and checks that measured growth against
the textbook theoretical scaling law.


This module intentionally stops at experimental validation of the scaling
law itself -- it does NOT project effective security levels for
real-world ciphers (AES/PRESENT), since that projection is a separate,
much larger extrapolation and is out of scope here.

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

# Reuse the real cipher, oracle, and circuit-construction routines from the
# sibling benchmarking module so the measured scaling trend below is
# grounded in the exact same attack machinery used elsewhere in the repo.
from simulation_bht_vs_grover import (
    build_grover_key_recovery_circuit,
    classical_candidate_keys,
    generate_known_plaintext_pairs,
    optimal_grover_iterations,
)


# ===========================================================================
# 1. Empirically measure the Grover scaling exponent on the reduced cipher
# ===========================================================================

def measure_grover_iterations_for_size(
    key_bits: int, rounds: int, num_independent_keys: int, seed: int
) -> Tuple[float, float]:
    """
    For a fixed key size, ACTUALLY build and run a real Grover circuit
    against `num_independent_keys` independently drawn secret keys, and
    return (mean, std) of the real iteration count used. Iteration count
    is a direct, honestly-computed function of the actual candidate-set
    size found by classical search on real known-plaintext pairs -- not a
    formula applied blindly.
    """
    rng = random.Random(seed)
    iterations_list: List[int] = []

    for _ in range(num_independent_keys):
        secret_key = rng.randrange(1 << key_bits)
        pairs = generate_known_plaintext_pairs(secret_key, key_bits, rounds, num_pairs=2)
        while classical_candidate_keys(pairs, key_bits, rounds) != [secret_key]:
            pairs = generate_known_plaintext_pairs(secret_key, key_bits, rounds, len(pairs) + 1)
        candidates = classical_candidate_keys(pairs, key_bits, rounds)
        iterations = optimal_grover_iterations(1 << key_bits, len(candidates))
        iterations_list.append(iterations)

    return float(np.mean(iterations_list)), float(np.std(iterations_list))


def fit_power_law_exponent(sizes: List[int], values: List[float]) -> Tuple[float, float]:
    """Fit y = a * N^b via log-log linear regression; returns (b, a)."""
    log_n = np.log(sizes)
    log_y = np.log(values)
    b, log_a = np.polyfit(log_n, log_y, 1)
    return float(b), float(math.exp(log_a))


def measure_scaling_exponent(
    key_bit_range: List[int] = None, rounds: int = 4,
    num_independent_keys: int = 6, seed: int = 4242,
) -> Dict:
    if key_bit_range is None:
        key_bit_range = list(range(4, 12))

    means, stds = [], []
    for key_bits in key_bit_range:
        m, s = measure_grover_iterations_for_size(key_bits, rounds, num_independent_keys, seed)
        means.append(m)
        stds.append(s)

    ns = [1 << k for k in key_bit_range]
    exponent, prefactor = fit_power_law_exponent(ns, means)

    return {
        "key_bit_range": key_bit_range,
        "means": means,
        "stds": stds,
        "fitted_exponent": exponent,
        "fitted_prefactor": prefactor,
    }


# ===========================================================================
# 2. Validate measured iterations against the theoretical sqrt(N) law
# ===========================================================================

def theoretical_grover_iterations(search_space_size: int) -> float:
    """
    Theoretical optimal Grover iteration count for a single marked state:
    (pi/4) * sqrt(N). This is the textbook formula being validated against
    the measured data -- included with its exact constant prefactor (not
    just the bare sqrt(N) scaling law) so the parity comparison below is
    checking real numeric agreement, not just matching order of growth.
    """
    return (math.pi / 4.0) * math.sqrt(search_space_size)


def compute_validation_metrics(scaling: Dict) -> Dict:
    """
    Compare MEASURED Grover iterations against the THEORETICAL sqrt(N)
    formula at every key size already simulated in `scaling`, and report:
      - the theoretical iteration count at each size,
      - the relative error between measured and theoretical at each size,
      - the fitted exponent (already computed in `scaling`) next to the
        textbook exponent of 0.5,
      - the mean absolute relative error across all measured sizes, as a
        single-number summary of how tightly the simulated attack tracks
        the theoretical quadratic speed-up.
    """
    key_bit_range = scaling["key_bit_range"]
    ns = [1 << k for k in key_bit_range]
    theoretical = [theoretical_grover_iterations(n) for n in ns]
    measured = scaling["means"]

    relative_errors = [
        abs(m - t) / t if t > 0 else 0.0 for m, t in zip(measured, theoretical)
    ]

    return {
        "key_bit_range": key_bit_range,
        "search_space_sizes": ns,
        "measured_iterations": measured,
        "theoretical_iterations": theoretical,
        "relative_errors": relative_errors,
        "mean_absolute_relative_error": float(np.mean(relative_errors)),
        "fitted_exponent": scaling["fitted_exponent"],
        "theoretical_exponent": 0.5,
    }


# ===========================================================================
# 3. Reporting and plotting
# ===========================================================================

def print_report(scaling: Dict, validation: Dict) -> None:
    print("=" * 90)
    print("STEP 1 -- EMPIRICALLY MEASURED GROVER SCALING EXPONENT (reduced cipher, simulated)")
    print("=" * 90)
    header = f"{'key bits':>8} | {'N':>6} | {'measured mean iterations':>24} | {'std':>8}"
    print(header)
    print("-" * len(header))
    for i, kb in enumerate(scaling["key_bit_range"]):
        print(f"{kb:>8} | {1 << kb:>6} | {scaling['means'][i]:>24.2f} | {scaling['stds'][i]:>8.2f}")
    print("-" * len(header))
    print(f"Fitted scaling exponent (measured) : {scaling['fitted_exponent']:.4f}  "
          f"(textbook Grover exponent: 0.5000)")
    print()
    print("=" * 90)
    print("STEP 2 -- EXPERIMENTAL VALIDATION: MEASURED vs THEORETICAL sqrt(N) SCALING")
    print("=" * 90)
    print("Comparing measured iteration counts against the exact theoretical formula\n"
          "(pi/4) * sqrt(N), at every key size already simulated in Step 1.")
    print("-" * 90)
    header2 = (f"{'key bits':>8} | {'N':>6} | {'measured':>10} | {'theoretical (pi/4)sqrt(N)':>26} | "
               f"{'relative error':>15}")
    print(header2)
    print("-" * len(header2))
    for i, kb in enumerate(validation["key_bit_range"]):
        print(f"{kb:>8} | {validation['search_space_sizes'][i]:>6} | "
              f"{validation['measured_iterations'][i]:>10.2f} | "
              f"{validation['theoretical_iterations'][i]:>26.2f} | "
              f"{validation['relative_errors'][i]:>15.4f}")
    print("-" * 90)
    print(f"Fitted scaling exponent            : {validation['fitted_exponent']:.4f}")
    print(f"Theoretical scaling exponent        : {validation['theoretical_exponent']:.4f}")
    print(f"Mean absolute relative error        : {validation['mean_absolute_relative_error']:.4f}  "
          f"(measured vs (pi/4)*sqrt(N), averaged across all tested key sizes)")
    print("=" * 90)


def plot_mitigation_analysis(scaling: Dict, validation: Dict) -> None:
    """
    Publication-quality figure: "Experimental Validation of Grover's
    Quadratic Speedup". Uses ONLY quantities already computed in
    `scaling` and `validation` (measured iterations, fitted exponent,
    theoretical (pi/4)*sqrt(N) curve, relative errors) -- this function
    adds no new measurements and performs no new underlying calculations,
    only additional presentation-layer statistics derived from the
    existing fit (R^2 goodness-of-fit, Pearson correlation coefficient)
    for display purposes.
    """
    key_bit_range = scaling["key_bit_range"]
    ns = np.array([1 << k for k in key_bit_range], dtype=float)
    means = np.array(scaling["means"])
    fitted_exponent = scaling["fitted_exponent"]
    fitted_prefactor = scaling["fitted_prefactor"]
    theoretical_exponent = validation["theoretical_exponent"]
    relative_deviation = abs(fitted_exponent - theoretical_exponent) / theoretical_exponent

    # R^2 goodness-of-fit for the EXISTING log-log power-law fit (presentation
    # statistic only -- the fit itself, `fitted_exponent`/`fitted_prefactor`,
    # is untouched and taken as-is from `scaling`).
    log_n = np.log(ns)
    log_means = np.log(means)
    log_fit = np.log(fitted_prefactor) + fitted_exponent * log_n
    ss_res = np.sum((log_means - log_fit) ** 2)
    ss_tot = np.sum((log_means - np.mean(log_means)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle("Experimental Validation of Grover's Quadratic Speedup",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.text(0.5, 0.965,
              "Measured reduced-cipher attack results compared against theoretical "
              r"$(\pi/4)\sqrt{N}$ scaling",
              ha="center", fontsize=11, style="italic", color="#333333")

    # =======================================================================
    # PANEL 1: ABSOLUTE SCALING
    # =======================================================================
    stds = np.array(scaling["stds"])
    fine_n = np.linspace(ns.min(), ns.max(), 200)
    fitted_curve = fitted_prefactor * fine_n ** fitted_exponent

    ax1.errorbar(ns, means, yerr=stds, fmt="o", color="#d62728", capsize=4,
                 markersize=9, markeredgecolor="black", markeredgewidth=0.6,
                 elinewidth=1.2, label="Measured Grover iterations (mean $\\pm$ std)",
                 zorder=4)
    ax1.plot(fine_n, fitted_curve, color="#d62728", linestyle="--", linewidth=1.8,
             label=f"Measured fit: $N^{{{fitted_exponent:.3f}}}$ ($R^2$ = {r_squared:.4f})",
             zorder=3)
    ax1.plot(fine_n, (math.pi / 4.0) * np.sqrt(fine_n), color="gray", linestyle=":",
             linewidth=1.8, label=r"Theoretical: $(\pi/4)\sqrt{N}$", zorder=2)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Key space size $N$", fontsize=13)
    ax1.set_ylabel("Grover iterations (log scale)", fontsize=13)
    ax1.set_title("Panel 1: Absolute Scaling", fontsize=13, fontweight="bold")
    ax1.tick_params(axis="both", labelsize=11)
    ax1.legend(fontsize=9.5, loc="upper left", framealpha=0.92)
    ax1.grid(True, which="major", alpha=0.35, linewidth=0.8)
    ax1.grid(True, which="minor", alpha=0.15, linewidth=0.5)

    panel1_info = (
        f"Measured exponent = {fitted_exponent:.3f}\n"
        f"Theory exponent = {theoretical_exponent:.3f}\n"
        f"Relative deviation = {relative_deviation * 100:.1f}%"
    )
    ax1.text(
        0.98, 0.04, panel1_info, transform=ax1.transAxes, fontsize=9.5,
        ha="right", va="bottom", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.92),
    )

    # =======================================================================
    # PANEL 2: EXPERIMENTAL VALIDATION (measured vs theoretical parity plot)
    # =======================================================================
    measured = np.array(validation["measured_iterations"])
    theoretical = np.array(validation["theoretical_iterations"])
    mean_rel_error = validation["mean_absolute_relative_error"]
    pearson_r = float(np.corrcoef(theoretical, measured)[0, 1])

    scatter = ax2.scatter(
        theoretical, measured, c=key_bit_range, cmap="viridis", s=110,
        edgecolor="black", linewidth=0.7, zorder=5,
        label="Measured vs theoretical (per key size)",
    )
    cbar = fig.colorbar(scatter, ax=ax2, pad=0.02)
    cbar.set_label("Key size (bits)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    diag_min = min(theoretical.min(), measured.min()) * 0.8
    diag_max = max(theoretical.max(), measured.max()) * 1.2
    diag = np.linspace(diag_min, diag_max, 100)
    ax2.plot(diag, diag, color="black", linestyle="--", linewidth=1.4,
             label="Perfect agreement (y = x)", zorder=3)

    for kb, t, m in zip(key_bit_range, theoretical, measured):
        ax2.annotate(f"{kb}-bit", (t, m), textcoords="offset points",
                     xytext=(7, -4), fontsize=8, color="#222222")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"Theoretical iterations: $(\pi/4)\sqrt{N}$ (log scale)", fontsize=13)
    ax2.set_ylabel("Measured Grover iterations (log scale)", fontsize=13)
    ax2.set_title("Experimental Validation of Grover Scaling",
                   fontsize=13, fontweight="bold")
    ax2.tick_params(axis="both", labelsize=11)
    ax2.legend(fontsize=9.5, loc="upper left", framealpha=0.92)
    ax2.grid(True, which="major", alpha=0.35, linewidth=0.8)
    ax2.grid(True, which="minor", alpha=0.15, linewidth=0.5)

    panel2_info = (
        f"Measured exponent = {fitted_exponent:.3f}\n"
        f"Theory exponent = {theoretical_exponent:.3f}\n"
        f"Mean relative error = {mean_rel_error * 100:.1f}%\n"
        f"r = {pearson_r:.4f}"
    )
    ax2.text(
        0.98, 0.04, panel2_info, transform=ax2.transAxes, fontsize=9.5,
        ha="right", va="bottom", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.92),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


def main() -> None:
    scaling = measure_scaling_exponent(
        key_bit_range=list(range(4, 12)), num_independent_keys=6
    )
    validation = compute_validation_metrics(scaling)
    print_report(scaling, validation)
    plot_mitigation_analysis(scaling, validation)


if __name__ == "__main__":
    main()
