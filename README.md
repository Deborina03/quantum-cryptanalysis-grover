# Quantum Cryptanalysis of Lightweight Block Ciphers Using Grover's Algorithm

A research-style implementation of quantum-assisted key recovery using Grover's algorithm on a reduced PRESENT-inspired lightweight block cipher. The project includes oracle construction, noise-aware simulation, resource analysis, benchmarking, scaling validation, and defensive mitigation studies implemented in Qiskit.

Unlike purely theoretical complexity visualizations, every figure in this repository is generated from actual algorithmic implementations executed on the Qiskit Aer simulator.

---

## Key Results at a Glance

| Metric | Value |
|----------|----------|
| Recovered Key Success Probability | 99.6% |
| Optimal Grover Iterations | 6 |
| Fitted Grover Scaling Exponent | 0.519 |
| Theoretical Scaling Exponent | 0.500 |
| Relative Deviation | 3.8% |
| R² Score | 0.9986 |
| Pearson Correlation | 0.9997 |

---

## Why a Reduced Cipher?

Real PRESENT is a 64-bit block / 80-bit key ISO/IEC-standardized lightweight cipher. Simulating a Grover oracle for the full-scale cipher is computationally infeasible on a classical machine because the search space grows exponentially.

This repository preserves the key design elements of PRESENT:

- S-box substitution
- Bit-permutation diffusion layer
- Round-constant-driven key schedule

while reducing the key/block sizes to a range that allows a genuine Grover oracle to be constructed and simulated. The goal is not to attack full PRESENT-80, but to experimentally study quantum key-recovery behavior, scaling laws, noise sensitivity, and mitigation strategies in a realistic yet tractable setting.

---

## Results

### 1. Grover Key Recovery

Demonstration of successful key recovery using Grover's algorithm on a reduced PRESENT-style cipher.

![Grover Recovery](figures/live_grover.png)

---

### 2. Noise-Aware Quantum Cryptanalysis

Impact of depolarizing noise on Grover search performance.

![Noise Analysis](figures/noise_analysis.png)

---

### 3. Classical vs Grover vs BHT Search Cost

Comparison of measured classical search, measured Grover search, and theoretical BHT collision-search complexity.

![BHT Comparison](figures/bht_vs_grover.png)

---

### 4. Experimental Validation of Grover Scaling

Measured attack scaling compared with the theoretical √N prediction.

![Scaling Validation](figures/scaling_validation.png)

---

### 5. Key Rotation as a Defense Mechanism

Effect of key rotation frequency on attacker cost and defender exposure window.

![Key Rotation Defense](figures/key_rotation.png)

---

### 6. Key-Size Mitigation Analysis

Resource-growth study showing how larger key sizes increase quantum attack cost and circuit requirements.

![Key Size Mitigation](figures/keysize_mitigation.png)

---

## Repository Structure

```text
src/
├── 01_simulation_live_grover_key_recovery.py
├── 02_simulation_noise_quantum.py
├── simulation_bht_vs_grover.py
├── simulation_bht_vs_grover_scaling.py
├── simulation_key_rotation_defense.py
├── solution_keysize_mitigation.py

figures/
├── live_grover.png
├── noise_analysis.png
├── bht_vs_grover.png
├── scaling_validation.png
├── key_rotation.png
└── keysize_mitigation.png

requirements.txt
README.md
```

---

## What Each Script Does

### `01_simulation_live_grover_key_recovery.py`

Builds a known-plaintext Grover oracle from a reduced PRESENT-inspired cipher, performs quantum key recovery on Qiskit Aer, validates the recovered key, and measures amplitude amplification across Grover iterations.

**Key Result:** 99.6% measured success probability at the optimal Grover iteration count.

---

### `02_simulation_noise_quantum.py`

Executes the same attack circuit under Qiskit Aer noise models using depolarizing error channels and compares ideal versus noisy execution.

**Key Result:** Demonstrates rapid degradation of amplitude amplification under realistic noise levels.

---

### `simulation_bht_vs_grover.py`

Benchmarks:

- Classical brute-force search
- Grover search
- Theoretical BHT collision-search complexity

using measured attack costs and simulation results.

---

### `simulation_bht_vs_grover_scaling.py`

Performs empirical scaling analysis across multiple key sizes and compares fitted power-law exponents against theoretical predictions.

Also reports:

- Circuit depth
- Qubit count
- Two-qubit gate count
- Resource growth trends

---

### `simulation_key_rotation_defense.py`

Studies the effectiveness of periodic key rotation as a mitigation strategy by modeling attack-cost accumulation versus data-exposure windows.

---

### `solution_keysize_mitigation.py`

Experimentally validates Grover's quadratic speedup by comparing measured Grover iteration counts against theoretical √N scaling predictions.

**Key Result:** Measured exponent 0.519 vs theoretical exponent 0.500.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

Run any script directly:

```bash
python src/01_simulation_live_grover_key_recovery.py
python src/02_simulation_noise_quantum.py
python src/simulation_bht_vs_grover.py
python src/simulation_bht_vs_grover_scaling.py
python src/simulation_key_rotation_defense.py
python src/solution_keysize_mitigation.py
```

Each script prints quantitative results to the terminal and generates publication-style figures using Matplotlib.

---

## Scope and Limitations

This repository explicitly distinguishes between measured simulation results and theoretical references.

- The reduced cipher is a stand-in for PRESENT and is not an attack on full PRESENT-80 or AES.
- BHT appears only as a theoretical comparison curve and is not simulated.
- All Grover attack experiments, scaling studies, resource analyses, and noise evaluations are executed on Qiskit Aer.
- No claims are made regarding the practical quantum security of real-world ciphers beyond the scales directly studied here.

---

## Dependencies

- qiskit
- qiskit-aer
- numpy
- matplotlib

See `requirements.txt` for exact versions.

---
