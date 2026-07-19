# Quantum Cryptanalysis of Lightweight Block Ciphers Using Grover's Algorithm

## Overview

This repository investigates the impact of quantum search algorithms on lightweight cryptographic systems used in IoT and RFID healthcare applications.

The work evaluates:

- Grover-based key recovery
- BHT collision-search complexity
- Quantum attack scaling
- Key rotation defense mechanisms
- Key-size mitigation strategies
- Noise-aware quantum attack models
- RFID healthcare security scenarios

---

## Research Highlights

### Grover Key Recovery

Simulation of quantum-assisted key recovery against a reduced PRESENT-like cipher.

### BHT vs Grover Scaling

Comparison of Grover search and BHT collision-search complexity for lightweight cryptographic systems.

### IoT Healthcare RFID Security

Evaluation of lightweight encryption used in RFID-enabled healthcare monitoring systems.

### Key Rotation Defense

Analysis of how frequent key rotation increases the cost of Grover-based attacks.

### Key Size Mitigation

Post-quantum security analysis demonstrating how larger key sizes compensate for Grover speedup.

### Noise-Aware Quantum Analysis

Investigation of hardware noise effects on practical quantum cryptanalysis.

---

## Repository Structure

```text
src/
paper/
figures/
```

---

## Technologies

- Python
- Qiskit
- Qiskit Aer
- NumPy
- Matplotlib

---

## Paper

The accompanying manuscript is available in:

```text
paper/quantum_cryptanalysis_paper.pdf
```
