import numpy as np
import matplotlib.pyplot as plt

print("\n===== SOLUTION 1: KEY SIZE MITIGATION =====\n")

# Key sizes
bits = np.arange(64, 257, 32)

# Security levels in bits (log2 scale)
classical_security = bits
quantum_security = bits / 2   # Grover halves security bits

for b, c, q in zip(bits, classical_security, quantum_security):
    print(f"Key size: {b} bits")
    print(f" Classical security ~ 2^{int(c)}")
    print(f" Quantum security (Grover) ~ 2^{int(q)}\n")

plt.figure(figsize=(8,6))

plt.plot(bits, classical_security, marker='o',
         label="Classical security (bits)")

plt.plot(bits, quantum_security, marker='o',
         label="Quantum security with Grover")

plt.xlabel("Key Size (bits)")
plt.ylabel("Effective Security (bits)")
plt.title("Mitigation Strategy: Increasing Key Size\n(Post-Quantum Perspective)")
plt.legend()
plt.grid(True)

plt.show()

