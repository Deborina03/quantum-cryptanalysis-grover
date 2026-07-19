import numpy as np
import matplotlib.pyplot as plt

# Key/search space sizes (bits)
bits = np.arange(4, 33, 2)

# Search space
N = 2 ** bits

# Complexity estimates
classical_bruteforce = N
grover_complexity = np.sqrt(N)
bht_complexity = N ** (1/3)

plt.figure(figsize=(8,6))
plt.plot(bits, classical_bruteforce, label="Classical brute force")
plt.plot(bits, grover_complexity, label="Grover (√N)")
plt.plot(bits, bht_complexity, label="BHT variant (N^(1/3))")

plt.yscale("log")
plt.xlabel("Key/Search bits")
plt.ylabel("Operations (log scale)")
plt.title("Quantum Cryptanalysis Scaling:\nGrover vs BHT Variant")
plt.legend()
plt.grid(True)
plt.show()

# Print numerical comparison for reference
print("\nSample complexity comparison:")
for b in [8, 16, 24]:
    N = 2**b
    print(f"\nBits = {b}")
    print(f"Classical ~ {N:.2e}")
    print(f"Grover    ~ {np.sqrt(N):.2e}")
    print(f"BHT       ~ {N**(1/3):.2e}")

