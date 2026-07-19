import numpy as np
import matplotlib.pyplot as plt

print("\n===== SOLUTION 3: NOISE-AWARE QUANTUM MITIGATION =====\n")

np.random.seed(1)

iterations = np.arange(1, 21)

# Ideal Grover success probability
ideal_prob = np.sin((2*iterations+1)/8)**2

# Noise model: amplitude damping effect
noise_factor = 0.15

noisy_prob = ideal_prob * (1 - noise_factor)

print("Peak success probability (ideal):",
      round(max(ideal_prob), 3))

print("Peak success probability (noisy):",
      round(max(noisy_prob), 3))

print("\nInterpretation:")
print("- Noise lowers quantum attack reliability.")
print("- Real quantum computers suffer decoherence.")
print("- This reduces practical Grover effectiveness.")

plt.figure()

plt.plot(iterations, ideal_prob,
         marker='o', label="Ideal quantum attack")

plt.plot(iterations, noisy_prob,
         marker='o', label="With hardware noise")

plt.xlabel("Grover iterations")
plt.ylabel("Success probability")
plt.title("Noise Impact on Quantum Cryptanalysis")
plt.legend()
plt.grid(True)

plt.show()

