# ============================================================
# Simulation: Key Rotation Defense Against Grover Attack
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import math

# -----------------------------
# 1. Parameters
# -----------------------------

key_bits = 8   # lightweight key size
num_packets = 50  # number of transmitted packets

# Grover complexity formula:
# T ≈ (π/4) * sqrt(2^n)

def grover_effort(n):
    return (math.pi / 4) * math.sqrt(2 ** n)

static_grover = grover_effort(key_bits)

# -----------------------------
# 2. Static Key Scenario
# -----------------------------
# Only one Grover search required

static_total_effort = static_grover

# -----------------------------
# 3. Rotating Key Scenario
# -----------------------------
# Each packet uses new key
# Attacker must run Grover separately per packet

packet_counts = np.arange(1, num_packets + 1)
rotating_total_effort = static_grover * packet_counts

# -----------------------------
# 4. Print Numerical Results
# -----------------------------

print("\n===== KEY ROTATION DEFENSE SIMULATION =====")
print(f"Key size: {key_bits} bits")
print(f"Grover effort per key ≈ {static_grover:.2f} iterations")

print("\nStatic Key Scenario:")
print(f"Total attack effort ≈ {static_total_effort:.2f}")

print("\nRotating Key Scenario:")
print(f"Packets simulated: {num_packets}")
print(f"Total effort after {num_packets} packets ≈ {rotating_total_effort[-1]:.2f}")

print("\nInterpretation:")
print("• Static key: attacker runs Grover once.")
print("• Rotating keys: attacker must restart Grover per packet.")
print("• Total attack effort grows linearly with number of packets.")

# -----------------------------
# 5. Plot
# -----------------------------

plt.figure()

plt.plot(packet_counts, rotating_total_effort)
plt.axhline(static_total_effort, linestyle='--')

plt.xlabel("Number of Packets")
plt.ylabel("Total Grover Iterations Required")
plt.title("Impact of Key Rotation on Grover Attack Cost")

plt.show()
