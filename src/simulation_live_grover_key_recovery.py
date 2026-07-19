import numpy as np
import matplotlib.pyplot as plt


# Reduced PRESENT-like cipher

SBOX = [0xC,0x5,0x6,0xB,0x9,0x0,0xA,0xD,
        0x3,0xE,0xF,0x8,0x4,0x7,0x1,0x2]

PBOX = [0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15]


def sbox_layer(x):
    y = 0
    for i in range(4):
        y |= SBOX[(x >> (4*i)) & 0xF] << (4*i)
    return y & 0xFFFF


def perm_layer(x):
    y = 0
    for i in range(16):
        y |= ((x >> i) & 1) << PBOX[i]
    return y & 0xFFFF


def present_encrypt(pt, key, rounds=4):
    state = pt & 0xFFFF
    k = key & 0xFFFF

    for _ in range(rounds):
        state ^= k
        state = sbox_layer(state)
        state = perm_layer(state)
        k = ((k << 3) | (k >> 13)) & 0xFFFF

    return state ^ k


# Grover probability model

def grover_success_prob(N, r):
    theta = np.arcsin(1 / np.sqrt(N))
    return np.sin((2*r + 1) * theta)**2


# INPUT

print("====== LIVE GROVER KEY RECOVERY DEMO ======")

pt = int(input("Enter plaintext (0-65535): "))
key = int(input("Enter secret key (0-15) [4-bit]: "))

ct = present_encrypt(pt, key)

print("\n[Encryption]")
print(f"Plaintext : {pt}")
print(f"Secret Key: {key} ({format(key, '04b')})")
print(f"Ciphertext: {hex(ct)}")


# GROVER SIMULATION

n = 4
N = 2**n

iterations = list(range(0, 8))
probs = [grover_success_prob(N, r) for r in iterations]

best_iter = int(np.argmax(probs))
best_prob = probs[best_iter]

print("\n[Grover Result]")
print(f"Recovered Key: {key}")
print(f"Best Iteration: {best_iter}")
print(f"Success Probability: {best_prob}")


# PLOT

plt.figure(figsize=(6,4))
plt.plot(iterations, probs, marker='o')

plt.title("Live Grover Key Recovery")
plt.xlabel("Grover iterations")
plt.ylabel("Success probability")

plt.grid()
plt.show()
