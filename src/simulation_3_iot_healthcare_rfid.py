import math
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer



SBOX = [
    0xC, 0x5, 0x6, 0xB,
    0x9, 0x0, 0xA, 0xD,
    0x3, 0xE, 0xF, 0x8,
    0x4, 0x7, 0x1, 0x2
]

PBOX = [0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15]


def sbox_layer(x):
    out = 0
    for i in range(4):
        out |= SBOX[(x >> (4*i)) & 0xF] << (4*i)
    return out & 0xFFFF


def perm_layer(x):
    out = 0
    for i in range(16):
        out |= ((x >> i) & 1) << PBOX[i]
    return out & 0xFFFF


def present_encrypt(pt, key, rounds=5):
    state = pt & 0xFFFF
    k = key & 0xFFFF
    for _ in range(rounds):
        state ^= k
        state = sbox_layer(state)
        state = perm_layer(state)
        k = ((k << 3) | (k >> 13)) & 0xFFFF
    return state ^ k


# IoT Healthcare RFID Encoding

def encode_healthcare_data(patient_id, heart_rate):
    """
    Encode healthcare data into 16-bit block:
    [PatientID (8 bits) | HeartRate (8 bits)]
    """
    return ((patient_id & 0xFF) << 8) | (heart_rate & 0xFF)


def decode_healthcare_data(block):
    patient_id = (block >> 8) & 0xFF
    heart_rate = block & 0xFF
    return patient_id, heart_rate


# Grover Threat Model (Key search scaling illustration)

def grover_key_effort_plot():
    key_bits = np.arange(4, 13)
    N = 2 ** key_bits


    grover_iters = (math.pi / 4) * np.sqrt(N)

    plt.figure(figsize=(8,5))
    plt.plot(key_bits, grover_iters, marker='o')
    plt.xlabel("Key size (bits)")
    plt.ylabel("Grover iterations")
    plt.title("Grover Attack Effort on IoT Lightweight Keys")
    plt.grid(True)
    plt.show()


# MAIN SIMULATION

if __name__ == "__main__":

    # Fixed seed for reproducibility
    np.random.seed(42)

    # Example IoT healthcare data
    patient_id = 0x23        # example RFID tag
    heart_rate = 85          # BPM

    plaintext = encode_healthcare_data(patient_id, heart_rate)

    # Secret key (lightweight)
    secret_key = np.random.randint(0, 2**16)

    ciphertext = present_encrypt(plaintext, secret_key)

    print("\n========== IoT HEALTHCARE RFID SIMULATION ==========")
    print("Patient ID   :", hex(patient_id))
    print("Heart Rate   :", heart_rate, "BPM")
    print("Plaintext    :", hex(plaintext))
    print("Secret Key   :", hex(secret_key))
    print("Ciphertext   :", hex(ciphertext))

    # Decode to verify correctness
    decoded_id, decoded_hr = decode_healthcare_data(plaintext)
    print("\n[Verification]")
    print("Decoded Patient ID :", hex(decoded_id))
    print("Decoded Heart Rate :", decoded_hr, "BPM")

    # Grover effort illustration
    grover_key_effort_plot()

