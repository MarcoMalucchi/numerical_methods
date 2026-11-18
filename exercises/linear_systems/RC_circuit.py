import numpy as np
import matplotlib.pyplot as plt
import cmath
from linear_systems.elimination_algorithms import Solve, gauss_jordan_inverse

'''
GOAL OF THE SCRIPT: to solve an RC circuit supplied with AC currents, everything is known, just need to solve the linear system
using a little of complex algebra.
'''

# --- Resistances, Capacitances ---
R = np.array([1,2,1,2,1,2])*1e3   # Ohm
C = np.array([1, 0.5])*1e-6   # Farad
w = 1e3 # Hz, supply frequency
Vcc = 3 # V, supply amplitude

# --- Assosiated linear system ---

A00 = 1/R[0]+1/R[3]+1j*w*C[0]
A01 = -1j*w*C[0]

A10 = A01
A11 = 1/R[1]+1/R[4]+1j*w*(C[0]+C[1])
A12 = -1j*w*C[1]

A21 = A12
A22 = 1/R[2]+1/R[5]+1j*w*C[1]

A = np.array([
    [A00, A01, 0],
    [A10, A11, A12],
    [0, A21, A22],
], dtype=np.complex128)

b = Vcc / R[:3]

voltages = np.linalg.solve(A, b)
voltages_1, _ = Solve(A, b)

header = f"{'Node':<6} {'np.linalg.solve':>24} {'Solve':>24}"

print("Voltages in real and imaginary parts")
print(header)
for i, (v_numpy, v_own) in enumerate(zip(voltages, voltages_1, strict=True)):
    print(f"V{i:<5} {v_numpy:>22.2f} V {v_own:>22.2f} V")

print("\nVoltages in terms of magnitude and phase")
print(header)
for i, (v_numpy, v_own) in enumerate(zip(voltages, voltages_1, strict=True)):
    mag_numpy, phase_numpy = cmath.polar(v_numpy)
    mag_own, phase_own = cmath.polar(v_own)

    numpy_text = f"{mag_numpy:.2f} V ∠ {np.degrees(phase_numpy):.2f}°"
    own_text = f"{mag_own:.2f} V ∠ {np.degrees(phase_own):.2f}°"
    print(f"V{i:<5} {numpy_text:>24} {own_text:>24}")
