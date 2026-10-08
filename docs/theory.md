# MomentumPlane — Theory Notes

## 1. Coherent Injection and the Momentum Comb

Consider N wave-packets injected at lattice positions r_j = (x_j, y_j), each with the same wavevector k_0 = (k_x0, k_y0) and Gaussian envelope of width sigma:

    psi_j(x, y) = A * exp(-|r - r_j|^2 / (2 sigma^2)) * exp(i k_0 . r)

The total injected field is the coherent superposition:

    psi(r) = sum_j psi_j(r)

Taking the 2D Fourier transform:

    psi_tilde(k) = sum_j exp(i (k - k_0) . r_j) * A_tilde(k - k_0)

where A_tilde is the Fourier transform of the Gaussian envelope (another Gaussian of width 1/sigma).

The key insight: the sum sum_j exp(i Delta_k . r_j) is a discrete lattice structure factor. For a regular square sub-lattice with spacing s:

    |sum_j exp(i Delta_k . r_j)|^2 = [sin(N_x Delta_kx s / 2) / sin(Delta_kx s / 2)]^2
                                      * [sin(N_y Delta_ky s / 2) / sin(Delta_ky s / 2)]^2

This is the classic diffraction grating intensity: sharp peaks at Delta_kx = 2*pi*m/s, Delta_ky = 2*pi*n/s, with peak width ~ 1/(N_sites * s) and peak height ~ N_sites^2.

This is the momentum plane. The regular injection geometry writes a crystal into momentum space.

## 2. Discrete-Time Quantum Walk (DTQW)

A 2D DTQW lives on the Hilbert space H = H_position x H_coin, where H_position = C^N x C^N and H_coin = C^4 (four directions: up, right, down, left).

One evolution step:

    |psi(t+1)> = S * (I_pos x C) * |psi(t)>

### Coin Operator C

The Hadamard coin is the tensor product of two 2x2 Hadamards:

    H = (1/sqrt(2)) [[1, 1], [1, -1]]
    C_4 = H x H

The Grover coin is the diffusion operator:

    G = (2/N) J - I

where J is the all-ones matrix. Both are unitary (verified in tests).

### Conditional Shift S

    S |x, y, d> = |x + dx_d, y + dy_d, d>

with directions d in {up=(-1,0), right=(0,1), down=(1,0), left=(0,-1)}.

### Dispersion Relation

For an infinite lattice, the DTQW has two dispersion branches:

    E_pm(k_x, k_y) = pm arccos( (cos k_x + cos k_y) / sqrt(2) )   [Hadamard]

This means different momentum components propagate at different group velocities, causing the initially sharp momentum peaks to broaden and develop caustic structures over time.

## 3. Effect of Quantum Walk on Momentum Peaks

The DTQW does not change the positions of the momentum peaks (those are fixed by the injection geometry), but it modulates their:

1. Width — dispersion causes ballistic spreading ~ t
2. Relative intensity — phase accumulation re-weights peaks
3. Internal structure — caustics and ring patterns appear at long times
4. Coherence — with phase jitter > 0, peaks wash out (decoherence)

## 4. Phase Jitter as Decoherence

When each injected packet gets a random phase phi_j ~ N(0, delta^2), the structure factor becomes:

    |sum_j exp(i Delta_k . r_j + i phi_j)|^2

For large N and delta >= 1, the cross-terms average to zero, and the momentum comb washes into a broad background. This is a simple model of decoherence through spatial phase disorder.

## 5. Energy Conservation (Parseval)

The FFT satisfies Parseval's theorem:

    sum_{x,y} |psi(x,y)|^2 = (1/N^2) sum_{kx,ky} |psi_tilde(kx,ky)|^2

This is verified in test_synthesizer.py::test_parseval_energy_conservation.

The DTQW is unitary, so total probability is conserved at every step:

    sum |psi(t)|^2 = sum |psi(0)|^2   for all t

Verified in test_lattice.py::test_unitarity_preserves_norm.

## 6. Why This Matters

The momentum plane is a visual fingerprint of the injection geometry and dynamics. By analysing the peak positions, widths, and intensities, one can invert the problem: given a momentum-plane pattern, reconstruct the injection lattice and the quantum-walk parameters. This has analogues in:

- X-ray crystallography — momentum-space diffraction to atomic structure
- Quantum simulation — engineering specific momentum distributions
- Signal processing — coded aperture imaging
- High-performance computing — the DTQW is a benchmark for sparse linear algebra on accelerators
