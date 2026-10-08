"""
lattice.py — 量子跳跃与晶格演化

Implements a discrete-time quantum walk (DTQW) on a 2D lattice.  Each
evolution step applies a coin operator (Hadamard or general SU(4)) followed
by a conditional shift operator that moves the walker in a direction
determined by its internal coin state.

This is the "LatticeHop" module: it converts the initially injected
position-space wavefunction into a delocalised, phase-coherent superposition
through repeated unitary evolution.

Physics
-------
One DTQW step:

    |psi(t+1)> = S * C * |psi(t)>

where C is the coin operator acting on the internal (direction) degree of
freedom, and S is the conditional shift:

    S |x, y, d> = |x + dx_d, y + dy_d, d>

For a 2D walk with 4-direction coin (up/down/left/right), the Hilbert space
is C^N x C^N x C^4.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


# 4-direction basis: 0=up, 1=right, 2=down, 3=left
_DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def hadamard_coin() -> np.ndarray:
    """Return the 4x4 balanced Hadamard coin (tensor of two 2x2 Hadamards)."""
    H2 = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    return np.kron(H2, H2)


def grover_coin() -> np.ndarray:
    """Return the 4x4 Grover diffusion coin."""
    N = 4
    return 2 * np.ones((N, N), dtype=np.complex128) / N - np.eye(N, dtype=np.complex128)


@dataclass
class LatticeHop:
    """Discrete-time quantum walk evolution on a 2D lattice.

    Parameters
    ----------
    grid_size : int
        Lattice dimension (N x N).
    coin_type : str
        "hadamard" or "grover".
    coin_angle : float
        Rotation angle (radians) applied to the coin before each step;
        0 = standard coin, non-zero introduces chiral bias.
    boundary : str
        "periodic" (torus) or "reflective".
    """

    grid_size: int = 128
    coin_type: str = "hadamard"
    coin_angle: float = 0.0
    boundary: str = "periodic"

    # internal
    _coin: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.coin_type.lower() == "hadamard":
            base = hadamard_coin()
        elif self.coin_type.lower() == "grover":
            base = grover_coin()
        else:
            raise ValueError(f"Unknown coin_type: {self.coin_type}")

        if self.coin_angle != 0.0:
            # apply a phase rotation to break symmetry
            rot = np.diag(np.exp(1j * self.coin_angle * np.arange(4)))
            self._coin = rot @ base
        else:
            self._coin = base

    # ------------------------------------------------------------------
    def _apply_coin(self, psi: np.ndarray) -> np.ndarray:
        """Apply coin operator: psi shape (N, N, 4)."""
        # einsum: coin acts on last axis
        return np.einsum("cd,xyd->xyc", self._coin, psi)

    def _apply_shift(self, psi: np.ndarray) -> np.ndarray:
        """Conditional shift with periodic or reflective boundary."""
        N = self.grid_size
        shifted = np.zeros_like(psi)
        for d, (dx, dy) in enumerate(_DIRECTIONS):
            layer = psi[:, :, d]
            if self.boundary == "periodic":
                shifted[:, :, d] = np.roll(np.roll(layer, dx, axis=0), dy, axis=1)
            else:  # reflective
                # simple zero-padding reflection
                src_x = np.clip(np.arange(N) + dx, 0, N - 1)
                src_y = np.clip(np.arange(N) + dy, 0, N - 1)
                shifted[:, :, d] = layer[np.ix_(src_x, src_y)]
        return shifted

    # ------------------------------------------------------------------
    def initialize_state(self, psi_position: np.ndarray) -> np.ndarray:
        """Embed a position-space complex field into the 4-coin Hilbert space.

        The initial coin state is an equal superposition of all 4 directions,
        which maximises delocalisation speed.

        Parameters
        ----------
        psi_position : np.ndarray
            Complex field of shape (N, N) from the Injector.

        Returns
        -------
        np.ndarray
            State of shape (N, N, 4).
        """
        N = self.grid_size
        coin_init = np.ones(4, dtype=np.complex128) / 2.0  # uniform superposition
        psi = np.zeros((N, N, 4), dtype=np.complex128)
        for d in range(4):
            psi[:, :, d] = psi_position * coin_init[d]
        return psi

    # ------------------------------------------------------------------
    def step(self, psi: np.ndarray) -> np.ndarray:
        """Evolve one DTQW step: coin then shift."""
        psi = self._apply_coin(psi)
        psi = self._apply_shift(psi)
        return psi

    def evolve(self, psi: np.ndarray, n_steps: int) -> np.ndarray:
        """Evolve for n_steps, returning the final state."""
        for _ in range(n_steps):
            psi = self.step(psi)
        return psi

    def evolve_with_history(
        self, psi: np.ndarray, n_steps: int, record_every: int = 1
    ) -> list[np.ndarray]:
        """Evolve and record position-space probability density at intervals.

        Returns
        -------
        list[np.ndarray]
            Each entry is (N, N) real-valued probability density.
        """
        history = []
        for t in range(n_steps):
            psi = self.step(psi)
            if (t + 1) % record_every == 0:
                history.append(self.probability_density(psi))
        return history

    # ------------------------------------------------------------------
    @staticmethod
    def probability_density(psi: np.ndarray) -> np.ndarray:
        """Sum over coin degrees to get position-space |psi|^2."""
        return np.sum(np.abs(psi) ** 2, axis=-1)

    @staticmethod
    def position_field(psi: np.ndarray) -> np.ndarray:
        """Sum coin amplitudes coherently to recover a complex position field."""
        return np.sum(psi, axis=-1)
