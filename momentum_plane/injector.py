"""
injector.py — 动量注入器

Periodically injects coherent wave-packets onto a 2D spatial grid at fixed
stride intervals, all sharing the same momentum direction.  This is the
"source" term of the MomentumPlane engine: every injection event seeds a
Gaussian envelope modulated by a plane-wave phase factor.

Physics
-------
A single injected packet at lattice site (x0, y0) with wavevector (kx, ky)
and spatial width sigma takes the form:

    psi(x, y) = exp(-[(x-x0)^2 + (y-y0)^2] / (2 sigma^2))
                * exp(i (kx x + ky y))

Multiple packets are superposed coherently (amplitudes add, not intensities),
which is what produces the interference structure in momentum space.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Injector:
    """Periodic coherent wave-packet injector on a 2D grid.

    Parameters
    ----------
    grid_size : int
        Number of lattice sites along each dimension (N x N grid).
    stride : int
        Spatial interval (in lattice units) between adjacent injection sites.
    wavevector : tuple[float, float]
        Shared (kx, ky) momentum direction for every injected packet.
    sigma : float
        Gaussian envelope width (lattice units).
    amplitude : float
        Peak amplitude of each packet.
    phase_jitter : float
        Random phase noise per packet (0 = perfectly coherent).
    seed : Optional[int]
        RNG seed for reproducible phase jitter.
    """

    grid_size: int = 128
    stride: int = 16
    wavevector: tuple[float, float] = (0.8, 0.0)
    sigma: float = 3.0
    amplitude: float = 1.0
    phase_jitter: float = 0.0
    seed: Optional[int] = None

    # internal state
    _rng: np.random.Generator = field(init=False, repr=False)
    _x: np.ndarray = field(init=False, repr=False)
    _y: np.ndarray = field(init=False, repr=False)
    _injection_sites: list[tuple[int, int]] = field(
        init=False, repr=False, default_factory=list
    )

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        # centered coordinate grid
        half = self.grid_size // 2
        coords = np.arange(-half, half)
        self._x, self._y = np.meshgrid(coords, coords, indexing="ij")
        self._compute_injection_sites()

    # ------------------------------------------------------------------
    def _compute_injection_sites(self) -> None:
        """Place injection sites on a sub-lattice with the given stride."""
        sites = []
        half = self.grid_size // 2
        for i in range(-half, half, self.stride):
            for j in range(-half, half, self.stride):
                sites.append((i, j))
        self._injection_sites = sites

    # ------------------------------------------------------------------
    def inject(self, n_packets: Optional[int] = None) -> np.ndarray:
        """Generate the coherent superposition of injected wave-packets.

        Parameters
        ----------
        n_packets : Optional[int]
            If given, randomly select this many sites from the full sub-lattice.
            Otherwise inject at every site.

        Returns
        -------
        np.ndarray
            Complex field of shape (grid_size, grid_size) — the initial
            wavefunction in position space.
        """
        psi = np.zeros((self.grid_size, self.grid_size), dtype=np.complex128)
        kx, ky = self.wavevector

        sites = self._injection_sites
        if n_packets is not None and n_packets < len(sites):
            idx = self._rng.choice(len(sites), size=n_packets, replace=False)
            sites = [sites[i] for i in idx]

        for (x0, y0) in sites:
            phase = self._rng.normal(0, self.phase_jitter) if self.phase_jitter > 0 else 0.0
            envelope = np.exp(
                -((self._x - x0) ** 2 + (self._y - y0) ** 2) / (2 * self.sigma**2)
            )
            plane_wave = np.exp(1j * (kx * self._x + ky * self._y + phase))
            psi += self.amplitude * envelope * plane_wave

        return psi

    # ------------------------------------------------------------------
    @property
    def injection_sites(self) -> list[tuple[int, int]]:
        """Return the list of (x, y) lattice coordinates where packets are injected."""
        return list(self._injection_sites)

    @property
    def n_sites(self) -> int:
        return len(self._injection_sites)
