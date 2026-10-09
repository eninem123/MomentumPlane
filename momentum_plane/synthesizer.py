"""
synthesizer.py — 动量平面合成器

Transforms the position-space evolved wavefunction into momentum space via
2D FFT, producing the "FieldPlane": the interference pattern that emerges
when coherently injected packets undergo discrete quantum hopping.

This is where the physics becomes visually striking: a set of well-separated
injection sites in position space produces a *comb* of sharp peaks in
momentum space, and the quantum walk evolution modulates the relative phases
so that peaks converge, split, or form caustic-like structures.

Physics
-------
The momentum-space wavefunction is:

    psi_tilde(kx, ky) = FFT2[ psi(x, y) ]

The momentum-plane intensity is |psi_tilde|^2.  For N injection sites at
positions {r_j} with shared wavevector k0, the momentum amplitude is:

    psi_tilde(k) ~ sum_j exp(i (k - k0) . r_j) * envelope(k - k0)

which is a discrete lattice diffraction pattern — the "momentum plane"
that gives this project its name.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FieldPlane:
    """Momentum-plane field synthesiser.

    Parameters
    ----------
    grid_size : int
        FFT grid dimension (must match the lattice size).
    apply_shift : bool
        If True, use fftshift so zero momentum is at the array centre.
    window : Optional[str]
        Apodisation window applied before FFT to reduce ringing:
        None, "hann", "hamming", or "blackman".
    """

    grid_size: int = 128
    apply_shift: bool = True
    window: str | None = None

    _window_array: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.window is not None:
            self._window_array = self._make_window()

    def _make_window(self) -> np.ndarray:
        N = self.grid_size
        if self.window == "hann":
            w1d = np.hanning(N)
        elif self.window == "hamming":
            w1d = np.hamming(N)
        elif self.window == "blackman":
            w1d = np.blackman(N)
        else:
            raise ValueError(f"Unknown window: {self.window}")
        return np.outer(w1d, w1d)

    # ------------------------------------------------------------------
    def transform(self, psi_position: np.ndarray) -> np.ndarray:
        """Compute the 2D FFT of a position-space complex field.

        Parameters
        ----------
        psi_position : np.ndarray
            Complex field of shape (N, N).

        Returns
        -------
        np.ndarray
            Complex momentum-space field of shape (N, N).
        """
        field = psi_position
        if self._window_array is not None:
            field = field * self._window_array
        psi_k = np.fft.fft2(field)
        if self.apply_shift:
            psi_k = np.fft.fftshift(psi_k)
        return psi_k

    # ------------------------------------------------------------------
    def intensity(self, psi_position: np.ndarray) -> np.ndarray:
        """Momentum-plane intensity |psi_tilde(k)|^2, normalised to [0, 1]."""
        psi_k = self.transform(psi_position)
        inten = np.abs(psi_k) ** 2
        mx = inten.max()
        if mx > 0:
            inten /= mx
        return inten

    # ------------------------------------------------------------------
    def raw_intensity(self, psi_position: np.ndarray) -> np.ndarray:
        """Momentum-plane intensity |psi_tilde(k)|^2, NOT normalised.

        Use this for convergence analysis where the absolute peak height
        carries meaning (e.g. tracking whether peaks sharpen or broaden).
        """
        psi_k = self.transform(psi_position)
        return np.abs(psi_k) ** 2

    # ------------------------------------------------------------------
    def phase(self, psi_position: np.ndarray) -> np.ndarray:
        """Momentum-space phase map in [-pi, pi]."""
        psi_k = self.transform(psi_position)
        return np.angle(psi_k)

    # ------------------------------------------------------------------
    def momentum_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (kx, ky) meshgrid in angular frequency units (fftshifted)."""
        N = self.grid_size
        k = 2 * np.pi * np.fft.fftfreq(N)
        if self.apply_shift:
            k = np.fft.fftshift(k)
        return np.meshgrid(k, k, indexing="ij")

    # ------------------------------------------------------------------
    def peak_positions(
        self, psi_position: np.ndarray, threshold: float = 0.3, min_distance: int = 4
    ) -> list[tuple[int, int]]:
        """Find local maxima in the momentum-plane intensity.

        Simple non-maximum suppression: a pixel is a peak if it is above
        *threshold* (relative to max) and has no higher pixel within
        *min_distance* Chebyshev distance.

        Returns
        -------
        list[tuple[int, int]]
            (row, col) pixel coordinates of detected peaks.
        """
        inten = self.intensity(psi_position)
        peaks = []
        # iterate over candidates above threshold
        candidates = np.argwhere(inten > threshold)
        # sort by intensity descending
        candidates = sorted(candidates, key=lambda p: inten[p[0], p[1]], reverse=True)
        for (r, c) in candidates:
            # check distance to existing peaks
            too_close = any(
                max(abs(r - pr), abs(c - pc)) < min_distance for (pr, pc) in peaks
            )
            if not too_close:
                peaks.append((int(r), int(c)))
        return peaks
