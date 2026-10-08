"""
synthesizer.py — momentum plane field synthesizer via 2D FFT
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class FieldPlane:
    grid_size: int = 128
    apply_shift: bool = True
    window: Optional[str] = None
    _window_array: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.window is not None:
            self._window_array = self._make_window()

    def _make_window(self):
        N = self.grid_size
        if self.window == "hann": w1d = np.hanning(N)
        elif self.window == "hamming": w1d = np.hamming(N)
        elif self.window == "blackman": w1d = np.blackman(N)
        else: raise ValueError(f"Unknown window: {self.window}")
        return np.outer(w1d, w1d)

    def transform(self, psi_position):
        field = psi_position
        if self._window_array is not None:
            field = field * self._window_array
        psi_k = np.fft.fft2(field)
        if self.apply_shift:
            psi_k = np.fft.fftshift(psi_k)
        return psi_k

    def intensity(self, psi_position):
        psi_k = self.transform(psi_position)
        inten = np.abs(psi_k) ** 2
        mx = inten.max()
        if mx > 0: inten /= mx
        return inten

    def phase(self, psi_position):
        return np.angle(self.transform(psi_position))

    def momentum_coordinates(self):
        N = self.grid_size
        k = 2 * np.pi * np.fft.fftfreq(N)
        if self.apply_shift: k = np.fft.fftshift(k)
        return np.meshgrid(k, k, indexing="ij")

    def peak_positions(self, psi_position, threshold=0.3, min_distance=4):
        inten = self.intensity(psi_position)
        peaks = []
        candidates = np.argwhere(inten > threshold)
        candidates = sorted(candidates, key=lambda p: inten[p[0], p[1]], reverse=True)
        for (r, c) in candidates:
            too_close = any(max(abs(r-pr), abs(c-pc)) < min_distance for (pr, pc) in peaks)
            if not too_close:
                peaks.append((int(r), int(c)))
        return peaks
