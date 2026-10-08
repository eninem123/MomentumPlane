"""
boundary.py — 几何边界掩码与拓扑约束

Geometric boundary masks for confining the quantum walk to specific shapes.
Inspired by topological photonics: waves confined to lattice boundaries
propagate unidirectionally and defect-immune.

Supported shapes:
  - circle    : solid disk (absorbing boundary at radius R)
  - ring      : annular region (R_inner < r < R_outer)
  - polygon   : regular N-gon (3=triangle, 4=square, 6=hexagon...)
  - strip     : horizontal/vertical waveguide
  - custom    : user-provided boolean mask

Usage:
    from momentum_plane.boundary import BoundaryMask
    mask = BoundaryMask(grid_size=128, shape="ring", inner_radius=20, outer_radius=40)
    # Apply after each lattice step:
    psi = mask.apply(psi)  # zeros out amplitudes outside the boundary
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BoundaryMask:
    """Geometric boundary mask for confining quantum walk states.

    Parameters
    ----------
    grid_size : int
        Lattice dimension (N x N).
    shape : str
        One of "circle", "ring", "polygon", "strip", "custom".
    radius : float
        Radius for circle (in lattice units). Default: grid_size // 3.
    inner_radius : float
        Inner radius for ring.
    outer_radius : float
        Outer radius for ring.
    n_sides : int
        Number of sides for polygon (3=triangle, 4=square, 6=hexagon).
    rotation : float
        Rotation angle for polygon (radians).
    strip_axis : str
        "horizontal" or "vertical" for strip shape.
    strip_width : int
        Width of strip in lattice units.
    custom_mask : np.ndarray, optional
        User-provided boolean mask of shape (N, N). True = inside boundary.
    boundary_type : str
        "absorbing" (zero outside) or "reflective" (not yet implemented).
    """

    grid_size: int = 128
    shape: str = "circle"
    radius: float | None = None
    inner_radius: float = 20.0
    outer_radius: float = 40.0
    n_sides: int = 6
    rotation: float = 0.0
    strip_axis: str = "horizontal"
    strip_width: int = 20
    custom_mask: np.ndarray | None = None
    boundary_type: str = "absorbing"

    _mask: np.ndarray = None

    def __post_init__(self) -> None:
        if self.radius is None:
            self.radius = self.grid_size // 3
        self._mask = self._build_mask()

    # ------------------------------------------------------------------
    def _build_mask(self) -> np.ndarray:
        """Build the boolean mask based on shape parameter."""
        N = self.grid_size
        cy, cx = N // 2, N // 2
        y, x = np.ogrid[:N, :N]
        dy, dx = y - cy, x - cx
        r = np.sqrt(dx**2 + dy**2)

        if self.shape == "circle":
            return r <= self.radius

        elif self.shape == "ring":
            return (r >= self.inner_radius) & (r <= self.outer_radius)

        elif self.shape == "polygon":
            return self._polygon_mask(dx, dy)

        elif self.shape == "strip":
            if self.strip_axis == "horizontal":
                return np.broadcast_to(np.abs(dy) <= self.strip_width // 2, (N, N)).copy()
            else:
                return np.broadcast_to(np.abs(dx) <= self.strip_width // 2, (N, N)).copy()

        elif self.shape == "custom":
            if self.custom_mask is None:
                raise ValueError("custom_mask must be provided when shape='custom'")
            if self.custom_mask.shape != (N, N):
                raise ValueError(
                    f"custom_mask shape {self.custom_mask.shape} != ({N}, {N})"
                )
            return self.custom_mask.astype(bool)

        else:
            raise ValueError(
                f"Unknown shape: {self.shape}. "
                "Use 'circle', 'ring', 'polygon', 'strip', or 'custom'."
            )

    def _polygon_mask(self, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        """Build a regular n-gon mask using the distance-to-edges method.

        A regular n-gon can be defined as the intersection of n half-planes.
        For each edge, compute the signed distance; inside if all distances >= 0.
        """
        n = self.n_sides
        rot = self.rotation
        R = self.radius  # circumradius

        # For each vertex angle, compute the edge normal
        full_shape = np.broadcast_shapes(dx.shape, dy.shape)
        mask = np.ones(full_shape, dtype=bool)
        for i in range(n):
            # Edge midpoint angle
            theta = 2 * np.pi * i / n + rot + np.pi / n
            # Edge normal direction (pointing outward)
            nx, ny = np.cos(theta), np.sin(theta)
            # Signed distance from edge line (positive = inside)
            # Edge line: nx*x + ny*y = R*cos(pi/n)
            dist = R * np.cos(np.pi / n) - (nx * dx + ny * dy)
            mask &= dist >= 0

        return mask

    # ------------------------------------------------------------------
    @property
    def mask(self) -> np.ndarray:
        """Return the boolean mask (True = inside boundary)."""
        return self._mask

    def apply(self, psi: np.ndarray) -> np.ndarray:
        """Apply the boundary mask to a state array.

        For absorbing boundaries: zeros out amplitudes outside the mask.
        Works with both (N, N) position fields and (N, N, 4) coin states.

        Parameters
        ----------
        psi : np.ndarray
            State of shape (N, N) or (N, N, 4).

        Returns
        -------
        np.ndarray
            Masked state (same shape, dtype preserved).
        """
        if psi.ndim == 2:
            return psi * self._mask
        elif psi.ndim == 3:
            # Broadcast mask over coin dimension: (N, N, 1) * (N, N, 4)
            return psi * self._mask[:, :, None]
        else:
            raise ValueError(f"Expected 2D or 3D array, got {psi.ndim}D")

    def coverage(self) -> float:
        """Return the fraction of lattice sites inside the boundary."""
        return float(self._mask.sum()) / self._mask.size

    def boundary_sites(self) -> np.ndarray:
        """Return coordinates of sites on the boundary edge (inside, adjacent to outside)."""
        m = self._mask
        # A boundary site is inside but has at least one 4-neighbor outside
        from scipy import ndimage

        eroded = ndimage.binary_erosion(m)
        return np.argwhere(m & ~eroded)
