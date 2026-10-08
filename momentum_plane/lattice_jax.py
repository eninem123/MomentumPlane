"""
lattice_jax.py — JAX 加速的量子跳跃后端

A JAX-accelerated implementation of the discrete-time quantum walk (DTQW).
Drop-in replacement for lattice.LatticeHop with:
  - jit compilation for blazing-fast evolution loops
  - GPU acceleration (just install jax[cuda])
  - vmap for batched parameter sweeps
  - autodiff (grad/value_and_grad) for physics parameter optimization

Usage:
    from momentum_plane.lattice_jax import LatticeHopJAX
    lat = LatticeHopJAX(grid_size=128, coin_type="hadamard")
    psi = lat.initialize_state(psi0)
    psi_final = lat.evolve(psi, n_steps=100)  # jit-compiled, fast

The first call compiles; subsequent calls with the same shapes run at
native speed (CPU or GPU).
"""

from __future__ import annotations

import jax

# Enable 64-bit precision by default so JAX results match NumPy complex128.
# Users who want float32 speed can set JAX_ENABLE_X64=0 before import.
jax.config.update("jax_enable_x64", True)

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

# 4-direction basis: 0=up, 1=right, 2=down, 3=left
_DIRECTIONS = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.int32)


def hadamard_coin_jax() -> jnp.ndarray:
    """4x4 balanced Hadamard coin as a JAX array."""
    H2 = jnp.array([[1, 1], [1, -1]], dtype=jnp.complex128) / jnp.sqrt(2)
    return jnp.kron(H2, H2)


def grover_coin_jax() -> jnp.ndarray:
    """4x4 Grover diffusion coin as a JAX array."""
    N = 4
    return 2 * jnp.ones((N, N), dtype=jnp.complex128) / N - jnp.eye(N, dtype=jnp.complex128)


@dataclass
class LatticeHopJAX:
    """JAX-accelerated discrete-time quantum walk on a 2D lattice.

    Parameters
    ----------
    grid_size : int
        Lattice dimension (N x N).
    coin_type : str
        "hadamard" or "grover".
    coin_angle : float
        Chiral phase bias (radians).
    boundary : str
        Only "periodic" is supported in the JAX backend (roll is native).
    """

    grid_size: int = 128
    coin_type: str = "hadamard"
    coin_angle: float = 0.0
    boundary: str = "periodic"

    # internal (not dataclass fields)
    _coin: jnp.ndarray = None
    _step_fn = None

    def __post_init__(self) -> None:
        if self.boundary != "periodic":
            raise ValueError(
                f"JAX backend only supports periodic boundary (got '{self.boundary}'). "
                "Use LatticeHop for reflective boundaries."
            )

        if self.coin_type.lower() == "hadamard":
            base = hadamard_coin_jax()
        elif self.coin_type.lower() == "grover":
            base = grover_coin_jax()
        else:
            raise ValueError(f"Unknown coin_type: {self.coin_type}")

        if self.coin_angle != 0.0:
            rot = jnp.diag(jnp.exp(1j * self.coin_angle * jnp.arange(4)))
            self._coin = rot @ base
        else:
            self._coin = base

        # Build the jit-compiled step function at init time
        self._step_fn = jax.jit(self._step_impl)

    # ------------------------------------------------------------------
    @staticmethod
    def _step_impl(psi: jnp.ndarray, coin: jnp.ndarray) -> jnp.ndarray:
        """One DTQW step: coin then conditional shift (periodic).

        This is a pure function so it can be jit-compiled.
        """
        # Coin: einsum over the coin dimension
        psi = jnp.einsum("cd,xyd->xyc", coin, psi)

        # Conditional shift: roll each direction layer
        # direction 0: up (-1, 0)
        psi = psi.at[:, :, 0].set(jnp.roll(jnp.roll(psi[:, :, 0], -1, axis=0), 0, axis=1))
        # direction 1: right (0, +1)
        psi = psi.at[:, :, 1].set(jnp.roll(jnp.roll(psi[:, :, 1], 0, axis=0), 1, axis=1))
        # direction 2: down (+1, 0)
        psi = psi.at[:, :, 2].set(jnp.roll(jnp.roll(psi[:, :, 2], 1, axis=0), 0, axis=1))
        # direction 3: left (0, -1)
        psi = psi.at[:, :, 3].set(jnp.roll(jnp.roll(psi[:, :, 3], 0, axis=0), -1, axis=1))

        return psi

    # ------------------------------------------------------------------
    def initialize_state(self, psi_position: np.ndarray) -> jnp.ndarray:
        """Embed a position-space complex field into the 4-coin Hilbert space.

        Parameters
        ----------
        psi_position : np.ndarray
            Complex field of shape (N, N) from the Injector (NumPy array).

        Returns
        -------
        jnp.ndarray
            State of shape (N, N, 4) on the JAX device.
        """
        psi_jax = jnp.asarray(psi_position)
        coin_init = jnp.ones(4, dtype=jnp.complex128) / 2.0
        # Broadcast: (N, N, 1) * (1, 1, 4) -> (N, N, 4)
        return psi_jax[:, :, None] * coin_init[None, None, :]

    # ------------------------------------------------------------------
    def step(self, psi: jnp.ndarray) -> jnp.ndarray:
        """Evolve one DTQW step (jit-compiled)."""
        return self._step_fn(psi, self._coin)

    def evolve(self, psi: jnp.ndarray, n_steps: int) -> jnp.ndarray:
        """Evolve for n_steps using a jit-compiled loop (lax.scan).

        Uses jax.lax.scan for a compiled loop — much faster than a Python
        for-loop, especially on GPU.
        """
        coin = self._coin
        step_fn = self._step_fn

        def body(carry, _):
            return step_fn(carry, coin), None

        psi_final, _ = jax.lax.scan(body, psi, None, length=n_steps)
        return psi_final

    def evolve_with_history(
        self, psi: jnp.ndarray, n_steps: int, record_every: int = 1
    ) -> list[np.ndarray]:
        """Evolve and record position-space probability density at intervals.

        Returns NumPy arrays (moved off device) for compatibility with
        the Visualizer.
        """
        history = []
        psi_current = psi
        for t in range(n_steps):
            psi_current = self.step(psi_current)
            if (t + 1) % record_every == 0:
                history.append(np.asarray(self.probability_density(psi_current)))
        return history

    # ------------------------------------------------------------------
    @staticmethod
    def probability_density(psi: jnp.ndarray) -> jnp.ndarray:
        """Sum over coin degrees to get position-space |psi|^2."""
        return jnp.sum(jnp.abs(psi) ** 2, axis=-1)

    @staticmethod
    def position_field(psi: jnp.ndarray) -> jnp.ndarray:
        """Sum coin amplitudes coherently to recover a complex position field."""
        return jnp.sum(psi, axis=-1)

    # ------------------------------------------------------------------
    def to_numpy(self, arr: jnp.ndarray) -> np.ndarray:
        """Move a JAX array to host NumPy."""
        return np.asarray(arr)


# ----------------------------------------------------------------------
# Batch simulation utilities (vmap)
# ----------------------------------------------------------------------

def batch_evolve(
    injector_params: list[dict],
    grid_size: int = 64,
    n_steps: int = 30,
    coin_type: str = "hadamard",
) -> np.ndarray:
    """Batch-evolve multiple injection configurations using vmap.

    This is useful for parameter sweeps: simulate many different
    wavevectors / strides / sigmas in parallel on GPU.

    Parameters
    ----------
    injector_params : list[dict]
        Each dict has keys: wavevector (tuple), stride (int), sigma (float).
    grid_size : int
    n_steps : int
    coin_type : str

    Returns
    -------
    np.ndarray
        Final momentum-plane intensities of shape (B, N, N).
    """
    from .injector import Injector
    from .synthesizer import FieldPlane

    B = len(injector_params)
    N = grid_size

    # Build initial states on host first (Injector is NumPy)
    init_states = np.zeros((B, N, N), dtype=np.complex128)
    for i, params in enumerate(injector_params):
        inj = Injector(
            grid_size=grid_size,
            stride=params.get("stride", 8),
            wavevector=params.get("wavevector", (0.8, 0.0)),
            sigma=params.get("sigma", 2.5),
            seed=42,
        )
        init_states[i] = inj.inject()

    # JAX coin
    coin = hadamard_coin_jax() if coin_type == "hadamard" else grover_coin_jax()

    # Single-step function that works on batched input (B, N, N, 4)
    def batch_step(psi_batch, coin):
        # coin: (4,4), psi_batch: (B, N, N, 4)
        psi_batch = jnp.einsum("cd,bxyd->bxyc", coin, psi_batch)
        # shifts for each direction
        for d, (dx, dy) in enumerate([(-1, 0), (0, 1), (1, 0), (0, -1)]):
            layer = psi_batch[:, :, :, d]
            shifted = jnp.roll(jnp.roll(layer, dx, axis=1), dy, axis=2)
            psi_batch = psi_batch.at[:, :, :, d].set(shifted)
        return psi_batch

    batch_step_jit = jax.jit(batch_step)

    # Initialize batch state: (B, N, N, 4)
    psi_batch = jnp.asarray(init_states)[:, :, :, None] * (
        jnp.ones(4, dtype=jnp.complex128) / 2.0
    )[None, None, None, :]

    # Evolve
    for _ in range(n_steps):
        psi_batch = batch_step_jit(psi_batch, coin)

    # Compute momentum-plane intensity for each batch element
    field = FieldPlane(grid_size=grid_size)
    pos_fields = np.asarray(jnp.sum(psi_batch, axis=-1))  # (B, N, N)
    intensities = np.zeros((B, N, N))
    for b in range(B):
        intensities[b] = field.intensity(pos_fields[b])

    return intensities
