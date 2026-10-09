"""
inverse_design.py — 动量平面逆向设计

Given a target momentum-plane distribution, use gradient descent (JAX autodiff)
to find the injection parameters (wavevector, packet width, phase offset) that
produce the closest match.

This is the "inverse problem": instead of simulating forward from known parameters,
we specify the desired output and let the optimizer find the inputs.

Method:
  1. Define a differentiable forward pass: inject → DTQW → FFT → momentum intensity
  2. Loss = MSE(target, actual) or -correlation(target, actual)
  3. Adam optimizer with jax.grad over [kx, ky, sigma, phase_offset]

Usage:
    from momentum_plane.inverse_design import optimize, target_with_peaks
    target = target_with_peaks(64, [(20, 32), (44, 32)])  # 2 peaks
    result = optimize(target, n_iter=200, lr=0.02)
    print(result.params)       # optimized kx, ky, sigma, phase_offset
    print(result.loss_history) # convergence curve
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

# Enable 64-bit for numerical consistency with the NumPy backend.
jax.config.update("jax_enable_x64", True)


# ----------------------------------------------------------------------
# Differentiable injection
# ----------------------------------------------------------------------

def _injection_sites(grid_size: int, stride: int) -> np.ndarray:
    """Precompute injection site coordinates (fixed, not differentiable).

    Returns array of shape (M, 2) with (row, col) of each injection site.
    Sites form a centered regular sub-lattice with the given stride.
    """
    N = grid_size
    cy, cx = N // 2, N // 2
    n = N // (2 * stride)
    sites = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            py, px = cy + i * stride, cx + j * stride
            if 0 <= py < N and 0 <= px < N:
                sites.append((py, px))
    return np.array(sites, dtype=np.int32) if sites else np.zeros((0, 2), dtype=np.int32)


def differentiable_inject(
    kx: jnp.ndarray,
    ky: jnp.ndarray,
    sigma: jnp.ndarray,
    phase_offset: jnp.ndarray,
    grid_size: int,
    stride: int,
) -> jnp.ndarray:
    """Differentiable coherent wave-packet injection.

    All physical parameters (kx, ky, sigma, phase_offset) are JAX scalars
    and therefore differentiable.  The injection geometry (site positions)
    is fixed by stride and grid_size.

    Parameters
    ----------
    kx, ky : jnp.ndarray (scalar)
        Wavevector components (radians per lattice site).
    sigma : jnp.ndarray (scalar)
        Gaussian packet width (must be > 0).
    phase_offset : jnp.ndarray (scalar)
        Global phase offset (radians).
    grid_size : int
        Lattice dimension N.
    stride : int
        Injection spacing (fixed integer).

    Returns
    -------
    jnp.ndarray
        Complex field of shape (N, N), L2-normalized.
    """
    N = grid_size
    y, x = jnp.mgrid[0:N, 0:N]  # (N, N)

    sites = _injection_sites(N, stride)
    if len(sites) == 0:
        return jnp.zeros((N, N), dtype=jnp.complex128)

    site_y = jnp.asarray(sites[:, 0])  # (M,)
    site_x = jnp.asarray(sites[:, 1])  # (M,)

    # Gaussian envelopes for all sites: (N, N, M)
    dx = x[:, :, None] - site_x[None, None, :]
    dy = y[:, :, None] - site_y[None, None, :]
    gaussians = jnp.exp(-(dx**2 + dy**2) / (2.0 * sigma**2))

    # Plane-wave phase at each site: (M,)
    phases = jnp.exp(1j * (kx * site_x + ky * site_y + phase_offset))

    # Sum over sites: (N, N)
    field = jnp.sum(gaussians * phases[None, None, :], axis=2)

    # L2 normalize
    norm = jnp.sqrt(jnp.sum(jnp.abs(field) ** 2))
    field = field / jnp.where(norm > 0, norm, 1.0)

    return field


# ----------------------------------------------------------------------
# Differentiable DTQW (consistent with lattice_jax.py)
# ----------------------------------------------------------------------

def _hadamard_coin() -> jnp.ndarray:
    H2 = jnp.array([[1, 1], [1, -1]], dtype=jnp.complex128) / jnp.sqrt(2)
    return jnp.kron(H2, H2)


def _grover_coin() -> jnp.ndarray:
    N = 4
    return 2 * jnp.ones((N, N), dtype=jnp.complex128) / N - jnp.eye(N, dtype=jnp.complex128)


def _dtqw_step(psi: jnp.ndarray, coin: jnp.ndarray) -> jnp.ndarray:
    """One DTQW step: coin then conditional shift (periodic boundary).

    Identical logic to lattice_jax.LatticeHopJAX._step_impl.
    """
    psi = jnp.einsum("cd,xyd->xyc", coin, psi)
    psi = psi.at[:, :, 0].set(jnp.roll(psi[:, :, 0], -1, axis=0))
    psi = psi.at[:, :, 1].set(jnp.roll(psi[:, :, 1], 1, axis=1))
    psi = psi.at[:, :, 2].set(jnp.roll(psi[:, :, 2], 1, axis=0))
    psi = psi.at[:, :, 3].set(jnp.roll(psi[:, :, 3], -1, axis=1))
    return psi


def _evolve(psi: jnp.ndarray, coin: jnp.ndarray, n_steps: int) -> jnp.ndarray:
    """Evolve for n_steps using lax.scan (compiled loop)."""
    def body(carry, _):
        return _dtqw_step(carry, coin), None
    psi_final, _ = jax.lax.scan(body, psi, None, length=n_steps)
    return psi_final


# ----------------------------------------------------------------------
# Forward pass (fully differentiable)
# ----------------------------------------------------------------------

def forward_pass(
    params: jnp.ndarray,
    grid_size: int,
    stride: int,
    n_steps: int,
    coin_type: str = "hadamard",
) -> jnp.ndarray:
    """Differentiable forward pass: inject → DTQW → FFT → normalized momentum intensity.

    Parameters
    ----------
    params : jnp.ndarray of shape (4,)
        [kx, ky, sigma_raw, phase_offset].
        sigma = softplus(sigma_raw) + 0.5 ensures positivity and minimum width.
    grid_size, stride, n_steps : int
    coin_type : str
        "hadamard" or "grover".

    Returns
    -------
    jnp.ndarray
        Normalized momentum-plane intensity of shape (N, N), sums to 1.
    """
    kx, ky, sigma_raw, phase_offset = params[0], params[1], params[2], params[3]
    sigma = jax.nn.softplus(sigma_raw) + 0.5

    # 1. Differentiable injection
    psi0 = differentiable_inject(kx, ky, sigma, phase_offset, grid_size, stride)

    # 2. Embed into 4-coin Hilbert space (equal superposition)
    coin_init = jnp.ones(4, dtype=jnp.complex128) / 2.0
    psi = psi0[:, :, None] * coin_init[None, None, :]

    # 3. DTQW evolution
    coin = _hadamard_coin() if coin_type == "hadamard" else _grover_coin()
    psi = _evolve(psi, coin, n_steps)

    # 4. Position field (coherent sum over coins)
    pos_field = jnp.sum(psi, axis=-1)

    # 5. 2D FFT → momentum plane intensity
    momentum = jnp.fft.fftshift(jnp.fft.fft2(pos_field))
    intensity = jnp.abs(momentum) ** 2

    # Normalize to sum=1
    total = jnp.sum(intensity)
    intensity = intensity / jnp.where(total > 0, total, 1.0)

    return intensity


# ----------------------------------------------------------------------
# Loss functions
# ----------------------------------------------------------------------

def loss_mse(
    params: jnp.ndarray,
    target: jnp.ndarray,
    grid_size: int,
    stride: int,
    n_steps: int,
    coin_type: str,
) -> jnp.ndarray:
    """Mean squared error between actual and target momentum intensity."""
    actual = forward_pass(params, grid_size, stride, n_steps, coin_type)
    return jnp.mean((actual - target) ** 2)


def loss_correlation(
    params: jnp.ndarray,
    target: jnp.ndarray,
    grid_size: int,
    stride: int,
    n_steps: int,
    coin_type: str,
) -> jnp.ndarray:
    """Negative Pearson correlation (minimize = maximize correlation)."""
    actual = forward_pass(params, grid_size, stride, n_steps, coin_type)
    a = actual - jnp.mean(actual)
    t = target - jnp.mean(target)
    denom = jnp.sqrt(jnp.sum(a**2) * jnp.sum(t**2)) + 1e-10
    corr = jnp.sum(a * t) / denom
    return -corr


# ----------------------------------------------------------------------
# Adam optimizer
# ----------------------------------------------------------------------

@dataclass
class InverseDesignConfig:
    """Configuration for inverse design optimization."""

    grid_size: int = 64
    stride: int = 8
    n_steps: int = 20
    coin_type: str = "hadamard"
    n_iter: int = 200
    lr: float = 0.02
    loss_type: str = "mse"  # "mse" or "correlation"
    seed: int = 42
    # Adam hyperparameters
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


@dataclass
class InverseDesignResult:
    """Result of inverse design optimization."""

    params: dict  # optimized {kx, ky, sigma, phase_offset}
    loss_history: list[float]
    initial_momentum: np.ndarray
    final_momentum: np.ndarray
    target_momentum: np.ndarray
    config: InverseDesignConfig = field(default_factory=InverseDesignConfig)


def optimize(
    target: np.ndarray,
    config: InverseDesignConfig | None = None,
) -> InverseDesignResult:
    """Optimize injection parameters to match a target momentum distribution.

    Uses Adam optimizer with JAX autodiff.  The forward pass is fully
    differentiable, so gradients flow through injection → DTQW → FFT.

    Parameters
    ----------
    target : np.ndarray
        Target momentum-plane intensity of shape (N, N), will be normalized.
    config : InverseDesignConfig, optional
        Optimization configuration.

    Returns
    -------
    InverseDesignResult
    """
    if config is None:
        config = InverseDesignConfig()

    N = config.grid_size
    target_jax = jnp.asarray(target, dtype=jnp.float64)
    target_jax = target_jax / jnp.sum(target_jax)

    # Build closures that capture static hyperparameters.
    # The jit-compiled functions only take dynamic args (params, target).
    def _forward(params):
        return forward_pass(params, N, config.stride, config.n_steps, config.coin_type)

    def _loss(params, tgt):
        actual = _forward(params)
        if config.loss_type == "mse":
            return jnp.mean((actual - tgt) ** 2)
        else:
            a = actual - jnp.mean(actual)
            t = tgt - jnp.mean(tgt)
            denom = jnp.sqrt(jnp.sum(a**2) * jnp.sum(t**2)) + 1e-10
            return -jnp.sum(a * t) / denom

    # JIT-compiled gradient and loss (only dynamic args are traced)
    grad_fn = jax.jit(jax.grad(_loss))
    loss_fn_jit = jax.jit(_loss)

    # Initialize parameters randomly
    key = jax.random.PRNGKey(config.seed)
    kx0 = jax.random.uniform(key, (), minval=-1.0, maxval=1.0)
    ky0 = jax.random.uniform(key, (), minval=-1.0, maxval=1.0)
    sigma_raw0 = jnp.array(0.5)  # softplus(0.5)+0.5 ≈ 1.46
    phase0 = jax.random.uniform(key, (), minval=0.0, maxval=2 * jnp.pi)
    params = jnp.array([kx0, ky0, sigma_raw0, phase0], dtype=jnp.float64)

    # Record initial momentum
    initial_momentum = np.asarray(
        forward_pass(params, N, config.stride, config.n_steps, config.coin_type)
    )

    # Adam state
    m = jnp.zeros_like(params)
    v = jnp.zeros_like(params)
    loss_history = []

    # Optimization loop
    for t in range(1, config.n_iter + 1):
        grads = grad_fn(params, target_jax)

        # Adam update
        m = config.beta1 * m + (1 - config.beta1) * grads
        v = config.beta2 * v + (1 - config.beta2) * grads**2
        m_hat = m / (1 - config.beta1**t)
        v_hat = v / (1 - config.beta2**t)
        params = params - config.lr * m_hat / (jnp.sqrt(v_hat) + config.eps)

        # Record loss every 10 iterations
        if t % 10 == 0 or t == 1:
            loss_val = float(loss_fn_jit(params, target_jax))
            loss_history.append(loss_val)

    # Final momentum
    final_momentum = np.asarray(
        forward_pass(params, N, config.stride, config.n_steps, config.coin_type)
    )

    # Convert params to dict with physical values
    kx_opt = float(params[0])
    ky_opt = float(params[1])
    sigma_opt = float(jax.nn.softplus(params[2]) + 0.5)
    phase_opt = float(params[3])

    return InverseDesignResult(
        params={
            "kx": kx_opt,
            "ky": ky_opt,
            "sigma": sigma_opt,
            "phase_offset": phase_opt,
        },
        loss_history=loss_history,
        initial_momentum=initial_momentum,
        final_momentum=final_momentum,
        target_momentum=np.asarray(target_jax),
        config=config,
    )


# ----------------------------------------------------------------------
# Target generation helpers
# ----------------------------------------------------------------------

def target_with_peaks(
    grid_size: int,
    peak_positions: list[tuple[int, int]],
    peak_sigma: float = 3.0,
) -> np.ndarray:
    """Create a target momentum distribution with Gaussian peaks at specified positions.

    Parameters
    ----------
    grid_size : int
        N.
    peak_positions : list of (row, col)
        Positions of peaks in the (N, N) momentum plane.
    peak_sigma : float
        Width of each Gaussian peak.

    Returns
    -------
    np.ndarray
        Target intensity of shape (N, N), normalized to sum=1.
    """
    N = grid_size
    y, x = np.mgrid[0:N, 0:N]
    target = np.zeros((N, N), dtype=np.float64)
    for py, px in peak_positions:
        target += np.exp(-((x - px) ** 2 + (y - py) ** 2) / (2 * peak_sigma**2))
    target = target / target.sum()
    return target


def target_ring(
    grid_size: int,
    radius: float = 15.0,
    thickness: float = 3.0,
) -> np.ndarray:
    """Create a ring-shaped target momentum distribution.

    Parameters
    ----------
    grid_size : int
    radius : float
        Ring radius from center.
    thickness : float
        Ring thickness (Gaussian falloff).

    Returns
    -------
    np.ndarray
        Target intensity of shape (N, N), normalized.
    """
    N = grid_size
    cy, cx = N // 2, N // 2
    y, x = np.mgrid[0:N, 0:N]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    target = np.exp(-((r - radius) ** 2) / (2 * thickness**2))
    target = target / target.sum()
    return target
