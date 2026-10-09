"""
visualizer.py — 动量平面输出与可视化

Renders position-space probability density, momentum-plane intensity heatmaps,
phase-space trajectories, and animated GIFs of the full injection → hopping →
convergence pipeline.

This module is deliberately backend-agnostic: it produces matplotlib figures
that can be saved to disk, embedded in notebooks, or served by Streamlit.
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")  # non-interactive backend
from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm

# A physically-inspired colormap: deep navy -> cyan -> magenta -> gold
_MOMENTUM_CMAP = "inferno"


@dataclass
class Visualizer:
    """Render MomentumPlane simulation outputs.

    Parameters
    ----------
    dpi : int
        Figure resolution.
    figsize : tuple[float, float]
        Figure size in inches.
    cmap : str
        Matplotlib colormap for intensity heatmaps.
    use_log : bool
        If True, render intensity on a log scale (better for sharp peaks).
    """

    dpi: int = 120
    figsize: tuple[float, float] = (6, 5)
    cmap: str = _MOMENTUM_CMAP
    use_log: bool = True

    # ------------------------------------------------------------------
    def heatmap(
        self,
        data: np.ndarray,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        save_path: str | None = None,
    ) -> plt.Figure:
        """Render a 2D intensity heatmap.

        Parameters
        ----------
        data : np.ndarray
            2D real array (intensity or probability density).
        save_path : Optional[str]
            If given, save the figure to this path.

        Returns
        -------
        plt.Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        plot_data = data + 1e-12 if self.use_log else data
        norm = LogNorm(vmin=plot_data.min(), vmax=plot_data.max()) if self.use_log else None
        im = ax.imshow(plot_data, cmap=self.cmap, norm=norm, origin="lower")
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    def side_by_side(
        self,
        position_data: np.ndarray,
        momentum_data: np.ndarray,
        step_label: str = "",
        save_path: str | None = None,
    ) -> plt.Figure:
        """Render position-space density and momentum-plane intensity side by side.

        This is the canonical MomentumPlane visual: left panel shows where the
        wavefunction is in real space, right panel shows the diffraction /
        convergence pattern in momentum space.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)

        # position space
        pos = position_data + 1e-12
        axes[0].imshow(pos, cmap="viridis", norm=LogNorm(vmin=pos.min(), vmax=pos.max()), origin="lower")
        axes[0].set_title(f"Position Space |ψ(x,y)|²  {step_label}", fontsize=11)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")

        # momentum space
        mom = momentum_data + 1e-12
        im = axes[1].imshow(mom, cmap=self.cmap, norm=LogNorm(vmin=mom.min(), vmax=mom.max()), origin="lower")
        axes[1].set_title(f"Momentum Plane |ψ̃(kx,ky)|²  {step_label}", fontsize=11)
        axes[1].set_xlabel("kx")
        axes[1].set_ylabel("ky")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    def phase_portrait(
        self,
        phase_data: np.ndarray,
        title: str = "Momentum-Space Phase",
        save_path: str | None = None,
    ) -> plt.Figure:
        """Render a phase map with a cyclic colormap."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        im = ax.imshow(phase_data, cmap="twilight", vmin=-np.pi, vmax=np.pi, origin="lower")
        ax.set_title(title, fontsize=13)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="phase (rad)")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    def animate_evolution(
        self,
        position_history: Sequence[np.ndarray],
        momentum_history: Sequence[np.ndarray],
        save_path: str,
        fps: int = 10,
        title_prefix: str = "Step",
    ) -> None:
        """Animate the side-by-side position/momentum evolution as a GIF.

        Parameters
        ----------
        position_history : Sequence[np.ndarray]
            List of position-space density arrays, one per recorded step.
        momentum_history : Sequence[np.ndarray]
            List of momentum-plane intensity arrays, same length.
        save_path : str
            Output .gif path.
        fps : int
            Frames per second.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)

        pos0 = position_history[0] + 1e-12
        mom0 = momentum_history[0] + 1e-12
        im_pos = axes[0].imshow(
            pos0, cmap="viridis",
            norm=LogNorm(vmin=pos0.min(), vmax=pos0.max()), origin="lower"
        )
        im_mom = axes[1].imshow(
            mom0, cmap=self.cmap,
            norm=LogNorm(vmin=mom0.min(), vmax=mom0.max()), origin="lower"
        )
        axes[0].set_title("Position Space", fontsize=11)
        axes[1].set_title("Momentum Plane", fontsize=11)
        for ax in axes:
            ax.set_xlabel("x" if ax is axes[0] else "kx")
            ax.set_ylabel("y" if ax is axes[0] else "ky")
        plt.colorbar(im_mom, ax=axes[1], fraction=0.046, pad=0.04)
        fig.suptitle(f"{title_prefix} 0", fontsize=13)

        def update(frame: int):
            pos = position_history[frame] + 1e-12
            mom = momentum_history[frame] + 1e-12
            im_pos.set_data(pos)
            im_pos.set_norm(LogNorm(vmin=pos.min(), vmax=pos.max()))
            im_mom.set_data(mom)
            im_mom.set_norm(LogNorm(vmin=mom.min(), vmax=mom.max()))
            fig.suptitle(f"{title_prefix} {frame}", fontsize=13)
            return im_pos, im_mom

        anim = FuncAnimation(fig, update, frames=len(position_history), interval=1000 // fps, blit=False)
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
        plt.close(fig)

    # ------------------------------------------------------------------
    def convergence_plot(
        self,
        peak_intensities: Sequence[float],
        title: str = "Momentum Peak Convergence",
        save_path: str | None = None,
    ) -> plt.Figure:
        """Plot how the dominant momentum-peak intensity evolves over steps."""
        fig, ax = plt.subplots(figsize=(7, 4), dpi=self.dpi)
        ax.plot(peak_intensities, "-o", color="#e25822", markersize=4, linewidth=1.5)
        ax.set_xlabel("Evolution Step")
        ax.set_ylabel("Peak Intensity (raw)")
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig
