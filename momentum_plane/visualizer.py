"""
visualizer.py — heatmaps, phase portraits, animated GIFs
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation, PillowWriter
from dataclasses import dataclass
from typing import Optional, Sequence

_MOMENTUM_CMAP = "inferno"

@dataclass
class Visualizer:
    dpi: int = 120
    figsize: tuple = (6, 5)
    cmap: str = _MOMENTUM_CMAP
    use_log: bool = True

    def heatmap(self, data, title="", xlabel="", ylabel="", save_path=None):
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        plot_data = data + 1e-12 if self.use_log else data
        norm = LogNorm(vmin=plot_data.min(), vmax=plot_data.max()) if self.use_log else None
        im = ax.imshow(plot_data, cmap=self.cmap, norm=norm, origin="lower")
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        if save_path: fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    def side_by_side(self, position_data, momentum_data, step_label="", save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)
        pos = position_data + 1e-12
        axes[0].imshow(pos, cmap="viridis", norm=LogNorm(vmin=pos.min(), vmax=pos.max()), origin="lower")
        axes[0].set_title(f"Position Space {step_label}", fontsize=11)
        axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
        mom = momentum_data + 1e-12
        im = axes[1].imshow(mom, cmap=self.cmap, norm=LogNorm(vmin=mom.min(), vmax=mom.max()), origin="lower")
        axes[1].set_title(f"Momentum Plane {step_label}", fontsize=11)
        axes[1].set_xlabel("kx"); axes[1].set_ylabel("ky")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        if save_path: fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    def phase_portrait(self, phase_data, title="Momentum-Space Phase", save_path=None):
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        im = ax.imshow(phase_data, cmap="twilight", vmin=-np.pi, vmax=np.pi, origin="lower")
        ax.set_title(title, fontsize=13)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="phase (rad)")
        plt.tight_layout()
        if save_path: fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    def animate_evolution(self, position_history, momentum_history, save_path, fps=10, title_prefix="Step"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)
        pos0 = position_history[0] + 1e-12
        mom0 = momentum_history[0] + 1e-12
        im_pos = axes[0].imshow(pos0, cmap="viridis", norm=LogNorm(vmin=pos0.min(), vmax=pos0.max()), origin="lower")
        im_mom = axes[1].imshow(mom0, cmap=self.cmap, norm=LogNorm(vmin=mom0.min(), vmax=mom0.max()), origin="lower")
        axes[0].set_title("Position Space", fontsize=11)
        axes[1].set_title("Momentum Plane", fontsize=11)
        for ax in axes:
            ax.set_xlabel("x" if ax is axes[0] else "kx")
            ax.set_ylabel("y" if ax is axes[0] else "ky")
        plt.colorbar(im_mom, ax=axes[1], fraction=0.046, pad=0.04)
        fig.suptitle(f"{title_prefix} 0", fontsize=13)
        def update(frame):
            pos = position_history[frame] + 1e-12
            mom = momentum_history[frame] + 1e-12
            im_pos.set_data(pos)
            im_pos.set_norm(LogNorm(vmin=pos.min(), vmax=pos.max()))
            im_mom.set_data(mom)
            im_mom.set_norm(LogNorm(vmin=mom.min(), vmax=mom.max()))
            fig.suptitle(f"{title_prefix} {frame}", fontsize=13)
            return im_pos, im_mom
        anim = FuncAnimation(fig, update, frames=len(position_history), interval=1000//fps, blit=False)
        anim.save(save_path, writer=PillowWriter(fps=fps))
        plt.close(fig)

    def convergence_plot(self, peak_intensities, title="Momentum Peak Convergence", save_path=None):
        fig, ax = plt.subplots(figsize=(7, 4), dpi=self.dpi)
        ax.plot(peak_intensities, "-o", color="#e25822", markersize=4, linewidth=1.5)
        ax.set_xlabel("Evolution Step"); ax.set_ylabel("Peak Intensity (norm.)")
        ax.set_title(title, fontsize=13); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path: fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig
