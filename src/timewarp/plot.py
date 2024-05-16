import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


def plot_tfr_grid(tfr, title=None, figsize=None, vmin=-1, vmax=1, show=True):
    """Plot TFRs of channels in a grid."""
    grid = dict(
        Fp1=(0, 3),
        Fpz=(0, 4),
        Fp2=(0, 5),
        AF7=(1, 2),
        AF3=(1, 3),
        AFz=(1, 4),
        AF4=(1, 5),
        AF8=(1, 6),
        F7=(2, 0),
        F5=(2, 1),
        F3=(2, 2),
        F1=(2, 3),
        Fz=(2, 4),
        F2=(2, 5),
        F4=(2, 6),
        F6=(2, 7),
        F8=(2, 8),
        FT7=(3, 0),
        FC5=(3, 1),
        FC3=(3, 2),
        FC1=(3, 3),
        FCz=(3, 4),
        FC2=(3, 5),
        FC4=(3, 6),
        FC6=(3, 7),
        FT8=(3, 8),
        T7=(4, 0),
        C5=(4, 1),
        C3=(4, 2),
        C1=(4, 3),
        Cz=(4, 4),
        C2=(4, 5),
        C4=(4, 6),
        C6=(4, 7),
        T8=(4, 8),
        TP7=(5, 0),
        CP5=(5, 1),
        CP3=(5, 2),
        CP1=(5, 3),
        CPz=(5, 4),
        CP2=(5, 5),
        CP4=(5, 6),
        CP6=(5, 7),
        TP8=(5, 8),
        P7=(6, 0),
        P5=(6, 1),
        P3=(6, 2),
        P1=(6, 3),
        Pz=(6, 4),
        P2=(6, 5),
        P4=(6, 6),
        P6=(6, 7),
        P8=(6, 8),
        PO7=(7, 2),
        PO3=(7, 3),
        POz=(7, 4),
        PO4=(7, 5),
        PO8=(7, 6),
        O1=(8, 3),
        Oz=(8, 4),
        O2=(8, 5),
        P9=(7, 0),
        P10=(7, 8),
    )
    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    xticks = np.quantile(tfr.times[tfr.times >= 0], [0, 0.25, 0.5, 0.75, 1])
    fig, axes = plt.subplots(9, 9, sharex=True, sharey=True, figsize=figsize)
    for ax in axes.flat:  # turn all axes off by default
        ax.set_axis_off()
    for ch in set(tfr.ch_names) - {"Iz"}:
        ax = axes[grid[ch]]
        ax.axis("on")
        tfr.plot(
            picks=[ch],
            axes=ax,
            cmap="RdBu",
            cnorm=cnorm,
            colorbar=False,
            show=False,
            verbose=False,
        )
        ax.axvline(ymax=0.8, color="black", linewidth=0.5, linestyle="--")
        ax.set_xlabel(None)
        ax.set_xticks(xticks)
        ax.set_xticklabels([0, 25, 50, 75, 100])
        ax.set_ylabel(None)
        ax.text(0.03, 0.85, ch, transform=ax.transAxes, size=8)
        cbarimage = ax.images[0]  # for colorbar
    for row in range(9):  # show y-axis labels only in left column
        ax = axes[(row, 0)]
        if ax.axison:
            ax.set_ylabel(r"$\it{f}$ (Hz)", size=8)
            ax.tick_params(labelsize=7)
    for col in range(9):  # show x-axis labels only in bottom row
        ax = axes[(8, col)]
        if ax.axison:
            ax.set_xlabel(r"$\it{t}$ (%)", size=8)
            ax.tick_params(labelsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    cbar = fig.colorbar(cbarimage, ax=fig.axes[-1], orientation="horizontal")
    cbar.ax.tick_params(labelsize=7)
    if show:
        plt.show()
    return fig
