"""Time-warp variable-length TFRs."""

from .timewarp import (
    generate_epochs,
    plot_tfr_grid,
    tfr_timewarp,
    tfr_timewarp_multichannel,
)

__all__ = ["tfr_timewarp", "tfr_timewarp_multichannel", "plot_tfr_grid", "generate_epochs"]

__version__ = "0.1.0"
