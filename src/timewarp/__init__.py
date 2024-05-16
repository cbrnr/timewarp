"""Time-warp variable-length time-frequency maps."""

from timewarp.plot import plot_tfr_grid
from timewarp.timewarp import tfr_timewarp, tfr_timewarp_multichannel
from timewarp.utils import generate_epochs

__all__ = ["generate_epochs", "plot_tfr_grid", "tfr_timewarp", "tfr_timewarp_multichannel"]

__version__ = "0.1.0"
