"""Time-warp variable-length TFRs."""

import numpy as np
from scipy.signal import resample_poly
import matplotlib.pyplot as plt
from mne import create_info, EpochsArray
from mne.time_frequency import EpochsTFR


def tfr_timewarp(tfr, durations):
    """Timewarp TFR for variable-length epochs.

    Parameters
    ----------
    tfr : mne.time_frequency.EpochsTFR
        Precomputed EpochsTFR using fixed-length epochs. Time-warping is based on the
        duration of the longest epoch.
    durations : numpy.ndarray
        Duration of each epoch (in s).

    Returns
    -------
    warped : mne.time_frequency.EpochsTFR
        Time-warped EpochsTFR.
    """
    fs = tfr.info["sfreq"]
    start = np.zeros_like(durations, dtype=int)
    stop = np.round(durations * fs).astype(int) + 1
    length = np.round(tfr.times[-1] * fs).astype(int) + 1
    baseline = tfr.times < 0
    data = np.empty((*tfr.data.shape[:-1], length))
    for i, epoch in enumerate(tfr.data):
        cropped = epoch[..., np.arange(start[i], stop[i]) + baseline.sum()]
        data[i] = resample_poly(cropped, length, cropped.shape[-1], axis=-1, padtype="line")
    data = np.concatenate((tfr.data[..., baseline], data), axis=-1)
    return EpochsTFR(tfr.info, data, tfr.times[:data.shape[-1]], tfr.freqs)


def generate_epochs(n=30, chs=1, fs=500, f1=10, f2=20, baseline=0, append=0):
    """Create toy data consisting of epochs with variable random lengths.

    Each of the n epochs contains an oscillation with f1 Hz in its first half and an
    oscillation with f2 Hz in its second half. MNE supports only rectangular
    (constant-length) epochs, so all epochs that are shorter than the longest epoch will be
    zero-padded at the end.

    Parameters
    ----------
    n : int
        Number of epochs.
    chs : int
        Number of channels.
    fs : int | float
        Sampling frequency (in Hz).
    f1 : int | float
        Oscillation frequency in first epoch half (in Hz).
    f2 : int | float
        Oscillation frequency in second epoch half (in Hz).
    baseline : int | float
        Baseline duration (in s). Contains small non-zero values.
    append : int | float
        Zero-padding (in s) added to the longest epoch.

    Returns
    -------
    epochs : mne.EpochsTFR
        Epochs containing toy data.
    durations : numpy.ndarray
        Duration of each epoch (in s).
    """
    rng = np.random.default_rng(1)
    durations = rng.lognormal(mean=0, sigma=2, size=n) + 1  # in s
    array = np.zeros((n, chs, len(np.arange(0, durations.max(), 1 / fs))))

    for i, dur in enumerate(durations):
        t = np.arange(0, dur, 1 / fs)
        half1 = slice(0, len(t) // 2)
        half2 = slice(len(t) // 2, len(t))
        array[i, :, half1] = 3e-6 * np.sin(2 * np.pi * f1 * t[half1])
        array[i, :, half2] = 1e-6 * np.sin(2 * np.pi * f2 * t[half2])

    info = create_info(chs, sfreq=fs, ch_types="eeg")
    if baseline > 0:
        values = np.full((n, chs, int(baseline * fs)), 1e-7)
        array = np.concatenate((values, array), axis=-1)
    if append > 0:
        zeros = np.zeros((n, chs, int(append * fs)))
        array = np.concatenate((array, zeros), axis=-1)
    epochs = EpochsArray(array, info, tmin=-baseline)
    return epochs, durations


def plot_tfr_grid(tfr, title=None, figsize=None, show=True):
    """Plot TFRs of channels in a grid."""
    grid = dict(Fp1=(0, 3), Fpz=(0, 4), Fp2=(0, 5), AF7=(1, 2), AF3=(1, 3), AFz=(1, 4),
                AF4=(1, 5), AF8=(1, 6), F7=(2, 0), F5=(2, 1), F3=(2, 2), F1=(2, 3),
                Fz=(2, 4), F2=(2, 5), F4=(2, 6), F6=(2, 7), F8=(2, 8), FT7=(3, 0),
                FC5=(3, 1), FC3=(3, 2), FC1=(3, 3), FCz=(3, 4), FC2=(3, 5), FC4=(3, 6),
                FC6=(3, 7), FT8=(3, 8), T7=(4, 0), C5=(4, 1), C3=(4, 2), C1=(4, 3),
                Cz=(4, 4), C2=(4, 5), C4=(4, 6), C6=(4, 7), T8=(4, 8), TP7=(5, 0),
                CP5=(5, 1), CP3=(5, 2), CP1=(5, 3), CPz=(5, 4), CP2=(5, 5), CP4=(5, 6),
                CP6=(5, 7), TP8=(5, 8), P7=(6, 0), P5=(6, 1), P3=(6, 2), P1=(6, 3),
                Pz=(6, 4), P2=(6, 5), P4=(6, 6), P6=(6, 7), P8=(6, 8), PO7=(7, 2),
                PO3=(7, 3), POz=(7, 4), PO4=(7, 5), PO8=(7, 6), O1=(8, 3), Oz=(8, 4),
                O2=(8, 5), P9=(7, 0), P10=(7, 8))
    xticks = np.quantile(tfr.times[tfr.times >= 0], [0, 0.25, 0.5, 0.75, 1])
    fig, axes = plt.subplots(9, 9, sharex=True, sharey=True, figsize=figsize)
    for ax in axes.flat:  # turn all axes off by default
        ax.set_axis_off()
    for ch in set(tfr.ch_names) - {"Iz"}:
        ax = axes[grid[ch]]
        ax.axis("on")
        tfr.plot(picks=[ch], axes=ax, vmin=-1, vmax=1, colorbar=False, show=False,
                 verbose=False)
        ax.axvline(color="black", linewidth=0.5, linestyle="--")
        ax.set_xlabel(None)
        ax.set_xticks(xticks)
        ax.set_xticklabels([0, 25, 50, 75, 100])
        ax.set_ylabel(None)
        ax.text(0.03, 0.85, ch, transform=ax.transAxes, size=8)
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
    if show:
        fig.show()
    return fig.tight_layout()
