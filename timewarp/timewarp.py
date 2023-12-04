"""Time-warp variable-length TFRs."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from mne import EpochsArray, create_info, pick_types
from mne.time_frequency import EpochsTFR, tfr_multitaper
from scipy.signal import resample_poly as rs


def tfr_timewarp(tfr, durations, resample=None):
    """Timewarp TFR for variable-length epochs.

    Parameters
    ----------
    tfr : mne.time_frequency.EpochsTFR
        Precomputed EpochsTFR. Time-warping is based on the duration of the longest epoch.
    durations : numpy.ndarray
        Duration of each epoch (in s).
    resample : tuple | None
        If tuple, must consist of two integers, where the first element specifies the number
        of samples to resample the baseline period (times < 0) and the second element
        specifies the number of samples to resample the activity period (times >= 0). If
        None, do not resample.

    Returns
    -------
    warped : mne.time_frequency.EpochsTFR
        Time-warped EpochsTFR.
    """
    fs = tfr.info["sfreq"]
    start = np.zeros_like(durations, dtype=int)
    stop = np.round(durations * fs).astype(int) + 1  # + 1 because stop index is excluded
    times = tfr.times
    baseline_idx, activity_idx = times < 0, times >= 0
    if resample is None:
        n_baseline, n_activity = baseline_idx.sum(), activity_idx.sum()
    else:
        n_baseline, n_activity = resample
        # use artificial time limits -1 to 5 s
        times = np.hstack((np.linspace(-1, 0, resample[0]), np.linspace(0, 5, resample[1])))

    # time-warp activity
    activity = np.empty((*tfr.data.shape[:-1], n_activity))
    for i, epoch in enumerate(tfr.data):
        cropped = epoch[..., np.arange(start[i], stop[i]) + baseline_idx.sum()]
        activity[i] = rs(cropped, n_activity, cropped.shape[-1], axis=-1, padtype="line")

    # resample baseline if requested
    baseline = tfr.data[..., baseline_idx]
    if n_baseline != baseline_idx.sum():
        baseline = rs(baseline, n_baseline, baseline.shape[-1], axis=-1, padtype="line")

    data = np.concatenate((baseline, activity), axis=-1)

    return EpochsTFR(tfr.info, data, times, tfr.freqs)


def tfr_timewarp_multichannel(epochs, durations, freqs, n_cycles, resample=None, n_jobs=1):
    """Compute time-warped TFRs in parallel.

    Although MNE-Python functions for computing TFRs support parallel execution through the
    n_jobs parameter, they impose a significant memory burden when dealing with many epochs
    and channels. The reason is that time-warping requires EpochsTFR objects as opposed to
    AverageTFR objects, and all results must be available in memory (even when computing in
    parallel). Only after time-warping can a (time-warped) EpochsTFR object be reduced to an
    AverageTFR object. Therefore, this function divides the Epochs data into batches of a
    few channels each, which can be handled in parallel. The resulting EpochsTFR objects can
    then be time-warped, after which they are immediately reduced to AverageTFR objects.
    After processing all batches, the function combines the individual results into a final
    time-warped AverageTFR object containing all original channels.

    Parameters
    ----------
    epochs : mne.Epochs
        TODO
    durations : numpy.ndarray
        Duration of each epoch (in s).
    freqs : array-like
        TODO
    n_cycles : array-like
        TODO
    resample : tuple | None
        If tuple, must consist of two integers, where the first element specifies the number
        of samples to resample the baseline period (times < 0) and the second element
        specifies the number of samples to resample the activity period (times >= 0). If
        None, do not resample.
    n_jobs : int
        Number of jobs running in parallel (should be at most the number of CPU cores).
    """
    chs = pick_types(epochs.info, eeg=True, meg=True)
    for i in range(0, len(chs), n_jobs):
        ch = chs[i : i + n_jobs]
        tfr = tfr_multitaper(
            epochs,
            freqs,
            n_cycles,
            picks=ch,
            average=False,
            n_jobs=min(n_jobs, len(ch)),
            return_itc=False,
        )
        tfr.crop(tmin=tfr.times[0] + 0.5, tmax=tfr.times[-1] - 0.5)
        tmp = tfr_timewarp(tfr, durations, resample).average()
        if i == 0:
            tfr_warped = tmp
        else:
            tfr_warped.add_channels([tmp])
    return tfr_warped


def generate_epochs(n=30, chs=1, fs=500, f1=10, f2=20, baseline=0):
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
    epochs = EpochsArray(array, info, tmin=-baseline)
    return epochs, durations


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
