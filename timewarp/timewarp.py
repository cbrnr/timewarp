import numpy as np
from scipy.signal import resample_poly
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
        Baseline duration (in s).
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
        array[i, :, half1] = 2e-6 * np.sin(2 * np.pi * f1 * t[half1])
        array[i, :, half2] = 1e-6 * np.sin(2 * np.pi * f2 * t[half2])

    info = create_info(chs, sfreq=fs, ch_types="eeg")
    if baseline > 0:
        zeros = np.zeros((n, chs, int(baseline * fs)))
        array = np.concatenate((zeros, array), axis=-1)
    if append > 0:
        zeros = np.zeros((n, chs, int(append * fs)))
        array = np.concatenate((array, zeros), axis=-1)
    epochs = EpochsArray(array, info, tmin=-baseline)
    return epochs, durations
