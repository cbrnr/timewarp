import numpy as np
from mne import EpochsArray, create_info


def generate_epochs(n=30, chs=1, fs=500, f1=10, f2=20, baseline=0):
    """Create toy data consisting of epochs with variable random lengths.

    Each of the n epochs contains an oscillation with f1 Hz in its first half and an
    oscillation with f2 Hz in its second half. MNE supports only rectangular (constant-length)
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
