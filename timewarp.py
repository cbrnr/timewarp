import numpy as np
from scipy.signal import resample_poly
import mne


def generate_epochs(n=30, fs=500, f1=10, f2=20, baseline=0):
    """Create toy data consisting of variable length epochs.

    Each epoch contains an oscillation with f1 Hz in the first half and an
    oscillation with f2 Hz in the second half. MNE supports only rectangular
    (constant-length) epochs, so all epochs that are shorter than the longest
    epoch will be zero-padded at the end.

    Parameters
    ----------
    n : int
        Number of epochs.
    fs : int | float
        Sampling frequency (in Hz).
    f1 : int | float
        Frequency of oscillation in first epoch half (in Hz).
    f2 : int | float
        Frequency of oscillation in second epoch half (in Hz).
    baseline : int | float
        Baseline duration (in s). If 0, no baseline is added.

    Returns
    -------
    epochs : mne.EpochsTFR
        Epochs containing toy data.
    durations : numpy.ndarray
        Duration of each epoch (in s).
    """
    rng = np.random.default_rng(1)
    durations = rng.lognormal(mean=0, sigma=2, size=n) + 1  # in seconds
    array = np.zeros((n, len(np.arange(0, durations.max(), 1 / fs))))

    for i, dur in enumerate(durations):
        t = np.arange(0, dur, 1 / fs)
        half1 = slice(0, len(t) // 2)
        half2 = slice(len(t) // 2, len(t))
        array[i][half1] = np.sin(2 * np.pi * f1 * t[half1]) * 1e-6
        array[i][half2] = np.sin(2 * np.pi * f2 * t[half2]) * 0.5e-6

    info = mne.create_info(1, sfreq=fs, ch_types="eeg")
    if baseline > 0:
        array = np.column_stack((np.zeros((n, int(baseline * fs))), array))
    epochs = mne.EpochsArray(array[:, np.newaxis, :], info, tmin=-baseline)
    return epochs, durations


def tfr_timewarp(tfr, durations):
    """Timewarp TFR for variable-length epochs.

    Parameters
    ----------
    tfr : mne.time_frequency.EpochsTFR
        Precomputed EpochsTFR using fixed-length epochs.
    durations : numpy.ndarray
        Duration of each epoch (in s).

    Returns
    -------
    warped : mne.time_frequency.EpochsTFR
        Time-warped EpochsTFR.
    """
    tstart = np.zeros_like(durations, dtype=int)
    tstop = (durations * tfr.info["sfreq"]).astype(int) + 1
    max_samp = np.max(tstop - tstart)
    data = np.empty((*tfr.data.shape[:-1], max_samp))
    baseline = tfr.times < 0
    for i, epoch in enumerate(tfr.data):
        cropped = epoch[..., np.arange(tstart[i], tstop[i]) + baseline.sum()]
        data[i] = resample_poly(cropped, up=max_samp, down=cropped.shape[-1],
                                axis=-1, padtype="line")
    data = np.concatenate((tfr.data[..., baseline], data), axis=-1)
    return mne.time_frequency.EpochsTFR(tfr.info, data, tfr.times, tfr.freqs)


# generate toy data
epochs, durations = generate_epochs(baseline=2.5)
epochs.plot_image(colorbar=False, evoked=False, title="Epochs", show=False)

# plot classical TFR
freqs = np.arange(1, 51)
tfr = mne.time_frequency.tfr_multitaper(epochs, freqs=freqs, n_cycles=freqs,
                                        picks=0, average=False,
                                        return_itc=False)
tfr.average().plot(baseline=None, mode="ratio", show=False)

# plot time-warped TFR
tfr_warped = tfr_timewarp(tfr, durations)
tfr_warped.average().plot(baseline=None, mode="ratio", show=False)
