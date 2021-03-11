import numpy as np
from scipy.signal import resample_poly
from mne.time_frequency import EpochsTFR


def tfr_timewarp(tfr, durations):
    """Timewarp TFR for variable-length epochs.

    Parameters
    ----------
    tfr : mne.time_frequency.EpochsTFR
        Precomputed EpochsTFR using fixed-length epochs. Time-warping is based
        on the duration of the longest epoch.
    durations : numpy.ndarray
        Duration of each epoch (in s).

    Returns
    -------
    warped : mne.time_frequency.EpochsTFR
        Time-warped EpochsTFR.
    """
    start = np.zeros_like(durations, dtype=int)
    stop = np.round(durations * tfr.info["sfreq"]).astype(int) + 1
    longest = np.max(stop - start)
    baseline = tfr.times < 0
    data = np.empty((*tfr.data.shape[:-1], longest))
    for i, epoch in enumerate(tfr.data):
        cropped = epoch[..., np.arange(start[i], stop[i]) + baseline.sum()]
        data[i] = resample_poly(cropped, up=longest, down=cropped.shape[-1],
                                axis=-1, padtype="line")
    data = np.concatenate((tfr.data[..., baseline], data), axis=-1)
    return EpochsTFR(tfr.info, data, tfr.times[:data.shape[-1]], tfr.freqs)
