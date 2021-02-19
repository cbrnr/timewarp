import numpy as np
from scipy.signal import resample_poly
from mne.time_frequency import EpochsTFR


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
    return EpochsTFR(tfr.info, data, tfr.times, tfr.freqs)
