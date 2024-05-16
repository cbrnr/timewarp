"""Time-warp variable-length time-frequency maps."""

import numpy as np
from mne import pick_types
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
