# Time-warping time-frequency maps
## Motivation

## Workflow
1. Load raw data and create epochs around events of interest. Make sure that epochs completely cover the *longest* epoch in the data. Since MNE supports only constant-length epochs, shorter epochs will contain irrelevant data at the end. This is OK, because the next steps will take care of this issue. Therefore, after this step you should have an `mne.Epochs` object consisting of `n` epochs (let's call it `epochs`) and an array of `n` epoch durations (in seconds) (let's call it `durations`).
2. Now we compute a standard time-frequency representation (TFR) from the epoched data. MNE currently includes three TFR functions in `mne.time_frequency`, namely `tfr_morlet`, `tfr_multitaper`, and `tfr_stockwell`. All of them produce a suitable TFR that can be used for time-warping. However, these functions can compute either averaged (`average=True`, the default) or single-epoch (`average=False`) TFRs. Since time-warping requires single-epoch TFRs (an `mne.time_frequency.EpochsTFR` object), we need to pass `average=False` (we will average them later).
3. Finally, we time-warp the `mne.time_frequency.EpochsTFR` object by passing it to the `tfr_timewarp` function, together with the `durations` defined in the first step. This will stretch all single-epoch TFRs to the same length, a process which we refert to as time-warping. Note that `tfr_timewarp` returns another `mne.time_frequency.EpochsTFR` object with the same dimensions as the input object. However, the data from time 0 to the last time point is now time-warped, which means that it cannot be interpreted as time in seconds, but time as a percentage (ranging from 0% to 100%).
4. Plotting (or post-processing) the time-warped `mne.time_frequency.EpochsTFR` object usually involves averaging over all epochs first. This can be achieved by calling the `average` method.

## Example

