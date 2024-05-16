"""Time-warping example using real EEG data."""

# make sure that the working directory is set to the examples folder

import mne
import numpy as np
import pandas as pd
from timewarp import plot_tfr_grid, tfr_timewarp_multichannel

# load data (get it from https://osf.io/download/5fhye/)
raw = mne.io.read_raw_bdf("S01.bdf", exclude=[f"EXG{i}" for i in range(1, 9)], preload=True)
fs = raw.info["sfreq"]
events = mne.find_events(raw, uint_cast=True)
events = events[events[:, 2] == 2, :]  # keep only problem onset events (2)

# preprocess
raw.drop_channels("Status")
raw.set_montage("biosemi64")
raw.set_eeg_reference("average")

# load meta data
metadata = pd.read_csv("metadata.csv")
tmax = metadata["rt"].max()

epochs = mne.Epochs(
    raw,
    events,
    event_id=dict(onset=2),
    tmin=-1.5 - 0.5,  # add 0.5s before each baseline
    tmax=tmax + 0.5,  # add 0.5s after each epoch
    baseline=None,
    reject_by_annotation=True,
    metadata=metadata,
    preload=True,
).resample(100)
freqs = np.arange(2, 31, 0.5)

# retrieved problems
query = "rt > 0 and correct == 0 and strategy == 'retrieve'"
durations = epochs[query].metadata["rt"].values

tfr_retrieve = tfr_timewarp_multichannel(epochs[query], durations, freqs, freqs, n_jobs=4)
tfr_retrieve.apply_baseline(baseline=(None, -0.25), mode="percent")

# procedural problems
query = "rt > 0 and correct == 0 and strategy == 'procedure'"
durations = epochs[query].metadata["rt"].values

tfr_procedure = tfr_timewarp_multichannel(epochs[query], durations, freqs, freqs, n_jobs=4)
tfr_procedure.apply_baseline(baseline=(None, -0.25), mode="percent")

tfr_diff = tfr_retrieve - tfr_procedure

plot_tfr_grid(tfr_diff, "S01", figsize=(15, 10))
