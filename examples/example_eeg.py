"""Time-warping example using real EEG data."""

from pathlib import Path
import numpy as np
import pandas as pd
import mne
from timewarp import tfr_timewarp_multichannel, plot_tfr_grid


# load data
fpath = Path("/Users/clemens/Downloads/testfiles")
raw = mne.io.read_raw_bdf(fpath / "S01.bdf", exclude=[f"EXG{i}" for i in range(1, 9)],
                          preload=True)
fs = raw.info["sfreq"]
events = mne.find_events(raw, uint_cast=True)
events = events[events[:, 2] == 2, :]  # keep only problem onset events (2)

# preprocess
raw.drop_channels("Status")
raw.set_montage("biosemi64")
bad_segments = np.loadtxt(fpath / "S01_bad_segments.csv", skiprows=1, delimiter=",") / fs
raw.set_annotations(mne.Annotations(*bad_segments.T, "bad_segment"))

with open(fpath / "S01_bad_channels.csv") as f:
    raw.info["bads"] = f.read().strip().split(",")

raw.set_eeg_reference("average")

ica = mne.preprocessing.read_ica(fpath / "S01-ica.fif.gz")
ica.apply(raw)

# load meta information
log = pd.read_csv(fpath / "S01_EEG.csv", usecols=["thisItem", "corr"])
log.columns = ["item", "correct"]
tmp = log["item"].str.split(expand=True)
tmp.columns = ["op1", "op", "op2"]
tmp["op1"] = tmp["op1"].astype(int)
tmp["op2"] = tmp["op2"].astype(int)
strategy = pd.read_csv(fpath / "S01_Strategy.csv", usecols=["thisItem", "strat_Keys"])
strategy.columns = ["item", "strategy"]
strategy["strategy"].replace({"num_1": "retrieve", "num_2": "procedure",
                              "num_3": "procedure", "num_4": "other"}, inplace=True)
log = log.merge(strategy, how="left")
log.drop(columns="item", inplace=True)
rt = pd.read_csv(fpath / "S01_RT.csv")
metadata = pd.concat((tmp, log, rt), axis="columns")

tmax = metadata["rt"].max()

epochs = mne.Epochs(raw, events, event_id=dict(onset=2), tmin=-2, tmax=tmax, baseline=None,
                    reject_by_annotation=True, metadata=metadata, preload=True)
freqs = np.arange(2, 36, 0.5)

# retrieved problems
query = "rt > 0 and correct == 0 and strategy == 'retrieve'"
durations = epochs[query].metadata["rt"].values

tfr_warped = tfr_timewarp_multichannel(epochs[query], durations, freqs, freqs, n_jobs=4)
tfr_warped.save("S01-retrieve-tfr.h5")

# procedural problems
query = "rt > 0 and correct == 0 and strategy == 'procedure'"
durations = epochs[query].metadata["rt"].values

tfr_warped = tfr_timewarp_multichannel(epochs[query], durations, freqs, freqs, n_jobs=4)
tfr_warped.save("S01-procedure-tfr.h5")

# show difference between retrieved and procedural problems
tfr_retrieve = mne.time_frequency.read_tfrs("S01-retrieve-tfr.h5", condition=0)
tfr_procedure = mne.time_frequency.read_tfrs("S01-procedure-tfr.h5", condition=0)

tfr_diff = tfr_retrieve - tfr_procedure

plot_tfr_grid(tfr_diff, "S01", figsize=(15, 10))
