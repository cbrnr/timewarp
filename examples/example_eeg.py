from pathlib import Path
import numpy as np
import pandas as pd
import mne
from mne.time_frequency import tfr_multitaper
from mne.baseline import rescale
from timewarp import tfr_timewarp


# load data
fpath = Path("/Users/clemens/Downloads/testfiles")
raw = mne.io.read_raw_bdf(fpath / "S01.bdf",
                          exclude=[f"EXG{i}" for i in range(1, 9)],
                          preload=True)
fs = raw.info["sfreq"]
events = mne.find_events(raw, uint_cast=True)
events = events[events[:, 2] == 2, :]  # keep only problem onset events (2)

# preprocess
raw.drop_channels("Status")
raw.set_montage("biosemi64")
raw.filter(1, 50)
bad_segments = np.loadtxt(fpath / "S01_bad_segments.csv", skiprows=1,
                          delimiter=",", ndmin=2) / fs
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
strategy = pd.read_csv(fpath / "S01_Strategy.csv",
                       usecols=["thisItem", "strat_Keys"])
strategy.columns = ["item", "strategy"]
strategy["strategy"].replace({"num_1": "retrieve", "num_2": "procedure",
                              "num_3": "procedure", "num_4": "other"},
                             inplace=True)
log = log.merge(strategy)
log.drop(columns="item", inplace=True)
rt = pd.read_csv(fpath / "S01_RT.csv")
metadata = pd.concat((tmp, log, rt), axis="columns")

tmax = metadata["rt"].max()

epochs = mne.Epochs(raw, events, event_id=dict(onset=2), tmin=-2, tmax=tmax,
                    baseline=None, reject_by_annotation=True, preload=True,
                    metadata=metadata)

freqs = np.arange(1, 36, 0.5)

# retrieved problems
query = "rt > 0 and correct == 0 and strategy == 'retrieve'"

# plot classical TFR
tfr1 = tfr_multitaper(epochs[query], freqs=freqs, n_cycles=freqs, picks="P3",
                      average=False, return_itc=False).crop(tmin=-1.5)
tfr1.average().plot(baseline=(None, 0), mode="percent", dB=False)

# plot time-warped TFR
tfr1_warped = tfr_timewarp(tfr1, epochs[query].metadata["rt"].values).average()
tfr1_warped.data = rescale(tfr1_warped.data, tfr1_warped.times,
                           baseline=(None, 0), mode="percent")
tfr1_warped.plot()

# plots for procedural problems
query = "rt > 0 and correct == 0 and strategy == 'procedure'"

# plot classical TFR
tfr2 = tfr_multitaper(epochs[query], freqs=freqs, n_cycles=freqs, picks="P3",
                      average=False, return_itc=False).crop(tmin=-1.5)
tfr2.average().plot(baseline=(None, 0), mode="percent", dB=False)

# plot time-warped TFR
tfr2_warped = tfr_timewarp(tfr2, epochs[query].metadata["rt"].values).average()
tfr2_warped.data = rescale(tfr2_warped.data, tfr2_warped.times,
                           baseline=(None, 0), mode="percent")
tfr2_warped.plot()

# difference between retrieved and procedural problems
tfr_diff = tfr1_warped - tfr2_warped
tfr_diff.plot()