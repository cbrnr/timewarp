from pathlib import Path
import numpy as np
import pandas as pd
import mne
from mne.time_frequency import tfr_multitaper
from timewarp import tfr_timewarp


# load data
fpath = Path("/Users/clemens/Downloads/testfiles")
raw = mne.io.read_raw_bdf(fpath / "S01.bdf",
                          exclude=[f"EXG{i}" for i in range(1, 9)],
                          preload=True)
fs = raw.info["sfreq"]
events = mne.find_events(raw, uint_cast=True)

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
log = pd.concat((tmp, log, rt), axis="columns")

tmax = log["rt"].max()

epochs = mne.Epochs(raw, events, event_id=dict(onset=2), tmin=-2, tmax=tmax,
                    baseline=None, reject_by_annotation=False, preload=True)

freqs = np.arange(1, 36, 0.5)

# plots for retrieved problems
ix = (log["rt"] > 0) & (log["correct"] == 0) & (log["strategy"] == "retrieve")

# plot classical TFR
tfr1 = tfr_multitaper(epochs[ix], freqs=freqs, n_cycles=freqs, picks="C3",
                      average=False, return_itc=False)
tfr1.average().plot(baseline=(None, 0), mode="ratio", dB=True)

# plot time-warped TFR
tfr1_warped = tfr_timewarp(tfr1, log["rt"][ix].values)
tfr1_warped.average().plot(baseline=(None, 0), mode="ratio", dB=True)

# plots for procedural problems
ix = (log["rt"] > 0) & (log["correct"] == 0) & (log["strategy"] == "procedure")

# plot classical TFR
tfr2 = tfr_multitaper(epochs[ix], freqs=freqs, n_cycles=freqs, picks="C3",
                      average=False, return_itc=False)
tfr2.average().plot(baseline=(None, 0), mode="ratio", dB=True)

# plot time-warped TFR
tfr2_warped = tfr_timewarp(tfr2, log["rt"][ix].values)
tfr2_warped.average().plot(baseline=(None, 0), mode="ratio", dB=True)
