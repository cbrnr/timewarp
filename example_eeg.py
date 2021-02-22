from pathlib import Path
import numpy as np
import pandas as pd
import mne
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

# load response times (epoch length)
durations = np.genfromtxt("/Users/clemens/Downloads/testfiles/S01_RT.csv",
                          skip_header=1)
tmax = durations.max()

# load accuracy information
correct = pd.read_csv(fpath / "S01_EEG.csv")["corr"].values

epochs = mne.Epochs(raw, events, event_id=dict(onset=2), tmin=-2, tmax=tmax,
                    baseline=None, reject_by_annotation=False, preload=True)

sel = (durations > 0) & (correct == 0)  # epoch selection

# plot classical TFR
freqs = np.arange(1, 36, 0.5)
tfr = mne.time_frequency.tfr_multitaper(epochs[sel], freqs=freqs,
                                        n_cycles=freqs, picks="C3",
                                        average=False, return_itc=False)
tfr.average().plot(baseline=(None, 0), mode="ratio", dB=True)

# plot time-warped TFR
tfr_warped = tfr_timewarp(tfr, durations[sel])
tfr_warped.average().plot(baseline=(None, 0), mode="ratio", dB=True)
