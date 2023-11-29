from collections import defaultdict
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from timewarp import plot_tfr_grid, tfr_timewarp_multichannel

# load data
subject = "s01"
fpath = Path("./data") / subject
raw = mne.io.read_raw_bdf(fpath / "s01.bdf", preload=True)
raw.set_montage("biosemi64")
fs = raw.info["sfreq"]

# load behavioral data
cols = ["GeneralDomain", "Domain_Nr", "Congruency", "Stim_RT", "Stim_Corr"]
df = pd.read_csv(fpath / (subject + ".csv"), usecols=cols)[cols]
df.columns = ["domain", "subdomain", "congruent", "rt", "correct"]
df["domain"] = df["domain"].replace({
    "Naturwissenschaften": "nature",
    "Mathematik": "math"
})
df["correct"] = (df["correct"] == 1).astype("boolean")
df["congruent"] = (df["congruent"] == 1).astype("boolean")
df.insert(0, "epoch", range(1, len(df) + 1))
df.insert(0, "id", subject)

# load manual artifact selections
if (fname := fpath / (subject + "-bad_segments.csv")).exists():
    bad_segments = np.loadtxt(fname, skiprows=1, delimiter=",", ndmin=2) / fs
    raw.set_annotations(mne.Annotations(*bad_segments.T, "bad_segment"))

# load manual bad channel selections
if (fname := fpath / (subject + "-bad_channels.csv")).exists():
    with open(fname) as f:
        raw.info["bads"] = f.read().strip().split(",")

# find events
events = mne.find_events(raw, uint_cast=True)
events = events[np.in1d(events[:, 2], [1, 2, 3, 4, 5, 100, 200]), :]

counts = np.bincount(events[:, 2])
counts = defaultdict(
    lambda: 0,
    {i: count for i, count in enumerate(counts) if count > 0}
)

# fix missing events for specific recordings
if counts[1] < 200:
    for missing in range(counts[5]):
        index = np.nonzero(events[:, 2] == 5)[0][missing]
        pos = events[index + 1, 0] - int(1.5 * fs)
        events = np.insert(events, index + 1, [pos, 0, 1], axis=0)

    counts = np.bincount(events[:, 2])
    counts = defaultdict(
        lambda: 0,
        {i: count for i, count in enumerate(counts) if count > 0}
    )

raw.load_data()
raw.drop_channels(["Status"])

# average reference (without bad channels)
raw.set_eeg_reference("average")

# interpolate bad channels
if raw.info["bads"]:
    raw.interpolate_bads()

# load and apply ICA
if (fname := fpath / (subject + "-ica.fif.gz")).exists():
    ica = mne.preprocessing.read_ica(fname)
    ica.apply(raw)

# create epochs
tmax = df["rt"].max()
epochs = mne.Epochs(
    raw,
    events[events[:, 2] == 2],
    tmin=-1.25 - 0.5,
    tmax=tmax + 0.51,  # FIXME: gotta make epochs long enough...
    baseline=None,
    preload=True,
    metadata=df,
).resample(100)

epochs = epochs["domain == 'math'"]  # only keep math domain
freqs = np.arange(2, 31)

# congruent problems
query = "rt > 0 and correct and congruent"
durations = epochs[query].metadata["rt"].values

tfr_congruent = tfr_timewarp_multichannel(
    epochs[query], durations, freqs, freqs, n_jobs=4
)
tfr_congruent.apply_baseline(baseline=(None, -0.25), mode="percent")

# incongruent problems
query = "rt > 0 and correct and not congruent"
durations = epochs[query].metadata["rt"].values

tfr_incongruent = tfr_timewarp_multichannel(
    epochs[query], durations, freqs, freqs, n_jobs=4
)
tfr_incongruent.apply_baseline(baseline=(None, -0.25), mode="percent")

tfr_diff = tfr_congruent - tfr_incongruent

plot_tfr_grid(tfr_diff, "S01", figsize=(15, 10))
