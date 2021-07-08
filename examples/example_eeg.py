"""Time-warping example using real EEG data."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mne.time_frequency import tfr_multitaper
from timewarp import tfr_timewarp


def plot_tfr_grid(tfr, title=None, figsize=None, show=True):
    """Plot TFRs of channels in a grid."""
    grid = dict(Fp1=(0, 3), Fpz=(0, 4), Fp2=(0, 5), AF7=(1, 2), AF3=(1, 3), AFz=(1, 4),
                AF4=(1, 5), AF8=(1, 6), F7=(2, 0), F5=(2, 1), F3=(2, 2), F1=(2, 3),
                Fz=(2, 4), F2=(2, 5), F4=(2, 6), F6=(2, 7), F8=(2, 8), FT7=(3, 0),
                FC5=(3, 1), FC3=(3, 2), FC1=(3, 3), FCz=(3, 4), FC2=(3, 5), FC4=(3, 6),
                FC6=(3, 7), FT8=(3, 8), T7=(4, 0), C5=(4, 1), C3=(4, 2), C1=(4, 3),
                Cz=(4, 4), C2=(4, 5), C4=(4, 6), C6=(4, 7), T8=(4, 8), TP7=(5, 0),
                CP5=(5, 1), CP3=(5, 2), CP1=(5, 3), CPz=(5, 4), CP2=(5, 5), CP4=(5, 6),
                CP6=(5, 7), TP8=(5, 8), P7=(6, 0), P5=(6, 1), P3=(6, 2), P1=(6, 3),
                Pz=(6, 4), P2=(6, 5), P4=(6, 6), P6=(6, 7), P8=(6, 8), PO7=(7, 2),
                PO3=(7, 3), POz=(7, 4), PO4=(7, 5), PO8=(7, 6), O1=(8, 3), Oz=(8, 4),
                O2=(8, 5), P9=(7, 0), P10=(7, 8))
    fig, axes = plt.subplots(9, 9, sharex=True, sharey=True, figsize=figsize)
    for ax in axes.flat:  # turn all axes off by default
        ax.set_axis_off()
    for ch in set(tfr.ch_names) - {"Iz"}:
        ax = axes[grid[ch]]
        ax.axis("on")
        tfr.plot(picks=[ch], axes=ax, vmin=-1, vmax=1, colorbar=False, show=False,
                 verbose=False)
        ax.axvline(color="black", linewidth=0.5, linestyle="--")
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.text(0.95, 0.85, ch, transform=ax.transAxes, size=8, horizontalalignment="right")
    for row in range(9):  # show y-axis labels only in left column
        ax = axes[(row, 0)]
        if ax.axison:
            ax.set_ylabel(r"$\it{f}$ (Hz)", size=8)
            ax.tick_params(labelsize=7)
    for col in range(9):  # show x-axis labels only in bottom row
        ax = axes[(8, col)]
        if ax.axison:
            ax.set_xlabel(r"$\it{t}$ (s)", size=8)
            ax.tick_params(labelsize=7)
    fig.suptitle(title)
    if show:
        fig.show()
    return fig.tight_layout()


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
raw.filter(1, 50)
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
freqs = np.arange(1, 36, 0.5)

# retrieved problems
query = "rt > 0 and correct == 0 and strategy == 'retrieve'"
durations = epochs[query].metadata["rt"].values
chs = mne.pick_types(epochs.info, eeg=True)
chunk = 4  # this should equal the number of CPU cores (for parallel computation)
for i in range(0, len(chs), chunk):
    ch = chs[i:i + chunk]
    tfr = tfr_multitaper(epochs[query], freqs, freqs, picks=ch, n_jobs=min(chunk, len(ch)),
                         average=False, return_itc=False).crop(tmin=-1.5)
    tmp = tfr_timewarp(tfr, durations).average()
    tmp.apply_baseline(baseline=(None, -0.5), mode="percent")
    if i == 0:
        tfr_warped = tmp
    else:
        tfr_warped.add_channels([tmp])

tfr_warped.save("S01-retrieve-tfr.h5")

# procedural problems
query = "rt > 0 and correct == 0 and strategy == 'procedure'"
durations = epochs[query].metadata["rt"].values
chs = mne.pick_types(epochs.info, eeg=True)
chunk = 4  # this should equal the number of CPU cores (for parallel computation)
for i in range(0, len(chs), chunk):
    ch = chs[i:i + chunk]
    tfr = tfr_multitaper(epochs[query], freqs, freqs, picks=ch, n_jobs=min(chunk, len(ch)),
                         average=False, return_itc=False).crop(tmin=-1.5)
    tmp = tfr_timewarp(tfr, durations).average()
    tmp.apply_baseline(baseline=(None, -0.5), mode="percent")
    if i == 0:
        tfr_warped = tmp
    else:
        tfr_warped.add_channels([tmp])

tfr_warped.save("S01-procedure-tfr.h5")

# show difference between retrieved and procedural problems
tfr_retrieve = mne.time_frequency.read_tfrs("S01-retrieve-tfr.h5", condition=0)
tfr_procedure = mne.time_frequency.read_tfrs("S01-procedure-tfr.h5", condition=0)

tfr_diff = tfr_retrieve - tfr_procedure

plot_tfr_grid(tfr_diff, "S01", figsize=(15, 10))
