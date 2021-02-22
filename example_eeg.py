import numpy as np
import mne
from timewarp import tfr_timewarp


# load data
raw = mne.io.read_raw_bdf("/Users/clemens/Downloads/testfiles/S01.bdf",
                          exclude=[f"EXG{i}" for i in range(1, 9)],
                          preload=True)
events = mne.find_events(raw, uint_cast=True)
raw.drop_channels("Status")
raw.filter(1, 50)

durations = np.genfromtxt("/Users/clemens/Downloads/testfiles/S01_RT.csv",
                          skip_header=1)
tmax = durations.max()

epochs = mne.Epochs(raw, events, event_id=dict(onset=2), tmin=-2, tmax=tmax,
                    baseline=None, preload=True)

# plot classical TFR
freqs = np.arange(1, 36, 0.5)
tfr = mne.time_frequency.tfr_multitaper(epochs[durations > 0], freqs=freqs,
                                        n_cycles=freqs, picks="C3",
                                        average=False, return_itc=False)
tfr.average().plot(baseline=None, mode="ratio", show=False)

# plot time-warped TFR
tfr_warped = tfr_timewarp(tfr, durations[durations > 0])
tfr_warped.average().plot(baseline=None, mode="ratio", show=False)
