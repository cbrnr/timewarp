import numpy as np
from scipy.signal import resample_poly
import mne


raw = mne.io.read_raw_bdf("S01.bdf",
                          exclude=["EXG" + str(i) for i in range(1, 9)],
                          preload=True)

events = mne.find_events(raw, uint_cast=True)
raw.drop_channels(["Status"])
raw.set_eeg_reference("average")

onsets = events[events[:, 2] == 2, 0]
offsets = events[np.in1d(events[:, 2], [31, 32]), 0]
durations = offsets - onsets
max_samps = durations.max()

tmin, tmax = -1.75, 5
baseline = [-1.75, 0]
activity = [0, 5]
delta = 0.5  # extra time to add to beginning and end of epochs
epochs = mne.Epochs(raw, events, dict(onset=2),
                    tmin=tmin - delta, tmax=tmax + delta,
                    baseline=None, preload=True)

freqs = np.arange(1, 51)
picks = [0]  # pick only the first channel
tfr = mne.time_frequency.tfr_multitaper(epochs, freqs=freqs, n_cycles=freqs,
                                        picks=picks, average=False,
                                        return_itc=False)

tfr.average().crop(tmin, tmax).plot(baseline=baseline, mode="ratio", dB=True)

# timewarp TFRs of individual epochs
tstop = epochs.time_as_index(durations / raw.info["sfreq"])
tstart = np.full_like(tstop, epochs.time_as_index(activity[0]))
trange = np.column_stack((tstart, tstop))
tfr_resamp = np.empty((*tfr.data.shape[:-1], max_samps))
for idx, epoch in enumerate(tfr.data):
    epoch_cropped = epoch[:, :, np.arange(*trange[idx])]
    tfr_resamp[idx] = resample_poly(epoch_cropped, max_samps,
                                    epoch_cropped.shape[-1], axis=-1)
    
prestim = tfr.data[..., np.arange(*epochs.time_as_index([tmin - delta, 0]))]

tfr_new = np.concatenate((prestim, tfr_resamp), axis=-1)
fill = tfr.data.shape[-1] - tfr_new.data.shape[-1]
tfr_new = np.concatenate((tfr_new, np.zeros((*tfr.data.shape[:-1], fill))),
                         axis=-1)

tfr_warped = tfr.copy()
tfr_warped.data = tfr_new

fig = tfr_warped.average().crop(tmin, tmax).plot(baseline=baseline,
                                                 mode="ratio", dB=True)
fig.get_axes()[0].set_xticklabels(["", "", "0%", "20%",
                                   "40%", "60%", "80%", "100%"])
