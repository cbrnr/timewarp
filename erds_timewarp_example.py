import numpy as np
from scipy.signal import resample_poly
import mne
from mne.io import read_raw_eeglab


raw = read_raw_eeglab("Eeglab_data.set", stim_channel=False, preload=True)
raw.set_eeg_reference("average")

events, _ = mne.events_from_annotations(raw, dict(square=1, rt=2))

new_events = []
previous = None
for event in events:
    if previous is None:
        previous = event
    elif (previous[2] == 1) and (event[2] == 2):
        new_events.append(previous)
        new_events.append(event)
    previous = event
new_events = np.array(new_events)

onsets = new_events[::2, 0]
offsets = new_events[1::2, 0]
durations = offsets - onsets
max_samps = durations.max()

tmin, tmax = -1, 2
epochs = mne.Epochs(raw, new_events, dict(square=1), tmin, tmax)

picks = [26]  # POz
freqs = np.arange(1, 51)

tfr = mne.time_frequency.tfr_multitaper(epochs, freqs=freqs, n_cycles=freqs,
                                        picks=picks, average=False,
                                        return_itc=False)
tfr.average().plot(baseline=(None, 0), mode="ratio", dB=True)


tstop = epochs.time_as_index(durations / raw.info["sfreq"])
tstart = np.full_like(tstop, epochs.time_as_index(0))
trange = np.column_stack((tstart, tstop))
tfr_resamp = np.empty((*tfr.data.shape[:-1], max_samps))

for idx, epoch in enumerate(tfr.data):
    epoch_cropped = epoch[:, :, np.arange(*trange[idx])]
    tfr_resamp[idx] = resample_poly(epoch_cropped, max_samps,
                                    epoch_cropped.shape[-1], axis=-1)

prestim = tfr.data[..., np.arange(*epochs.time_as_index([tmin, 0]))]

tfr_new = np.concatenate((prestim, tfr_resamp), axis=-1)
fill = tfr.data.shape[-1] - tfr_new.data.shape[-1]
tfr_new = np.concatenate((tfr_new, np.zeros((*tfr.data.shape[:-1], fill))),
                         axis=-1)

tfr_warped = tfr.copy()
tfr_warped.data = tfr_new

tfr_warped.average().plot(baseline=(None, 0), mode="ratio", dB=True,
                          vmin=-7, vmax=7)
