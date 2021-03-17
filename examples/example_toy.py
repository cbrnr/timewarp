import numpy as np
from mne.time_frequency import tfr_multitaper
from timewarp import tfr_timewarp, generate_epochs


# generate toy data
epochs, durations = generate_epochs(baseline=2.5, append=4)
epochs.plot_image(colorbar=False, evoked=False, title="Epochs")

# plot classical TFR
freqs = np.arange(1, 36)
tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=freqs, picks=0,
                     average=False, return_itc=False)
tfr.average().plot(baseline=None, mode="ratio", title="Classic")

# plot time-warped TFR
tfr_warped = tfr_timewarp(tfr, durations)
tfr_warped.average().plot(baseline=None, mode="ratio", title="Time-warped")
