"""Time-warping example using toy data."""

import numpy as np
from mne.time_frequency import tfr_multitaper
from timewarp import tfr_timewarp, generate_epochs


# generate toy data
epochs, durations = generate_epochs(baseline=2.5)
epochs.plot_image(colorbar=False, evoked=False, title="Epochs")

# plot classical TFR
freqs = np.arange(1, 36)
tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=freqs, average=False, return_itc=False)
tfr.average().plot(baseline=(-2.5, -0.5), mode="ratio", title="Classic")

# plot time-warped TFR
tfr_warped = tfr_timewarp(tfr, durations)
tfr_warped.average().plot(baseline=(-2.5, -0.5), mode="ratio", title="Time-warped")

# plot time-warped TFR with custom resampling (baseline 1000 and activity 8000 samples)
tfr_warped1 = tfr_timewarp(tfr, durations, resample=(1000, 8000))
tfr_warped1.average().plot(baseline=(-1, -0.25), mode="ratio", title="Time-warped (custom)")
