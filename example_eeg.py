import mne
from timewarp import tfr_timewarp


raw = mne.io.read_raw_bdf("/Users/clemens/Downloads/testfiles/S01.bdf",
                          exclude=[f"EXG{i}" for i in range(1, 9)],
                          preload=True)
events = mne.find_events(raw, uint_cast=True)
raw.drop_channels("Status")
