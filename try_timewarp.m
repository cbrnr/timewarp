fpath = './';

EEG = pop_biosig(fullfile(fpath, 'S01', 'S01.bdf'));

% find start (2) and stop (31, 32) events
start = find([EEG.event.type] == 2);
stop = find(ismember([EEG.event.type], [31, 32]));

% find epoch lengths
lens = [EEG.event(stop).latency] - [EEG.event(start).latency];
lats = [zeros(length(lens), 1), lens'] / EEG.srate * 1000;

% extend epochs by 0.5s in both negative and positive directions
EEG = pop_epoch(EEG, {}, [-1.75 - 0.5, 5 + 0.5], 'eventindices', start);

figure();
[ersp, itc, powbase, times, freqs, erspboot, itcboot, tfdata] = newtimef(...
    squeeze(EEG.data(1, :, :)), EEG.pnts, [EEG.xmin, EEG.xmax] * 1000, ...
    EEG.srate, 0, 'baseline', [-1750, 0], 'plotitc', 'off', ...
    'timewarp', lats, 'timewarpms', [0, 5000]);

figure();
[ersp, itc, powbase, times, freqs, erspboot, itcboot, tfdata] = newtimef(...
    squeeze(EEG.data(1, :, :)), EEG.pnts, [EEG.xmin, EEG.xmax] * 1000, ...
    EEG.srate, 0, 'baseline', [-1750, 0], 'plotitc', 'off');