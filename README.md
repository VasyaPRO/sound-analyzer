# Sound Analyzer

This is a command line TUI tool for sound visualization and pitch detection.
It is designed primarily for guitar tuning, but it can be used for other purposes as well.

Sound analyzer captures samples from default input device and displays information about it in real time.

Sound analyzer uses YIN algorithm for pitch detection.
It can visualize either CMNDF (from original YIN paper) or amplitude spectrum based on FFT.

## Hotkeys
For instrument tuning purposes it is useful to display cents (1/100 of a semitone) to nearest note.  
As the nearest note may change frequently you can also set target frequency manually using the following hotkeys:
* use `a`, `b`, `c`, `d`, `e`, `f`, `g` keys to set target frequency to correspondent note in 4th octave (from `C4` to `B4`).  
* use `Up`, `Down` keys to set target frequency on one *octave* higher or lower.  
* use `Left`, `Right` keys to set target frequency on one *semitone* higher or lower. 
* use `Esc` key to unset target frequency.  

You can also click and drag on graph with mouse to set your target frequency.

Other hotkeys:
* use `Tab` key to toggle between amplitude spectrum graph and CMNDF graph.  
* use `w` key to toggle Hann window on and off in amplitude spectrum mode.  
* use `Ctrl+c` key to terminate application.  
