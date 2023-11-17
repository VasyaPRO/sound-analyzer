# Sound Analyzer

This is a command line TUI tool for sound visualization and pitch detection.
It is designed primarily for guitar tuning, but it can be used for other purposes as well.

Sound analyzer captures samples from the default input device and displays information about it in real time.

Sound analyzer uses YIN algorithm for pitch detection.
It can visualize either CMNDF (from original YIN [paper](http://audition.ens.fr/adc/pdf/2002_JASA_YIN.pdf)) or amplitude spectrum based on FFT.

## Hotkeys
For instrument tuning purposes it is useful to display cents (1/100 of a semitone) to the nearest note.  
As the nearest note may change frequently you can also set the target frequency manually using the following hotkeys:
* use `a`, `b`, `c`, `d`, `e`, `f`, `g` keys to set target frequency to correspondent note in 4th octave (from `C4` to `B4`).  
* use `Up`, `Down` keys to set target frequency on one *octave* higher or lower.  
* use `Left`, `Right` keys to set target frequency on one *semitone* higher or lower. 
* use `Esc` key to unset the target frequency.  

You can also click and drag on the graph with your mouse to set the target frequency.

Other hotkeys:
* use `Tab` key to toggle between the amplitude spectrum graph and the CMNDF graph.  
* use `w` key to toggle the Hann window on and off in amplitude spectrum mode.  
* use `Ctrl+c` key to terminate the application.  

## Demo
![demo](https://github.com/VasyaPRO/sound-analyzer/assets/16972410/ded29dac-2f62-4e45-9e38-7ae2a2533473)
