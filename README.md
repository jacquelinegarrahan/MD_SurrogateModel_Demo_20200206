# MD_SurrogateModel_Demo_20200206
Demonstration of the surrogate model and EPICS interface

## Surrogate Model EPICS Interface Demo
This demo was presented at the MD meeting at SLAC on Feb 6, 2020. The current state of this tool is only meant for demonstration. The full capabilities and further development of these tools is still in progress.

## Installation
Dependencies for running the notebook include tensorflow version 1.15, and keras version 2.2.4.

To get these modules:
```
conda create -n smdemo -c conda-forge python=3.7 tensorflow keras pyepics pcaspy bokeh matplotlib scikit-learn h5py
conda activate smdemo

```
pcaspy requires an EPICS install and has instructions on its website:
https://pcaspy.readthedocs.io/en/latest/


## To Run the Demo

Open two terminal windows and three internet browser tabs/windows.

In the first terminal, type:

```
./serve.sh

```
In the other terminal:

```
./view.bash
```

In each of the three internet browser tabs/windows, open each of the GUIs:

http://localhost:5006/controls

http://localhost:5006/striptool

http://localhost:5006/image_viewer

## Possible issues
You may receive many Tensorflow warnings, due to depreciation - these can be ignored and will be fixed in the future.

It may be necessary to add a path to the epics installation in your .bashrc or .bash_profile

Please contact Lipi Gupta, lgupta@slac.stanford.edu if you have further issues
