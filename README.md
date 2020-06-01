# MD_SurrogateModel_Demo_20200206
Demonstration of the surrogate model and EPICS interface

## Surrogate Model EPICS Interface Demo
This demo was presented at the MD meeting at SLAC on Feb 6, 2020. The current state of this tool is only meant for demonstration. The full capabilities and further development of these tools is still in progress.

## Installation
Dependencies for running the notebook include tensorflow version 1.15, and keras version 2.2.4.

To get these modules:
```
$ conda env create -f environment.yml $
```
```
$ conda activate online-surrogate-model $
```

Tensorflow is installed using CPU only.


## To Run the Demo

Open two terminal windows and three internet browser tabs/windows. In each terminal, activate the online-surrogate-model conda environment.

In the first terminal, type (replacing) `protocol` with `pva` for PVAccess and `ca` for Channel Access:

```
$ python bin/cli.py serve start-server {protocol}
```
In the other terminal:

```
$ bokeh serve online_model/app/controls.py online_model/app/striptool.py online_model/app/image_viewer.py --args -p {protocol}
```

In each of the three internet browser tabs/windows, open each of the GUIs:

http://localhost:5006/controls

http://localhost:5006/striptool

http://localhost:5006/image_viewer



The PVAccess process variables can be monitored in an additional terminal window (with the conda environment activated) with the command:
```
$ python -m p4p.client.cli monitor {pvname}
```

The metadata on the NTNDArray process variable can be accessed from the command line using the command:
```
$ python -m p4p.client.cli --raw get smvm:x:y
```

The Channel Access process variables can be monitored using the command:
```
$ caget {pvname}
```


## Possible issues
You may receive many Tensorflow warnings, due to depreciation - these can be ignored and will be fixed in the future.

Please contact Lipi Gupta, lgupta@slac.stanford.edu if you have further issues

## Development

In addition to the project environmment a .yml file has been included for a development environment `environment-dev.yml`. If using this library, you can also make use of the package pre-commit hooks, which will format the code using black formatting (https://github.com/psf/black) before committing to the actual package.

These can be set up by running the following command inside the `online-surrogate-model-dev` conda environment:
`pre-commit install`
