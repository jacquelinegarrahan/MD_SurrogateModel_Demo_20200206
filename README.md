# MD_SurrogateModel_Demo_20200206
Demonstrates the integration of surrogate model with both Channel Access and PVAccess protocols.


## Installation
The environment for this project is managed using conda. The environment can be created directly from the environment.yml file.

```
$ conda env create -f environment.yml
```
```
$ conda activate online-surrogate-model
```

Note: Tensorflow is installed via pip as outlined in the tensorflow documentation. It is installed using CPU only for this project.


## To Run the Demo

Open two terminal windows and three internet browser tabs/windows. In each terminal, activate the online-surrogate-model conda environment.

In the first terminal, type (replacing `protocol` with `pva` for PVAccess and `ca` for Channel Access):

```
$ python bin/cli.py serve start-server {protocol}
```
In the other terminal:

```
$ bokeh serve online_model/app/dashboard.py --args -p {protocol}
```

In an internet browser tab, navigate to:

http://localhost:5006/dashboard

The controls in this dashboard can also be opened in individual windows using the command:

```
$ bokeh serve online_model/app/controls.py online_model/app/striptool.py online_model/app/image_viewer.py --args -p {protocol}
```

And navigating to the following pages:
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

In addition to the project environmment a .yml file has been included for a development environment `environment-dev.yml`. Using this environment, you can also make use of the package pre-commit hooks, which will format the code using black formatting (https://github.com/psf/black) before pushing any commits to github.

These can be set up by running the following command inside the `online-surrogate-model-dev` conda environment:
`pre-commit install`
