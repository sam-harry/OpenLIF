# OpenLIF
A piece of software built upon Python and OpenCV to enable easy and accurate 
LIF measurements. This work is released with the hope that it will save a
researcher somewhere a headache. It is shared under the CC BY 4.0 license,
see https://creativecommons.org/licenses/by/4.0/ for details. 

## Installation 

First clone this repository to a suitable location, then install python
dependancies with either anaconda or pip as below.

### Anaconda

Create a python environment using the tested packages in conda-spec.txt, e.g.

```
conda create --name olif --file conda-spec.txt
```

(note that this requires a linux OS).
Alternatively, create your own with numpy, scipy, matplotlib, and opencv
like

```
conda create --name olif -c conda-forge numpy scipy matplotlib x264 opencv
```

(note that the conda-forge channel provides a version ffmpeg that can encode
x264 and more recent package versions than the default channel).

### pip

Create a python environment e.g. using requirements.txt

```
python3 -m venv olif
source ./olif/bin/activate
pip3 install -r requirements.txt
```

Or create your own like

```
python3 -m venv olif
source ./olif/bin/activate
pip3 install numpy scipy matplotlib opencv-python
```

Note that your version of ffmpeg should include libx264 to write animations as
in the undular_bore_patching example if you use pip instead of conda.

## Usage

Then add openlif.py to the working directory or add the src directory to your 
python path, e.g.

```
sys.path.append('/home/user/OpenLIF/src/')
```

and import as usual

```
import openlif as olif
```

## Documentation

Code is written with PEP 8 in mind. Functions are sorted in the openlif.py
source code by purpose, each section is indicated by a comment starting with a
`#~~~~` and ending with `~~~~` with a desciptor in the middle, to make them
easier to find when scrolling.

Functions are written with PEP 3107 in mind.
Function documentation is avaialable from the code directly or through pydoc,
e.g. run the shell command `python3 -m pydoc openlif` from the OpenLIF/src
directory. Alternatively run `pydoc -b` to open and view in a web browser.

Additionally, an overview of the methods (and some details) discussing the
rationalle behind this code is included in doc/latex/main.pdf.
This document discusses some of the strengths, weaknesses, and alternatives for
implementing on another researchers data. It also contains the author's
opinion on some topics, guided by their own experience working with PLIF data,
and should be intepreted with such limitations in mind.
