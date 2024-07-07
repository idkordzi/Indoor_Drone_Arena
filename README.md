# Indoor Drone Arena

Python application that allows to calculate 3D position of a marker (e.g. a diode)
in a confined space based on camera images.

In the project, images are recorded using [Basler cameras](https://www.baslerweb.com/en/shop/aca1440-220um/) and later processed on [Jetson Nano](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-2gb-devkit) boards. Next, the obtained data is send to main computer (e.g. PC) where the final calculations are performed.

At the moment, it is possible to estimate position for only one marker at the time as the app is not prepared to match corresponding points from different images.

## Preparation

The app is fully written in Python. It is recommended to use [version 3.8](https://www.python.org/downloads/release/python-380/) or newer. All necessary packages can be installed using *requirements.txt* file:

> python3 -m pip install -r requirements.txt

To be able to communicate with Basler camera please install [pylon software](https://www.baslerweb.com/en/software/pylon/).

## Quick start

**1. Run camera devices in `full` mode**

Python class, that handles reading from camera and perform detection, offers several work modes. For the next step it will be necessary to send both frames and points coordinates to the main computer. On the camera device run:

> python run_device.py -d 3

**2. Perform system calibration**

In the project a square pattern shown in [[1]](#credits) was used to calculate projection matrixes. On the main computer run:

> python run_server.py -m 4

It will enable calibration work mode. Note that by default the pattern marks the beginning of the coordinate system used for calculations.

It is also possible to get the projection matrixes separately for each camera by running on each camera device:

> python run_device.py -c 3

For the second option remember to copy calculated matrixes to main computer as they will be used in the later step.

**3. Run camera devices in *detection* mode (optional)**

After calibration camera devices can be switched to `detection` mode to reduce workoad (only points coordinates will be sent). To do this terminate currently running program and run:

> python run_device.py -d 2

**4. Start estimating markers positions**

For the last step run on the main computer:

> python run_server.py -m 2

or to display cameras view as well (it requires camera devices to be run in `full` mode):

> python run_server.py -m 3

## Repository overview

- `arduino` contains arduino code for external camera triggering. It is used to synchronise the cameras
- `configs` contains general configuration file for the application
- `ip_config` contains helper bash scripts to change IP properities on Jetson Nano board
- `mocap` is the main library directory in which all source code is located
- `run_device.py` is a python script that handles running camera device in various work modes
- `run_server.py` is a python script that handles running main computer in various work modes

## Credits

1. Uematsu, Y., Teshima, T., Saito, H., Honghua, C., "D-Calib: Calibration Software for Multiple Cameras System", 2007, 14th International Conference on Image Analysis and Processing, DOI: 10.1109/iciap.2007.4362793

