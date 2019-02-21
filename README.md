# Distance
Finding the distance from the camera using Python and OpenCV.

## Usage:
To run the program on default settings:
```
./distance.py
```
You can change the focal length to sensor height ratio and dot's height using optional arguments:
```
./distance.py -fd 1.6 -dh 30
```

## Approximating fd ratio:
Place your camera in front of a wall. Measure the distance between the camera and the wall (that's our X) and the distance between the camera's height and the heighest point the camera can see (that's our Y). 

![fdratio](/docs/fd.png)

Calculate the fd ratio using this equation:

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{f}{d}&space;=&space;\frac{X}{2Y}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{f}{d}&space;=&space;\frac{X}{2Y}" title="\frac{f}{d} = \frac{X}{2Y}" /></a>

## Prerequisites

* Python 3
* OpenCV