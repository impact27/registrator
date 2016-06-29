## Synopsis

This python3 code does image registration for microfluidics devices. The scale, rotation and position between two images is extracted 

## Code Example
load the module

: import registration.channel as cr

Get a channel direction

: an0=cr.channel_angle(im0)

Get a channel width

: width0=cr.channel_width(im0,chanangle=an0)

Compare two images

: angle, scale, origin, im2=cr.register_channel(im0,im1)

## Motivation

Between series of images, the device mignt have moved, of the focal distance might have changed. 
This project will automatically detect these differences.

## Installation

After downloading, you can copy the image_registration folder in /usr/local/lib/python3.5/site-packages/

Or you can create a symbolic link:

ln -s $(pwd)/image_registration/ /usr/local/lib/python3.5/site-packages/


This project require python3, openCV 3 and the python openCV interface

On a mac with HomeBrew:

: brew install python3

: brew install opencv3 --with-python3

: ln -s /usr/local/opt/opencv3/lib/python3.5/site-packages/cv2.cpython-35m-darwin.so /usr/local/lib/python3.5/site-packages/cv2.so

## API Reference

The python help function gives the required infos as docstrings have been specified.

: help(cr)


## Tests

The file registration_test runs a series of tests

## Contributors

If you want to improve this code feel free to talk to me or send push requests

## License

This code is under GPLv3
