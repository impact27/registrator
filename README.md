## Synopsis

This python3 code does image registration. The scale, rotation and position between two images is extracted 

## Code Example
load the module

: import registrator.channel as cr

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

In the folder:

pip install .

## API Reference

The python help function gives the required infos as docstrings have been specified.

: help(cr)


## Tests

nosetests

## Contributors

If you want to improve this code feel free to talk to me or send push requests

## License

This code is under GPLv3
