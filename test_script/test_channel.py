# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:19:57 2016

@author: quentinpeter

Run this file to test the channel registration

Copyright (C) 2016  Quentin Peter

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

#%% load libraries
from matplotlib.pyplot import figure, plot, imshow, show,close,semilogy, hold
import matplotlib.image as mpimg
import numpy as np
import math
import importlib
#Load local libraries
import image_registration.image as ir
import image_registration.channel as cr

#%%Reload them in case they changed
importlib.reload(ir)
importlib.reload(cr)

#If you want the plots in external figues
#%matplotlib osx

#%% nload images
fns=['test_DATA/20160513/25.tif']
fns.append('test_DATA/20160513/10_1.tif')
fns.append('test_DATA/20160513/10_2.tif')
fns.append('test_DATA/20160513/10_3.tif')
fns.append('test_DATA/channel_opening.png')
imgs=[mpimg.imread(fn) for fn in fns]

#%%
#defines an useful plot
def plotreg(im0,im2,origin):
    imshow(im0)
    hold(True)
    imshow(cr.edge(im2),extent=ir.get_extent(origin, im2.shape),alpha=0.5)
    show()

#%% Test Channel direction and width
#Choose image
im0=imgs[1]
#get width and angle
width0,an0=cr.channel_width(im0, chanapproxangle=-np.pi/2)
#plot figure
figure()
imshow(im0)
#add direction
center=np.array(im0.shape)//2
chdir=[math.sin(an0),
       math.cos(an0)]
x=(np.array([0,chdir[1]]))*center[0]+center[1]
y=(np.array([0,chdir[0]]))*center[0]+center[0]
plot(x,y)
#add width
x=np.array([0, chdir[0]])*width0+center[1]
y=np.array([0,-chdir[1]])*width0+center[0]
plot(x,y)

#%% Detect an offset using cross correlation
#Use the edge function to extract the edges
im0=imgs[2]
im1=imgs[3]
e0=cr.edge(im0)
e1=cr.edge(im1)
origin=ir.find_shift_cc(e0,e1)

#plot The result
figure(2)
plotreg(im0,im1,origin)

#%% test rotation and offset
im0=imgs[1]
im1=imgs[2]
angle, scale, origin, im2=cr.register_channel(im0,im1, chanapproxangle=-np.pi/2)
figure(3)
plotreg(im0,im2,origin)

#%% Idem with image registration
importlib.reload(ir)
angle, scale, origin, im2=ir.register_images(cr.edge(im0),cr.edge(im1))
figure(4)
plotreg(im0,im2,origin)
#%% detect scale, rotation and offset
im0=imgs[0]
im1=imgs[1]
angle, scale, origin, im2=cr.register_channel(im0,im1, chanapproxangle=-np.pi/2)
figure(5)
plotreg(im0,im2,origin)
#%% Shows the limits of image registration
angle, scale, origin, im2=ir.register_images(cr.edge(im0),
                                             cr.edge(im1))
figure(6)
plotreg(im0,im2,origin)

#%% Match to image
importlib.reload(cr)
importlib.reload(ir)
im1=imgs[3]
im0=imgs[4][:,:,1]

angle, scale, origin, im2=cr.register_channel(im0,im1, chanapproxangle=-np.pi/2)

figure(7)
plotreg(im0,im2,origin)
