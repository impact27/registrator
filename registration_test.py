# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:19:57 2016

@author: quentinpeter

Run this file to test the channel registration
"""

#%% load libraries
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage.interpolation import zoom,rotate
import numpy
import math
import importlib
#Load local libraries
import registration as ir
import channel as cr

#%%Reload them in case they changed
importlib.reload(ir)
importlib.reload(cr)

#If you want the plots in external figues
#%matplotlib osx

#%% nload images
fns=['20160513/25.tif']
fns.append('20160513/10_1.tif')
fns.append('20160513/10_2.tif')
fns.append('20160513/10_3.tif')
fns.append('channel_opening.png')
imgs=[mpimg.imread(fn) for fn in fns]

#defines an useful plot
def plotreg(im0,im2,origin):
    plt.imshow(im0)
    plt.hold(True)
    plt.imshow(cr.edge(im2),extent=ir.get_extent(origin, im2.shape),alpha=0.5)
    plt.show()





#%% Test Channel direction and width
#Choose image
im0=imgs[1]
#get width and angle
an0=cr.channel_angle(im0)
width0=cr.channel_width(im0,chanangle=an0)
#plot figure
plt.figure(1)
plt.imshow(im0)
#add direction
center=numpy.array(im0.shape)//2
chdir=[math.sin(an0),
       math.cos(an0)]
x=(numpy.array([0,chdir[1]])+1)*center[1]
y=(numpy.array([0,chdir[0]])+1)*center[0]
plt.plot(x,y)
#add width
x=numpy.array([0, chdir[0]])*width0+center[1]
y=numpy.array([0,-chdir[1]])*width0+center[0]
plt.plot(x,y)

#%% Detect an offset using cross correlation
#Use the edge function to extract the edges
im0=imgs[2]
im1=imgs[3]
e0=cr.edge(im0)
e1=cr.edge(im1)
origin=ir.cross_correlation_shift(e0,e1)

#plot The result
plt.figure(2)
plotreg(im0,im1,origin)

#%% test rotation and offset
im0=imgs[1]
im1=imgs[2]
angle, scale, origin, im2=cr.register_channel(im0,im1)
plt.figure(3)
plotreg(im0,im2,origin)

#%% detect scale, rotation and offset
im0=imgs[0]
im1=imgs[1]

angle, scale, origin, im2=cr.register_channel(im0,im1)
plt.figure(4)
plotreg(im0,im2,origin)

#%% Match to image
importlib.reload(ir)
im1=imgs[3]
im0=imgs[4][:,:,1]

angle, scale, origin, im2=cr.register_channel(im0,im1)

plt.figure(5)
plotreg(im0,im2,origin)

#%% test with actual image
#This doesn't work... -> add filtering?
photo=mpimg.imread('IMG.jpg')
photo=photo.sum(-1)
part=zoom(rotate(photo,0),0.9)
photo2=photo[:int(.9*photo.shape[0])+1,:int(.9*photo.shape[1])]
angle, scale, origin, im2=ir.register_images(numpy.float32(photo2),numpy.float32(part))
#%%
plt.figure(6)
plotreg(photo2,part, origin)
