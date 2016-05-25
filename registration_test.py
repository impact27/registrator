# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:19:57 2016

@author: quentinpeter
"""

#%% load libraries
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage.interpolation import zoom,rotate
import importlib
import registration as ir
import channel as cr
importlib.reload(ir)
importlib.reload(cr)
import numpy
import math
from numpy.fft import fft2, fftshift

#%matplotlib osx

#%% nload images
fns=['20160513/25.tif']
fns.append('20160513/10_1.tif')
fns.append('20160513/10_2.tif')
fns.append('20160513/10_3.tif')
fns.append('channel_opening.png')
imgs=[mpimg.imread(fn) for fn in fns]


def plotreg(im0,im2,origin):
    pass
    plt.imshow(im0)
    plt.hold(True)
    plt.imshow(cr.edge(im2),extent=ir.get_extent(origin, im2.shape),alpha=0.5)
    plt.show()

#%%
importlib.reload(cr)
a=cr.channel_width(imgs[1])
print(a)
#%% try to xcorr image 2 and 3
importlib.reload(ir)
e0=cr.edge(imgs[2])
e1=cr.edge(imgs[3])
origin=ir.cross_correlation_shift(e0,e1)

# As we can see in image 2 & 3, the luminus pattern is not a function
# of the position of the device, but of the light.
# Therefore, we take the edge

plt.figure(1)
plt.imshow(e0,alpha=0.5)
plt.hold(True)
plt.imshow(e1,extent=ir.get_extent(origin, e1.shape),alpha=0.5)
plt.show()

#%%
plt.figure(10)
plt.imshow(abs(fftshift(fft2(cr.edge(imgs[2])))))

#%% test rotation
#%matplotlib osx 
#inline
importlib.reload(cr)
importlib.reload(ir)
e0=imgs[1]
e1=imgs[2]
angle, scale, origin, im2=cr.register_channel(e0,e1)
plt.figure(2)
plotreg(e0,im2,origin)

#%% detect scale
importlib.reload(cr)
importlib.reload(ir)
im0=imgs[1]
im1=zoom(rotate(im0,20,mode='nearest'),1.4)
an0=cr.channel_angle(im0)

plt.figure(3)
plt.imshow(im0)
center=numpy.array(im0.shape)//2
plt.plot((numpy.array([0,math.cos(an0)])+1)*center[1],(numpy.array([0,math.sin(an0)])+1)*center[0])

#%%

angle, scale, origin, im2=cr.register_channel(im0,im1,scale=1.4)


plt.figure(4)
plotreg(im0,im2,origin)

#%% detect scale
importlib.reload(ir)
importlib.reload(cr)
im0=imgs[0]
im1=imgs[1]


an0=cr.channel_angle(im0)
plt.figure(5)
plt.imshow(im0)
center=numpy.array(im0.shape)//2
plt.plot((-numpy.array([0,math.cos(an0)])+1)*center[1],(-numpy.array([0,math.sin(an0)])+1)*center[0])

angle, scale, origin, im2=cr.register_channel(im0,im1)
plt.figure(6)
plotreg(im0,im2,origin)

#%% detect scale
importlib.reload(ir)
im1=imgs[3]
im0=imgs[4][:,:,1]

angle, scale, origin, im2=cr.register_channel(im0,im1)

plt.figure(7)
plotreg(im0,im2,origin)

#%% test with actual image
"""
photo=mpimg.imread('IMG.jpg')
photo=photo.sum(-1)
part=zoom(rotate(photo,0),1.3)
#part=part[part.shape[0]//4:3*part.shape[0]//4,part.shape[0]//4:2*part.shape[0]//4]
plt.figure(0)
plt.imshow(photo)
plt.figure(1)
plt.imshow(numpy.float32(part))
#%%
importlib.reload(ir)
angle, scale, origin, im2=ir.register_images(numpy.float32(photo),numpy.float32(part))
#%%
plt.figure(2)
plt.imshow(photo)
plt.hold(True)
plt.imshow(cr.edge(im2),extent=ir.get_extent(origin, im2.shape),alpha=0.5)
plt.show()

"""