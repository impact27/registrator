# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:20:35 2016

@author: quentinpeter
"""
from matplotlib.pyplot import figure, plot, imshow, show,close,semilogy, hold
import matplotlib.image as mpimg
import numpy
import math
import importlib
#Load local libraries
import registration.image as ir
import registration.channel as cr
#%%
#defines an useful plot
def plotreg(im0,im2,origin):
    imshow(im0)
    hold(True)
    imshow(cr.edge(im2),extent=ir.get_extent(origin, im2.shape),alpha=0.5)
    show()
#%% test with actual image
photo=mpimg.imread('IMG.jpg')
photo=photo.sum(-1)
#%%
importlib.reload(ir)
part=ir.rotate_scale(numpy.float32(photo),numpy.pi/3,1.2)
#%%
importlib.reload(ir)
angle, scale, origin, im2=ir.register_images(numpy.float32(photo),
                                             numpy.float32(part))
#%%
#"""
figure()
plotreg(photo,im2, origin)

#%%

importlib.reload(ir)
im0=photo
im1=part
lp0, anglestep, log_base=ir.polar_fft(im0, islogr=True)
lp1, anglestep, log_base=ir.polar_fft(im1, islogr=True,
                                      anglestep=anglestep, radiimax=lp0.shape[1])
lp0=numpy.log(lp0)
lp1=numpy.log(lp1)
close(0)
close(1)
figure(0)
plot(lp0.mean(0))
plot(lp1.mean(0))
figure(1)
plot(lp0.mean(1))
plot(lp1.mean(1))

ir.shift_fft(lp0,lp1)
#"""