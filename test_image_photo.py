# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:20:35 2016

@author: quentinpeter

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
from matplotlib.pyplot import figure, plot, imshow, show,close,semilogy, hold
import matplotlib.image as mpimg
import numpy as np
import importlib
#Load local libraries
import image_registration.image as ir
import image_registration.channel as cr
#%%
#defines an useful plot
def plotreg(im0,im2,origin):
    imshow(im0)
    hold(True)
    imshow(cr.edge(im2),extent=ir.get_extent(origin, im2.shape),alpha=0.5)
    show()
#%% test with actual image
photo=mpimg.imread('test_DATA/IMG.jpg')
photo=photo.sum(-1)
#%%
importlib.reload(ir)
part=ir.rotate_scale(photo,np.pi/3,1.2)
#%%
importlib.reload(ir)
angle, scale, origin, im2=ir.register_images(photo,
                                             part)
#%%
#"""
figure(2)
plotreg(photo,im2, origin)

#%%

importlib.reload(ir)
im0=photo
im1=part
lp0, log_base=ir.polar_fft(im0, islogr=True)
lp1, log_base=ir.polar_fft(im1, islogr=True,
                           nangle=lp0.shape[1], radiimax=lp0.shape[1])
lp0=np.log(lp0)
lp1=np.log(lp1)
close(0)
close(1)
figure(0)
plot(lp0.mean(0))
plot(lp1.mean(0))
figure(1)
plot(lp0.mean(1))
plot(lp1.mean(1))

ir.find_shift_dft(lp0,lp1)
#"""