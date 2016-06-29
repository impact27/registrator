# -*- coding: utf-8 -*-
from matplotlib.pyplot import figure, plot, imshow, show,close,semilogy, hold
import matplotlib.image as mpimg
import numpy as np
import image_registration.image as ir
import image_registration.channel as cr
import importlib

#%% nload images
fns=['test_DATA/UVData/im0.tif']
fns.append('test_DATA/UVData/im1.tif')
fns.append('test_DATA/UVData/ba_e1105qt5_500ms.tif')
fns.append('test_DATA/UVData/ba_e1105qt5bg2_1000ms.tif')
fns.append('test_DATA/UVData/ba_e1105qt5bg3_1000ms.tif')
imgs=[mpimg.imread(fn) for fn in fns]

#%%
im0=imgs[2]#[:512,:512]
im1=ir.rotate_scale(im0-im0.mean(),np.pi/3,1.6)
#%%
importlib.reload(ir)

figure(0)
imshow(im0)
figure(1)
imshow(im1) 

#%%

importlib.reload(ir)
angle, scale, origin, im2 =ir.register_images(im0,im1)

t0=im0>im0.mean()
t1=im1>im1.mean()
t2=im2>im2.mean()
print(angle+np.pi/3)
print(1/scale-1.6)

#%%
figure(2)
imshow(t0,alpha=0.5,cmap='Greens')
imshow(t2,extent=ir.get_extent(origin, t2.shape),alpha=0.5,cmap='Reds')

#%%

lp0, anglestep, log_base=ir.polar_fft(im0, islogr=True)
lp1, anglestep, log_base=ir.polar_fft(im1, islogr=True,
                                      anglestep=anglestep,
                                      radiimax=lp0.shape[1])
close(3)
figure(3)
semilogy(lp0.mean(0))
semilogy(lp1.mean(0))
figure(4)
semilogy(lp1.mean(1))
semilogy(lp0.mean(1))
