# -*- coding: utf-8 -*-
import sys
sys.path.append('chreg')
from matplotlib.pyplot import figure, plot, imshow, show,close,semilogy, hold
import matplotlib.image as mpimg
import numpy as np
import math
import registration.image as ir
import registration.channel as cr
from scipy.ndimage.interpolation import zoom,rotate
import importlib
from numpy.fft import rfft2, fftshift,fft2

#%% nload images
fns=['UVData/im0.tif']
fns.append('UVData/im1.tif')
fns.append('UVData/ba_e1105qt5_500ms.tif')
fns.append('UVData/ba_e1105qt5bg2_1000ms.tif')
fns.append('UVData/ba_e1105qt5bg3_1000ms.tif')
imgs=[mpimg.imread(fn) for fn in fns]

#%%
im0=imgs[2][:512,:512]
im1=ir.rotate_scale(im0-im0.mean(),np.pi/3,1.6)
im0=np.float32(im0)
im1=np.float32(im1)
#%%
importlib.reload(ir)
for i in range(1000):
    ir.register_images(im0,im1)



#%%
"""
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
a=lp1.mean(1)
semilogy(np.r_[a[1283:],a[:1283]])
semilogy(lp0.mean(1))
#%%
import matplotlib.pyplot as plt
import cv2
img = cv2.medianBlur(ir.cv2prep(im0,dtype='uint8'),11)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY_INV,101,
                            np.uint8(np.std(img)))
kernel=np.ones((5,5))
#th3= cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel)
close(5)
close(6)
figure(5)
plt.hist(img[img>0], 256)
figure(6)
imshow(th3)
figure(7)
imshow(img)
imshow(th3, alpha=0.5)
figure(8)
imshow(im0)
#"""
#%%
X=np.asarray([-1,0,1])
Y=np.asarray([1,4,2])
print(ir.gauss_fit_log(X,Y))