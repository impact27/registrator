# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 14:05:10 2016

@author: quentinpeter

profile register_image
"""
import sys
sys.path.append('chreg')
import matplotlib.image as mpimg
import numpy as np
import registration.image as ir
import importlib
#%% nload images
fns=['test_DATA/UVData/im0.tif']
fns.append('test_DATA/UVData/im1.tif')
fns.append('test_DATA/UVData/ba_e1105qt5_500ms.tif')
fns.append('test_DATA/UVData/ba_e1105qt5bg2_1000ms.tif')
fns.append('test_DATA/UVData/ba_e1105qt5bg3_1000ms.tif')
imgs=[mpimg.imread(fn) for fn in fns]

#%%
im0=imgs[2][:512,:512]
im1=ir.rotate_scale(im0-im0.mean(),np.pi/3,1.6)
im0=np.float32(im0)
im1=np.float32(im1)
#%%
importlib.reload(ir)
#%%
for i in range(100):
    ir.register_images(im0,im1)

