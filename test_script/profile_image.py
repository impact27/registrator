# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 14:05:10 2016

@author: quentinpeter

profile register_image

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
import sys
sys.path.append('chreg')
import matplotlib.image as mpimg
import numpy as np
import image_registration.image as ir
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

