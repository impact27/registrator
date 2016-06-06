# -*- coding: utf-8 -*-
"""
Created on Thu May 19 11:32:34 2016

@author: quentinpeter
"""

import matplotlib.image as mpimg
import registration.channel as cr

fns=['20160513/25.tif']
fns.append('20160513/10_1.tif')

imgs=[mpimg.imread(fn) for fn in fns]

im0=imgs[0][:,:512]
im1=imgs[1][:,:512]
#%%
for i in range(100):
    angle, scale, origin, im2=cr.register_channel(im0,im1)
#%%