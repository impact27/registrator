# -*- coding: utf-8 -*-
"""
Created on Thu May 19 11:32:34 2016

@author: quentinpeter

profile register_channel

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

import matplotlib.image as mpimg
import registration.channel as cr

fns=['test_DATA/20160513/25.tif']
fns.append('test_DATA/20160513/10_1.tif')

imgs=[mpimg.imread(fn) for fn in fns]

im0=imgs[0][:,:512]
im1=imgs[1][:,:512]
#%%
for i in range(100):
    angle, scale, origin, im2=cr.register_channel(im0,im1)
#%%