from unittest import TestCase

import registrator.image as ir
import registrator.channel as cr
from tifffile import imread
import numpy as np
from . import __file__
import os
import matplotlib.image as mpimg
folder = os.path.dirname(__file__)


class TestImage(TestCase):
    def test_registration_UV(self):
        im0 = imread(folder + '/test_data/imUV.tif')
        im1 = ir.rotate_scale(im0 - im0.mean(), np.pi / 3, 1.6)

        angle, scale, origin, im2 = ir.register_images(im0, im1)
        self.assertTrue(np.abs(angle + np.pi / 3) < 1e-3)
        self.assertTrue(np.abs(scale * 1.6 - 1) < 1e-3)

    def test_channel_width(self):
        im0 = imread(folder + '/test_data/imch1.tif')
        width0, an0 = cr.channel_width(im0, chanapproxangle=-np.pi / 2)
        self.assertTrue(np.abs(width0 - 73.4) < 1)
        self.assertTrue(np.abs(an0 + 1.478) < 1e-2)

    def test_channel_cc(self):
        im0 = imread(folder + '/test_data/imch2.tif')
        im1 = imread(folder + '/test_data/imch3.tif')
        e0 = cr.edge(im0)
        e1 = cr.edge(im1)
        origin = ir.find_shift_cc(e0, e1)

        self.assertTrue(np.max(np.abs(origin - [110, -34])) < 2)

    def test_channel_registration(self):
        im0 = imread(folder + '/test_data/imch1.tif')
        im1 = imread(folder + '/test_data/imch2.tif')
        angle, scale, origin, im2 = cr.register_channel(im0, im1,
                                                        chanapproxangle=-np.pi / 2)
        self.assertTrue(np.abs(angle - 0.15) < 1e-2)
        self.assertTrue(np.abs(scale - 1) < 1e-2)
        self.assertTrue(np.max(np.abs(origin - np.array([-84, 68]))) < 2)

    def test_channel_edge_registration(self):
        im0 = imread(folder + '/test_data/imch1.tif')
        im1 = imread(folder + '/test_data/imch2.tif')
        angle, scale, origin, im2 = ir.register_images(
            cr.edge(im0), cr.edge(im1))
        self.assertTrue(np.abs(angle - 0.15) < 1e-2)
        self.assertTrue(np.abs(scale - 1) < 1e-2)
        self.assertTrue(np.max(np.abs(origin - np.array([-84, 68]))) < 2)

    def test_channel_registration_scale(self):
        im0 = imread(folder + '/test_data/imch0.tif')
        im1 = imread(folder + '/test_data/imch1.tif')
        angle, scale, origin, im2 = cr.register_channel(im0, im1,
                                                        chanapproxangle=-np.pi / 2)

        self.assertTrue(np.abs(angle + 0.124) < 1e-2)
        self.assertTrue(np.abs(scale / 4 - 1) < 1e-1)
        self.assertTrue(np.max(np.abs(origin - np.array([136, -16]))) < 2)


#     def test_channel_edge_registration_scale(self):
#         im0 = imread(folder+'/test_data/imch0.tif')
#         im1 = imread(folder+'/test_data/imch1.tif')
#         angle, scale, origin, im2=ir.register_images(cr.edge(im0),
#                                                      cr.edge(im1))
#         self.assertTrue(np.abs(angle+0.124) < 1e-2)
#         self.assertTrue(np.abs(scale/4-1) < 1e-1)#Fails
#         self.assertTrue(np.max(np.abs(origin - np.array([136, -16])))<2)

    def test_photo(self):
        photo = mpimg.imread(folder + '/test_data/IMG.jpg').sum(-1)
        part = ir.rotate_scale(photo, np.pi / 3, 1.2)
        angle, scale, origin, im2 = ir.register_images(photo, part)
        self.assertTrue(np.abs(angle + np.pi / 3) < 1e-2)
        self.assertTrue(np.abs(scale * 1.2 - 1) < 1e-2)
        self.assertTrue(np.max(np.abs(origin)) < 2)
