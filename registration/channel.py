# -*- coding: utf-8 -*-
"""
Created on Wed May 18 16:53:44 2016

@author: quentinpeter
"""
from numpy.fft import rfft2, fftshift,irfft
from scipy.ndimage.interpolation import map_coordinates
import math
import numpy as np
from . import image as reg
import cv2

def channel_width(im,chanangle=None,isccsedge=False):
    """Get an estimation of the channel width. 

    This function assumes two parallel lines along angle chanangle.
    The perpendicular line in the fourrier plane will have a mark of this,
    under the form of an oscillation at some frequency corresponding
    to the distance between the two parallel lines.
    This can be extracted by another fft.  
    
    This second fft might have large components at low frequency, 
    So the first few frequencies are neglected.
    The threshold is the first position below mean
    
    If the chanangle is not specified, the direction with higher 
    contribution will be picked
    
    If the edge and rfft have already been computed, set isrfftedge to True
    """
    #Compute the fft if it is not already done
    if not isccsedge:
        im= reg.dft_optsize(np.float32(edge(im)))
    #if the channel direction is not given, deduce it from channel_angle
    if chanangle is None:
        chanangle = channel_angle(im,isccsedge=True) 
    #get centered magnitude squared
    im = reg.centered_mag_sq_ccs(im)
    #get vector perpendicular to angle
    fdir=np.asarray([math.cos(chanangle),-math.sin(chanangle)])#y,x = 0,1
    #need to be in the RHS of the cadran for rfft
    if fdir[1]<0:
        fdir*=-1    
    #get center of shifted fft
    center=np.asarray([im.shape[0]//2,0])
    #get size
    shape=np.asarray([im.shape[0]//2,im.shape[1]])
    #get evenly spaced positions between 0 and 1 (not included) 
    pos=np.r_[:1:(shape.min()+1)*1j][:-1]
    #get index of a line of length 1 in normalized units from center 
    #in direction of chdir
    idx=((fdir*shape)[:,np.newaxis].dot(pos[np.newaxis])
         +center[:,np.newaxis])
    #get the line
    f=map_coordinates(im,idx)
    #The central line of the fft will have a periodic feature for parallel
    #lines which we can detect with fft
    f=abs(irfft(f**2))
    #the offset is determined by the first pixel below mean
    wmin=np.nonzero(f-f.mean()<0)[0][0]
    #find max excluding the first few points
    return (wmin+f[wmin:f.size//2].argmax())
    
def channel_angle(im,isccsedge=False):
    """Extract the channel angle from the rfft"""
    #Compute edge
    if not isccsedge:
        im=edge(im)
    #compute log fft
    lp, anglestep=reg.polar_fft(im,isccs=isccsedge)  
    
    #return max-pi/2
    return reg.clamp_angle(lp.sum(-1).argmax()*anglestep-np.pi/2)
    
def edge(im):
    """Extract the edges of an image
    
    This scale the image to be used with Canny from OpenCV    
    """
    #map the 16bit image into 8bit   
    e0=cv2.Canny(uint8sc(im),100,200)
    return e0
    
def register_channel(im0,im1,scale=None,ch0angle=None):
    """Register the images assuming they are channels
    
    If the scale difference is known pass it in scale 
    
    If the channel angle is known pass it in changle
    """
    #extract the channels edges
    e0=edge(im0)
    e1=edge(im1)
    fe0, fe1=reg.dft_optsize_same(np.float32(e0),np.float32(e1))
    
    #get angle from biggest angular feature
    a0=channel_angle(fe0,isccsedge=True)
    a1=channel_angle(fe1,isccsedge=True)
    angle=reg.clamp_angle(a0-a1)
    if ch0angle is not None:
        a0=ch0angle
        a1=a0-angle 
    #if the scale is unknown, assume channels in y direction (axis 0)
    if scale is None:
        #as we work with channels, try to deduce from width
        w0=channel_width(fe0,chanangle=a0,isccsedge=True)
        w1=channel_width(fe1,chanangle=a1,isccsedge=True)
        scale=w1/w0
    #scale and rotate
    im2=reg.rotate_scale(im1,angle,scale)
    #get edge from scaled and rotated im1
    #TODO: Remove the edges in e2 (use cv2.BORDER_REPLICATE?)
    e2=edge(im2)
    #find offset
    y,x=reg.find_shift_cc(e0,e2)
    #return all infos
    return angle, scale, [y, x], im2
    
    
def uint8sc(im):
    """scale the image to fit in a uint8"""
    im=im-im.min()
    im=im/(im.max()/255)
    return np.uint8(im)