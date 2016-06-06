# -*- coding: utf-8 -*-
"""
Created on Wed May 18 16:53:44 2016

@author: quentinpeter
"""
from numpy.fft import irfft
from scipy.ndimage.filters import gaussian_filter
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
        
    #get centered magnitude squared
    im = reg.centered_mag_sq_ccs(im)
    
    #if the channel direction is not given, deduce it from channel_angle
    if chanangle is None:
        chanangle = channel_angle(im,isshiftdftedge=True) 
    
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
    idx=np.float32(idx)
    f=cv2.remap(np.float32(im),idx[1,:],idx[0,:],cv2.INTER_LINEAR)
    f=np.squeeze(f)
    #The central line of the fft will have a periodic feature for parallel
    #lines which we can detect with fft
    f=abs(irfft(f**2))
    #filter to avoid "interferences"
    f=gaussian_filter(f,1)
    #the offset is determined by the first pixel below mean
    wmin=np.nonzero(f-f.mean()<0)[0][0]
    
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(f,'x')
    plt.plot([wmin,wmin],[0,f.max()])
    plt.plot([0,500],[f.mean()+3*f.std(),f.mean()+3*f.std()])
    #"""
    
     #find max excluding the first few points
    ret=reg.get_peak_pos(f[wmin:f.size//2])
    
    #return max and corresponding angle
    return (wmin+ret), chanangle
    
def channel_angle(im,isshiftdftedge=False):
    """Extract the channel angle from the rfft"""
    #Compute edge
    if not isshiftdftedge:
        im=edge(im)
    #compute log fft
    lp, anglestep=reg.polar_fft(im,isshiftdft=isshiftdftedge)  
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(lp.sum(-1),'x')
    plt.figure()
    plt.plot(np.log(lp).sum(-1),'x')
    #"""
    
    #get peak pos
    ret=reg.get_peak_pos(lp.sum(-1),wrap=True)
    #return max-pi/2
    return reg.clamp_angle(ret*anglestep-np.pi/2)
    
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
    
    #compute the angle and channel width of biggest angular feature
    w0,a0=channel_width(fe0,isccsedge=True)
    w1,a1=channel_width(fe1,isccsedge=True)
    
    #get angle diff
    angle=reg.clamp_angle(a0-a1)
    if ch0angle is not None:
        a0=ch0angle
        a1=a0-angle 
        
    #if the scale is unknown, ratio of the channels
    if scale is None:
        scale=w1/w0
    #scale and rotate
    e2=reg.rotate_scale(e1,angle,scale)
    #get edge from scaled and rotated im1
    fe2=reg.dft_optsize(np.float32(e2),shape=fe0.shape)
    #find offset
    y,x=reg.find_shift_dft(fe0,fe2, isccs=True)
    #return all infos
    return angle, scale, [y, x], e2
    
def uint8sc(im):
    """Scale the image to uint8
    """
    immin=im.min()
    immax=im.max()
    imrange=immax-immin
    return cv2.convertScaleAbs(im-immin, alpha=255/imrange)