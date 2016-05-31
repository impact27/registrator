# -*- coding: utf-8 -*-
"""
Created on Tue May 10 17:15:48 2016

@author: quentinpeter

This module does image registration. 


based on:
    
An FFT-Based Technique for Translation, Rotation, and Scale-Invariant
 Image Registration
B. Srinivasa Reddy and B. N. Chatterji


"""
#import libraries
from numpy.fft import irfft2,rfft2, fftshift
import numpy
import math
import cv2    
    
    
    
def register_images(im0,im1):
    """Finds the rotation, scaling and translation of im1 relative to im0
    
    TODO: use openCV dft to get more speed?
    """
    #Get rotation and scale
    angle, scale = find_rotation_scale(im0,im1)
    #apply rotation and scale
    im2=rotate_scale(im1,angle,scale)
    #Find offset
    #y,x=cross_correlation_shift(im0,im2)
    y,x=shift_fft(im0,im2)
    return angle, scale, [y, x], im2
    
def rotate_scale(im,angle,scale):
    """Rotates and scales the image"""
    rows,cols = im.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-angle*180/numpy.pi,1/scale)
    im = cv2.warpAffine(im,M,(cols,rows),
                        borderMode=cv2.BORDER_CONSTANT,
                        flags=cv2.INTER_CUBIC)#REPLICATE
    return im
    
def cross_correlation_shift(im0,im1,ylim=None,xlim=None,
                            xpadmode='constant', ypadmode='constant'):
    """Finds the best translation fit between im0 and im1 (top corner)
    
    The origin of im1 in the im0 referential is returned
    
    ylim and xlim limit the possible output.
    """
    #Remove mean
    im0=im0-im0.mean()
    im1=im1-im1.mean()
    #Save shapes as numpy array
    shape0=numpy.array(im0.shape)
    shape1=numpy.array(im1.shape)
    
    #Compute the offset and the pad (yleft,yright,xtop,xbottom)
    offset=1-shape1
    pad=numpy.lib.pad(-offset,(1,1),mode='edge')
    
    if ylim is not None or xlim is not None:
        #apply limit on padding
        if ylim is not None:
            pad[0]=-ylim[0]
            pad[1]=ylim[1]+(shape1-shape0)[0]
        if xlim is not None:
            pad[2]=-xlim[0]
            pad[3]=xlim[1]+(shape1-shape0)[1]
        #extract new offset from padding 
        offset=-pad[::2]
        #if the padding is negatif, cut the matrix
        cut=pad*(pad<0)
        #the left/top components should be positive
        cut[::2]*=-1
        #The right/bottom components can't be 0, replace by shape0
        cut[1::2]+=(cut[1::2]==0)*shape0
        #cut the matrix
        im0=im0[cut[0]:cut[1],cut[2]:cut[3]]
        #extract positive padding
        pad=pad*(pad>0)
        
    #separate pad for application on matrix
    ypad=(pad[0],pad[1])
    xpad=(pad[2],pad[3])
    #prepare matrix
    im0=cv2prep(im0)
    im0=numpy.lib.pad(im0,((0,0),xpad),mode=xpadmode)
    im0=numpy.lib.pad(im0,(ypad,(0,0)),mode=ypadmode)
    #compute Cross correlation matrix
    xc=cv2.matchTemplate(numpy.float32(im0),numpy.float32(im1),cv2.TM_CCORR)
    #Find maximum of abs (can be anticorrelated)
    idx=numpy.array(numpy.unravel_index(numpy.argmax(xc),xc.shape))        
    #Return origin in im0 units
    return idx+offset
    
def clamp(a):
    """return a between -pi/2 and pi/2 (in fourrier plane, +pi is the same)"""
    return (a+numpy.pi/2)%numpy.pi - numpy.pi/2
    
def shift_fft(im0,im1):
    """The "official" shift method"""
    #Need same shape
    assert im0.shape == im1.shape
    shape=numpy.array(im0.shape)
    #compute fft
    f0=rfft2(im0)
    f1=rfft2(im1)
    #compute matrix
    xc=irfft2((f0*f1.conj())/(abs(f0)*abs(f1)),s=shape)
    #find max
    idx=numpy.array(numpy.unravel_index(numpy.argmax(xc),xc.shape)) 
    #restrics to reasonable values
    idx[idx>shape//2]-=shape[idx>shape//2]
    return idx
    
    
def find_rotation_scale(im0,im1,isrfft=False):
    """Compares the images and return the best guess for the rotation angle,
    and scale difference
    
    Scale precision relative to first image.
    
    The angle and scale search can be limited by alim and slim respectively
    -pi/2< alim < pi/2
    """
    #Get log polar coordinates. choose the log base 
    lp1, anglestep, log_base=polar_fft(im1, islogr=True,isrfft=isrfft)
    lp0, anglestep, log_base=polar_fft(im0,
                                           radiimax=lp1.shape[1], 
                                           anglestep=anglestep,
                                           islogr=True,
                                           isrfft=isrfft)
    angle, scale = shift_fft(numpy.log(lp0),numpy.log(lp1))     
    #get angle in correct units
    angle*=anglestep
    #get scale in linear units
    scale=log_base ** (scale)
    #return angle and scale
    return angle, scale
    
    
    #old method
    """
    ,alim=None,slim=None):
    if alim is None:
        alim=[-numpy.pi/2,numpy.pi/2]
    #transform the limit in pixel size
    alim= numpy.int64((numpy.array(alim))/anglestep)
    if slim is not None:
        slim=numpy.int64(numpy.log(slim)/numpy.log(log_base))
    #compute the cross correlattion to extract the angle and scale     
    angle,scale= cross_correlation_shift(numpy.log(lp0),numpy.log(lp1),
                                         ylim=alim,xlim=slim,
                                         ypadmode='wrap')
    
    """
    

def get_extent(origin, shape):
    """Computes the extent for imshow() (see matplotlib doc)"""
    return [origin[1],origin[1]+shape[1],origin[0]+shape[0],origin[0]]
    
def polar_fft(image, isrfft=False, anglestep=None, radiimax=None,
              islogr=False):
    """Return fft in polar (or log-polar) units
    
    if the image is already fft2 and fftshift, set isrfft to True
    
    anglestep set the precision on the angle.
    If it is not specified, a value is deduced from the image size
    
    radiimax is the maximal radius (log of radius if islogr is true).
    if not provided, it is deduced from the image size
    
    To get log-polar, set islogr to True
    log_base is the base of the log. It is deduced from radiimax.
    Two images that will be compared should therefore have the same radiimax.
    """
    #get recentered fft if not already done
    if not isrfft:
        image=abs(fftshift(rfft2(image),axes=(0,)))
    
    #the center is shifted from 0,0 to the ~ center 
    #(eg. if 4x4 image, the center is at [2,2], as if 5x5)
    qshape = numpy.array([image.shape[0]//2,image.shape[1]])
    center = numpy.array([qshape[0],0]) 
    
    #if the angle Step is not given, take the number of pixel
    #on the perimeter as the target #=range/step
    if anglestep is None:
        anglestep=1/qshape.min()
    
    #get the theta range
    thetalin=numpy.arange(-numpy.pi/2,numpy.pi/2,anglestep,dtype=numpy.float32)
    nbangles=thetalin.size
    
    #For the radii, the units are comparable if the log_base and radiimax are
    #the same. Therefore, log_base is deduced from radiimax
    #The step is assumed to be 1
    if radiimax is None:
        radiimax = qshape.min()
        
    #fill theta matrix
    theta = numpy.empty((nbangles, radiimax), dtype=numpy.float32)
    theta.T[:] = thetalin
    
    #fill radius matrix
    radius = numpy.empty_like(theta)
    
    #also as the circle is an ellipse in the image, 
    #we want the radius to be from 0 to 1
    if islogr:
        #The log base solves log_radii_max=log_{log_base}(linear_radii_max)
        #where we decided arbitrarely that linear_radii_max=log_radii_max
        log_base=math.exp(math.log(radiimax) / radiimax)
        radius[:] = ((log_base ** numpy.arange(0,radiimax,dtype=numpy.float32))
                    /radiimax)
    else:
        radius[:] = numpy.linspace(0,1,radiimax, endpoint=False,
                                    dtype=numpy.float32)
    
    #get x y coordinates matrix (The units are 0 to 1, therefore a circle is
    #represented as an ellipse)
    y = qshape[0]*radius * numpy.sin(theta) + center[0]
    x = qshape[1]*radius * numpy.cos(theta) + center[1]
    
    #get output
    output=cv2.remap(cv2prep(image),x,y,cv2.INTER_LINEAR)#LINEAR, CUBIC,LANCZOS4
    #output=map_coordinates(image, [y, x]) 
    if islogr:
        return output, anglestep, log_base
    else:
        return output, anglestep
def cv2prep(im,dtype=None):
    """Prepare the image to be used with opencv (8bit image)"""
    if dtype is 'uint8':
        im=im-im.min()
        im=im/(im.max()/255)
        return numpy.uint8(im)
    return im