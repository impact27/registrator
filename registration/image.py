# -*- coding: utf-8 -*-
"""
Created on Tue May 10 17:15:48 2016

@author: Quentin Peter

This module does image registration. 
    
The scale difference should be reasonable (2x)
    
If you crop the images to reasonable size, the algorithm is much faster:
eg:
512x512   : 0.06s
1024x1024 : 0.42s
4096x4096 : 9s

TODO:
    -Add np.asarray to every input that should be a numpy array
    
based on:
    
An FFT-Based Technique for Translation, Rotation, and Scale-Invariant
 Image Registration
B. Srinivasa Reddy and B. N. Chatterji

Works with Python 3.5

"""
#import libraries
import numpy as np
import cv2
from scipy.optimize import curve_fit # /!\ SLOW! avoid
from scipy.ndimage.measurements import label


######################
#High level functions#
######################


def register_images(im0,im1):
    """Finds the rotation, scaling and translation of im1 relative to im0
    
    The algorithm uses gaussian fit for subpixel precision.
    
    The best case would be to have two squares images of the same size.
    The algorithm is faster if the size is a power of 2.
    """
    #Compute DFT (THe images are resized to the same size)
    f0, f1=dft_optsize_same(im0,im1)
    #Get rotation and scale
    angle, scale = find_rotation_scale(f0,f1, isccs=True)
    #apply rotation and scale
    im2=rotate_scale(im1,angle,scale)
    f2=dft_optsize(im2,shape=f0.shape)
    #Find offset
    y,x=find_shift_dft(f0,f2, isccs=True)
    return angle, scale, [y, x], im2


def find_rotation_scale(im0,im1,isccs=False):
    """Compares the images and return the best guess for the rotation angle,
    and scale difference.
    
    If the images are already DFT and in the CCS format, set isccs to True.
    Otherwise the function does it for you
    """
    #Get log polar coordinates. choose the log base 
    lp1, anglestep, log_base=polar_fft(im1, islogr=True, isccs=isccs)
    lp0, anglestep, log_base=polar_fft(im0, islogr=True, isccs=isccs,
                                            radiimax=lp1.shape[1], 
                                            anglestep=anglestep)
    #Find the shift with log of the log-polar images,
    #to compensate for dft intensity repartition
    angle, scale = find_shift_dft(cv2.log(lp0),cv2.log(lp1))
    #get angle in correct units
    angle*=anglestep
    #get scale in linear units
    scale=log_base ** (scale)
    #return angle and scale
    return angle, scale


def find_shift_dft(im0,im1, isccs=False, subpix=True):
    """Find the shift between two images using the DFT method
    
    This algorithm detect a shift using the global phase difference of the DFTs
    
    If the images are already DFT and in the CCS format, set isccs to true.
    In that case the images should have the same size.  
    
    If subpix is True, a gaussian fit is used for subpix precision
    """
    if not isccs:
        im0,im1=dft_optsize_same(im0,im1)
    else:
        #Work only if the shapes are the same
        assert(im0.shape==im1.shape)
    
    #f0*conj(f1)
    mulSpec=cv2.mulSpectrums(im0,im1,flags=0,conjB=True)
    #norm(f0)*norm(f1)
    normccs=cv2.sqrt(cv2.mulSpectrums(im0,im0,flags=0,conjB=True)*
                       cv2.mulSpectrums(im1,im1,flags=0,conjB=True))
    #compute the inverse DFT    
    xc=cv2.dft(ccs_normalize(mulSpec,normccs),
               flags=cv2.DFT_REAL_OUTPUT|cv2.DFT_INVERSE)
    #save shape
    shape=np.asarray(xc.shape)
    #find max
    idx=np.asarray(np.unravel_index(np.argmax(xc),shape)) 
    
    if subpix:
        #define search windows
        X=np.r_[-5:6]
        #get values along X and Y
        Y0=np.take(xc[:,idx[1]],np.r_[idx[0]+X[0]:idx[0]+X[-1]+1],mode='wrap')
        Y1=np.take(xc[idx[0],:],np.r_[idx[1]+X[0]:idx[1]+X[-1]+1],mode='wrap')
        #update idx
        idx=np.float64(idx)        
        idx[0]+=get_peak_pos(X,Y0)
        idx[1]+=get_peak_pos(X,Y1)
        
    #restrics to reasonable values
    idx[idx>shape//2]-=shape[idx>shape//2]
    return idx


def find_shift_cc(im0,im1,ylim=None,xlim=None):
    """Finds the best shift between im0 and im1 using cross correlation
    
    The origin of im1 in the im0 referential is returned
    
    ylim and xlim limit the possible output.
    
    No subpixel precision
    """
    
    #Remove mean
    im0=im0-im0.mean()
    im1=im1-im1.mean()
    #Save shapes as np array
    shape0=np.asarray(im0.shape)
    shape1=np.asarray(im1.shape)
    
    #Compute the offset and the pad (yleft,yright,xtop,xbottom)
    offset=1-shape1
    pad=np.lib.pad(-offset,(1,1),mode='edge')
    
    #apply limit on padding
    if ylim is not None:
        pad[0]=-ylim[0]
        pad[1]=ylim[1]+(shape1-shape0)[0]
    if xlim is not None:
        pad[2]=-xlim[0]
        pad[3]=xlim[1]+(shape1-shape0)[1]
    
    #pad image
    im0, offset = pad_img(im0,pad)
    #compute Cross correlation matrix
    xc=cv2.matchTemplate(np.float32(im0),np.float32(im1),cv2.TM_CCORR)
    #Find maximum of abs (can be anticorrelated)
    idx=np.asarray(np.unravel_index(np.argmax(xc),xc.shape))        
    #Return origin in im0 units
    return idx+offset
    
    
    
########################   
#Medium level functions#
########################



def dft_optsize(im, shape=None):
    """Resize image for optimal DFT and computes it"""
    #save shape
    initshape=im.shape
    #get optimal size
    if shape is None:
        ys=cv2.getOptimalDFTSize(initshape[0]) 
        xs=cv2.getOptimalDFTSize(initshape[1])
        shape=[ys,xs]
    #Add zeros to go to optimal size
    im = cv2.copyMakeBorder(im, 0, shape[0] - initshape[0],
                                0, shape[1] - initshape[1], 
                                borderType=cv2.BORDER_CONSTANT, value=0);
    #Compute dft ignoring 0 rows (0 columns can not be optimized)
    f = cv2.dft(im,nonzeroRows=initshape[0])
    return f
    
    
def dft_optsize_same(im0,im1):
    """Resize 2 image same size for optimal DFT and computes it"""
    #save shape
    shape0=im0.shape
    shape1=im1.shape
    #get optimal size
    ys=max(cv2.getOptimalDFTSize(shape0[0]),
           cv2.getOptimalDFTSize(shape1[0]))
    xs=max(cv2.getOptimalDFTSize(shape0[1]),
           cv2.getOptimalDFTSize(shape1[1]))
    shape=[ys,xs]
    f0=dft_optsize(im0,shape=shape)
    f1=dft_optsize(im1,shape=shape)
    return f0,f1
    
def rotate_scale(im,angle,scale):
    """Rotates and scales the image"""
    rows,cols = im.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-angle*180/np.pi,1/scale)
    im = cv2.warpAffine(im,M,(cols,rows),
                        borderMode=cv2.BORDER_CONSTANT,
                        flags=cv2.INTER_CUBIC)#REPLICATE
    return im
    

def polar_fft(image, isccs=False, anglestep=None, radiimax=None,
              islogr=False):
    """Return dft in polar (or log-polar) units, the angle step 
    (and the log base)
    
    if the image is already in CCS format, set isccs to True
    
    anglestep set the angle step.
    If it is not specified, a value is deduced from the image size
    
    radiimax is the maximal radius (log of radius if islogr is true).
    if not provided, it is deduced from the image size
    
    To get log-polar, set islogr to True
    log_base is the base of the log. It is deduced from radiimax.
    Two images that will be compared should therefore have the same radiimax.
    """
    image=np.float32(image)
    #get dft if not already done
    if not isccs:
        image=dft_optsize(image)
        
    #recenter dft
    image=centered_mag_sq_ccs(image)
    
    #the center is shifted from 0,0 to the ~ center 
    #(eg. if 4x4 image, the center is at [2,2], as if 5x5)
    qshape = np.asarray([image.shape[0]//2,image.shape[1]])
    center = np.asarray([qshape[0],0]) 
    
    #if the angle Step is not given, take the number of pixel
    #on the perimeter as the target #=range/step
    if anglestep is None:
        anglestep=np.pi/qshape.min()/2#range is pi, nbangle = 2r =~pi r
    
    #get the theta range
    theta=np.arange(-np.pi/2,np.pi/2,anglestep,dtype=np.float32)
    
    #For the radii, the units are comparable if the log_base and radiimax are
    #the same. Therefore, log_base is deduced from radiimax
    #The step is assumed to be 1
    if radiimax is None:
        radiimax = qshape.min()
    
    #also as the circle is an ellipse in the image, 
    #we want the radius to be from 0 to 1
    if islogr:
        #The log base solves log_radii_max=log_{log_base}(linear_radii_max)
        #where we decided arbitrarely that linear_radii_max=log_radii_max
        log_base=np.exp(np.log(radiimax) / radiimax)
        radius = ((log_base ** np.arange(0,radiimax,dtype=np.float32))
                    /radiimax)
    else:
        radius = np.linspace(0,1,radiimax, endpoint=False,
                                    dtype=np.float32)
    
    #get x y coordinates matrix (The units are 0 to 1, therefore a circle is
    #represented as an ellipse)
    y=cv2.gemm(np.sin(theta),radius,qshape[0],0,0,
               flags=cv2.GEMM_2_T)+center[0]
    x=cv2.gemm(np.cos(theta),radius,qshape[1],0,0,
               flags=cv2.GEMM_2_T)+center[1]
    
    #get output
    output=cv2.remap(image,x,y,cv2.INTER_LINEAR)#LINEAR, CUBIC,LANCZOS4
    if islogr:
        return output, anglestep, log_base
    else:
        return output, anglestep
        
        
##################    
#Helper functions#
##################
    
    
def pad_img(im,pad):
    """Pad positively with 0 or negatively (cut)
    
    Pad is (ytop, ybottom, xleft, xright)
    or (imin, imax, jmin, jmax)
    """
    #get shape
    shape=im.shape
    #extract offset from padding 
    offset=-pad[::2]
    #if the padding is negatif, cut the matrix
    cut=pad<0
    if cut.any():
        #Extract value for pad
        cut*=pad
        #the left/top components should be positive
        cut[::2]*=-1
        #The right/bottom components can't be 0, replace by shape0
        cut[1::2]+=(cut[1::2]==0)*shape
        #cut the image
        im=im[cut[0]:cut[1],cut[2]:cut[3]]
    #extract positive padding
    ppad=pad>0
    if ppad.any():
        pad=pad*ppad
        #separate pad for application on matrix
        ypad=(pad[0],pad[1])
        xpad=(pad[2],pad[3])
        #prepare matrix
        im=np.lib.pad(im,(ypad,xpad),mode='mean')
    return im, offset
    
    
def clamp_angle(a):
    """return a between -pi/2 and pi/2 (in fourrier plane, +pi is the same)"""
    return (a+np.pi/2)%np.pi - np.pi/2
    
    
def ccs_normalize(compIM,ccsnorm):
    ys=ccsnorm.shape[0]
    xs=ccsnorm.shape[1]
    #start with first column
    ccsnorm[2::2,0]=ccsnorm[1:ys-1:2,0]
    #continue with middle columns
    ccsnorm[:,2::2]=ccsnorm[:,1:xs-1:2]
    #finish whith last row if even
    if xs%2 is 0:
        ccsnorm[2::2,xs-1]=ccsnorm[1:ys-1:2,xs-1]
        
    return compIM/ccsnorm
      
      
def gauss_fit(X,Y):
    """
    Fit the function to a gaussian.
    
    /!\ This uses a slow curve_fit function! do not use if need speed!
    """
    #Can not have negative values
    Y[Y<0]=0
    #define gauss function
    def gaus(x,a,x0,sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
    #get first estimation for parameter
    mean=(X*Y).sum()/Y.sum()
    sigma=np.sqrt((Y*((X-mean)**2)).sum()/Y.sum())
    height=Y.max()
    #fit with curve_fit
    return curve_fit(gaus,X,Y,p0=[height,mean,sigma])
    
    
def gauss_fit_log(X,Y):
    """
    Fit the log of the input to the log of a gaussian.
    
    The least square method is used. 
    As this is a log, make sure the amplitude is >> noise
    
    See the gausslog_sympy.py file for explaination
    """
    #take log data
    Data=np.log(Y)
    #Get Di and Xi
    D=[(Data*X**i).sum() for i in range(3)]
    X=[(X**i).sum() for i in range(5)]
    #compute numerator and denominator for mean and variance
    num=(D[0]*(X[1]*X[4] - X[2]*X[3]) +
         D[1]*(X[2]**2   - X[0]*X[4]) +
         D[2]*(X[0]*X[3] - X[1]*X[2]))
    den=2*(D[0]*(X[1]*X[3] - X[2]**2) + 
           D[1]*(X[1]*X[2] - X[0]*X[3]) +
           D[2]*(X[0]*X[2] - X[1]**2))
    varnum=(-X[0]*X[2]*X[4] + X[0]*X[3]**2 + X[1]**2*X[4] -
            2*X[1]*X[2]*X[3] + X[2]**3)
    #if denominator is 0, can't do anything
    if abs(den) < 0.00001:
        print('Warning: zero denominator!',den)
        return None
    #compute mean and variance
    mean=num/den
    var=varnum/den
    #if variance is negative, the data are not a gaussian
    if var<0:
        print('Warning: negative Variance!',var)
        return None
    return mean,var
    
    
def center_of_mass(X,Y):
    """Get center of mass"""
    return (X*Y).sum()/Y.sum()
    
    
def get_peak_pos(X,Y):
    """Get the peak position"""
    #get cut value (10% biggest peak)
    cut = .1*Y.max()    
    #isolate peak
    peak=Y>cut
    peak,__=label(peak)
    peak=peak==peak[5]
    
    #get X any Y values corresponding to peak
    X=X[peak]
    Y=Y[peak]
    
    #if>2, use fit_log
    if peak.sum() > 2:
        ret,__=gauss_fit_log(X,Y)
        #if fails, use center_of_mass
        if ret is None:
            ret=center_of_mass(X,Y)
    elif peak.sum() >1:
        #If only 2 pixel, gauss fitting is imposible, use center_of_mass
        ret=center_of_mass(X,Y)
    else:
        #1 px peak is easy
        ret=0
    return ret
    

def get_extent(origin, shape):
    """Computes the extent for imshow() (see matplotlib doc)"""
    return [origin[1],origin[1]+shape[1],origin[0]+shape[0],origin[0]]
    
    
def centered_mag_sq_ccs(image):
    """return centered squared magnitude
    
    Check doc Intel* Image Processing Library
    https://www.comp.nus.edu.sg/~cs4243/doc/ipl.pdf
    
    The center is at position (ys//2, 0)    
    """
    #multiply image by image* to get squared magnitude
    image=cv2.mulSpectrums(image,image,flags=0,conjB=True)    
    
    ys=image.shape[0]
    xs=image.shape[1]
    
    #get correct size return
    ret=np.zeros((ys,xs//2+1))
    
    #first column:
    #center
    ret[ys//2,0]=image[0,0]
    #bottom
    ret[ys//2+1:,0]=image[1:ys-1:2,0]   
    #top (Inverted copy bottom)
    ret[ys//2-1::-1,0]=image[1::2,0] 
    
    #center columns
    ret[ys//2:,1:]=image[:(ys-1)//2+1,1::2]
    ret[:ys//2,1:]=image[(ys-1)//2+1:,1::2]
    
    #correct last line if even
    if xs%2 is 0:
        ret[ys//2+1:,xs//2]=image[1:ys-1:2,xs-1]   
        ret[:ys//2,xs//2]=0
        
    return ret
   
   

    
    
    
"""
        i=get_peak_pos(X,Y0)
        j=get_peak_pos(X,Y1)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(X,Y0,'x',label='data')
        plt.plot([i,i],[1,Y0.max()],label='logfit')
        plt.plot([-5,5],[.1*Y0[5],.1*Y0[5]])
        plt.figure()
        plt.plot(X,Y1,'x',label='data')
        plt.plot([j,j],[1,Y1.max()],label='logfit')
        plt.plot([-5,5],[.1*Y1[5],.1*Y1[5]])
"""