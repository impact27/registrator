# -*- coding: utf-8 -*-
"""
Created on Tue May 10 17:15:48 2016

@author: quentinpeter

This module does image registration. 


based on:
    
An FFT-Based Technique for Translation, Rotation, and Scale-Invariant
 Image Registration
B. Srinivasa Reddy and B. N. Chatterji

TODO:
    -Get Precision on angle/scale
    -Check if can tune log_base
    -images with different sizes
    
The scale difference should be reasonable (2x)
    
If you crop the images to reasonable size, the algorithm is much faster:
eg:
512x512   : 0.06s
1024x1024 : 0.42s
4096x4096 : 9s

"""
#import libraries
import numpy as np
import math
import cv2
from scipy.optimize import curve_fit
from scipy.ndimage.measurements import label

def register_images(im0,im1):
    """Finds the rotation, scaling and translation of im1 relative to im0
    """
    #resize for DFT
    im0, f0=dft_resize(im0)
    im1, f1=dft_resize(im1)
    #Get rotation and scale
    angle, scale = find_rotation_scale(f0,f1, isccs=True)
    #apply rotation and scale
    im2=rotate_scale(im1,angle,scale)
    __,f2=dft_resize(im2)
    #Find offset
    #y,x=cross_correlation_shift(im0,im2)
    y,x=shift_fft(f0,f2, isccs=True)
    return angle, scale, [y, x], im2
    
def dft_resize(im):
    """Resize image for optimal DFT"""
    #save shape
    shape=im.shape
    #get optimal size
    ys=cv2.getOptimalDFTSize(shape[0]) 
    xs=cv2.getOptimalDFTSize(shape[1])
    #Add zeros to go to optimal size
    im = cv2.copyMakeBorder(im, 0, ys - shape[0], 0, xs - shape[1], 
                            borderType=cv2.BORDER_CONSTANT, value=0);
    #Compute dft ignoring 0 rows (0 columns can not be optimized)
    f = cv2.dft(im,nonzeroRows=shape[0])
    return im,f
def rotate_scale(im,angle,scale):
    """Rotates and scales the image"""
    rows,cols = im.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-angle*180/np.pi,1/scale)
    im = cv2.warpAffine(im,M,(cols,rows),
                        borderMode=cv2.BORDER_CONSTANT,
                        flags=cv2.INTER_CUBIC)#REPLICATE
    return im
    
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
    
def cross_correlation_shift(im0,im1,ylim=None,xlim=None):
    """Finds the best translation fit between im0 and im1 (top corner)
    
    The origin of im1 in the im0 referential is returned
    
    ylim and xlim limit the possible output.
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
    
    im0, offset = pad_img(im0,pad)
    #compute Cross correlation matrix
    xc=cv2.matchTemplate(np.float32(im0),np.float32(im1),cv2.TM_CCORR)
    #Find maximum of abs (can be anticorrelated)
    idx=np.asarray(np.unravel_index(np.argmax(xc),xc.shape))        
    #Return origin in im0 units
    return idx+offset
    
def clamp(a):
    """return a between -pi/2 and pi/2 (in fourrier plane, +pi is the same)"""
    return (a+np.pi/2)%np.pi - np.pi/2
def fill_norm_ccs(ccs):
    ys=ccs.shape[0]
    xs=ccs.shape[1]
    #start with first column
    ccs[2::2,0]=ccs[1:ys-1:2,0]
    #continue with middle columns
    ccs[:,2::2]=ccs[:,1:xs-1:2]
    #finish whith last row if even
    if xs%2 is 0:
        ccs[2::2,xs-1]=ccs[1:ys-1:2,xs-1]
        
    return ccs
        
def gauss_fit(X,Y):
    """
    Fit the function to a gaussian.
    
    /!\ This uses a slow curve_fit function! do not use if need speed!
    """
    Y[Y<0]=0
    def gaus(x,a,x0,sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
    mean=(X*Y).sum()/Y.sum()
    sigma=np.sqrt((Y*((X-mean)**2)).sum()/Y.sum())
    height=Y.max()
    return curve_fit(gaus,X,Y,p0=[height,mean,sigma])
    
def gauss_fit_log(X,Y):
    """
    Fit the log of the input to the log of a gaussian.
    
    The least square method is used. 
    As this is a log, make sure the amplitude is >> noise
    """
    Data=np.log(Y)
    D=[(Data*X**i).sum() for i in range(3)]
    X=[(X**i).sum() for i in range(5)]
    num=(D[0]*(X[1]*X[4] - X[2]*X[3]) +
         D[1]*(X[2]**2   - X[0]*X[4]) +
         D[2]*(X[0]*X[3] - X[1]*X[2]))
    den=2*(D[0]*(X[1]*X[3] - X[2]**2) + 
           D[1]*(X[1]*X[2] - X[0]*X[3]) +
           D[2]*(X[0]*X[2] - X[1]**2))
    varnum=(-X[0]*X[2]*X[4] + X[0]*X[3]**2 + X[1]**2*X[4] -
            2*X[1]*X[2]*X[3] + X[2]**3)
    if abs(den) < 0.00001:
        print('Warning: zero denominator!',den)
        return None
    mean=num/den
    var=varnum/den
    if var<0:
        print('Warning: negative Variance!',var)
        return None
    return mean,var
def COM(X,Y):
    """
    Get center of mass
    """
    return (X*Y).sum()/Y.sum()
    
def getVal(X,Y,cut):
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
        #if fails, use COM
        if ret is None:
            ret=COM(X,Y)
    #if>1, use COM
    elif peak.sum() >1:
        ret=COM(X,Y)
    else:
        #default is 0
        ret=0
    return ret
     
def shift_fft(im0,im1, isccs=False, subpix=True):
    """The "official" shift method"""
    #TODO: different shapes
    if not isccs:
        __,im0=dft_resize(im0)
        __,im1=dft_resize(im1)
    
    mulSpec=cv2.mulSpectrums(im0,im1,flags=0,conjB=True)
    normccs=cv2.sqrt(cv2.mulSpectrums(im0,im0,flags=0,conjB=True)*
                       cv2.mulSpectrums(im1,im1,flags=0,conjB=True))
    normccs=fill_norm_ccs(normccs)
    
    xc=cv2.dft(mulSpec/normccs, flags=cv2.DFT_REAL_OUTPUT|cv2.DFT_INVERSE)
    """
    if im0.shape is not im1.shape:
        shapediff=np.array(im0.shape)-np.array(im1.shape)
        pad0=-shapediff*(shapediff<0)
        pad1=shapediff*(shapediff>0)
        im0=np.lib.pad(im0,((0,pad0[0]),(0,pad0[1])),mode='constant')
        im1=np.lib.pad(im1,((0,pad1[0]),(0,pad1[1])),mode='constant')
        """
    shape=np.asarray(xc.shape)
    #find max
    idx=np.asarray(np.unravel_index(np.argmax(xc),shape)) 
    
    if subpix:
        #define search windows
        X=np.r_[-5:6]
        #get values along X and Y
        Y0=np.take(xc[:,idx[1]],np.r_[idx[0]+X[0]:idx[0]+X[-1]+1],mode='wrap')
        Y1=np.take(xc[idx[0],:],np.r_[idx[1]+X[0]:idx[1]+X[-1]+1],mode='wrap')
        #Extract peak
        cut=3*xc.std()
        #update idx
        idx=np.float64(idx)        
        idx[0]+=getVal(X,Y0,cut)
        idx[1]+=getVal(X,Y1,cut)
        
    #restrics to reasonable values
    idx[idx>shape//2]-=shape[idx>shape//2]
    return idx
     
def find_rotation_scale(im0,im1,isccs=False):
    """Compares the images and return the best guess for the rotation angle,
    and scale difference
    
    Scale precision relative to first image.
    
    The angle and scale search can be limited by alim and slim respectively
    -pi/2< alim < pi/2
    """
    #Get log polar coordinates. choose the log base 
    lp1, anglestep, log_base=polar_fft(im1, islogr=True,isccs=isccs)
    lp0, anglestep, log_base=polar_fft(im0,radiimax=lp1.shape[1], 
                                           anglestep=anglestep,
                                           islogr=True,
                                           isccs=isccs)
    angle, scale = shift_fft(cv2.log(lp0),cv2.log(lp1))
     
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
        alim=[-np.pi/2,np.pi/2]
    #transform the limit in pixel size
    alim= np.int64((np.array(alim))/anglestep)
    if slim is not None:
        slim=np.int64(np.log(slim)/np.log(log_base))
    #compute the cross correlattion to extract the angle and scale     
    angle,scale= cross_correlation_shift(np.log(lp0),np.log(lp1),
                                         ylim=alim,xlim=slim,
                                         ypadmode='wrap')
    
    """
    

def get_extent(origin, shape):
    """Computes the extent for imshow() (see matplotlib doc)"""
    return [origin[1],origin[1]+shape[1],origin[0]+shape[0],origin[0]]
    
def centered_mag_sq_CCS(image):
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
   
def polar_fft(image, isccs=False, anglestep=None, radiimax=None,
              islogr=False):
    """Return fft in polar (or log-polar) units
    
    if the image is already fft2 and fftshift, set isccs to True
    
    anglestep set the precision on the angle.
    If it is not specified, a value is deduced from the image size
    
    radiimax is the maximal radius (log of radius if islogr is true).
    if not provided, it is deduced from the image size
    
    To get log-polar, set islogr to True
    log_base is the base of the log. It is deduced from radiimax.
    Two images that will be compared should therefore have the same radiimax.
    """
    #get dft if not already done
    if not isccs:
        __,image=dft_resize(image)
        
    #recenter dft
    image=centered_mag_sq_CCS(image)
    
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
        log_base=math.exp(math.log(radiimax) / radiimax)
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
    #y = qshape[0]*radius * np.sin(theta) + center[0]
    #x = qshape[1]*radius * np.cos(theta) + center[1]
    
    #get output
    output=cv2.remap(image,x,y,cv2.INTER_LINEAR)#LINEAR, CUBIC,LANCZOS4
    #output=map_coordinates(image, [y, x]) 
    if islogr:
        return output, anglestep, log_base
    else:
        return output, anglestep
def uint8sc(im):
    """scale the image to fit in a uint8"""
    im=im-im.min()
    im=im/(im.max()/255)
    return np.uint8(im)
    
    
    
"""
i=getVal(X,Y0,cut)
j=getVal(X,Y1,cut)
import matplotlib.pyplot as plt
plt.figure()
plt.plot(X,Y0,'x',label='data')
#plt.plot([popt0[1],popt0[1]],[1,Y0.max()],label='fit')
plt.plot([i,i],[1,Y0.max()],label='logfit')
plt.plot([-5,5],[cut,cut])
plt.figure()
plt.plot(X,Y1,'x',label='data')
#plt.plot([popt1[1],popt1[1]],[1,Y1.max()],label='fit')
plt.plot([j,j],[1,Y1.max()],label='logfit')
plt.plot([-5,5],[cut,cut])
"""