'''A collection of functions for use in Jupyter notebooks.'''

import sys
import os
## from datapype.datafits import DataFits    # Gets the function that makes DataFits io objects.
from darepype.drp.datafits import DataFits   # Gets the function that makes DataFits io objects.
import numpy as np
from astropy.io import fits               # Need this if you want to use astropy.io io objects.
from ipywidgets import interact           # Need this for interactive plots.
import configobj                          # What is this? Where does it come from? When is it used?
import scipy.ndimage as nd                # Functions for manipulating images.
#import matplotlib
#from matplotlib import pylab, mlab
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm     # Machinery for LogNorm scaling of intensities.
from matplotlib.colors import SymLogNorm  # Machinery for SymLogNorm scaling of intensities.
from matplotlib.colors import PowerNorm   # Machinery for LogNorm (e.g., square root) scaling of intensities.
from astropy.stats import mad_std         # This is the median absolute deviation, a more robust estimator than std.

## Define some functions:
## Al Harper modified 190322. Tweaked function head().
## Al Harper modified 190322. Added option to rowcolplot() for plotting only row or only column.
## Al Harper modified 190322. Plot height divided by 2 if plotrow=False or plotcol=False.
## Al Harper modified 190323. In displaypic, modified parameter structure and eliminated cursor. Created
##     displaypic2 that has the modified parameter structure, but still has the cursor.

##------------------------------------------------------------------------------------------------------------
## A couple elementary interactive image display functions, displaypic() and displaypic2().

def displaypic(image,titlestring,rows,cols,istart=(0.,65535.),ilims=(0.,65535.,100.),\
               figlims=(10,10), negative=False):
    '''
    A simple interactive image display tool with intensity sliders.
    istart = tuple with initial intensity lower and upper limits.
    ilims = tuple with lower limit, upper limit, and step size for the intensity sliders.
    figlims = tuple with x and y sizes of the image display.
    negative = False for white stars (photographic positive) and True for black stars (photographic negative).
    Al Harper modified 190323. Changed input parameter structure and eliminated cursor sliders from original
    version of displaypic.
    '''
    def showpic(vmn=istart[0],vmx=istart[1]):
        pic = image.copy()
        plt.figure(figsize = (figlims[0],figlims[1]))
        plt.title(titlestring)
        if negative == True:
            bw = 'gray_r'
        else: bw = 'gray'
        plt.imshow(pic, bw, interpolation='nearest', vmin = vmn, vmax = vmx)
        plt.colorbar(orientation = 'vertical',shrink = 0.8)
    interact(showpic,vmn=(ilims[0],ilims[1],ilims[2]),vmx=(ilims[0],ilims[1],ilims[2]))

def displaypic2(image,titlestring,rows,cols,istart=(0.,65535.),ilims=(0.,65535.,100.),rcstart=(0,0),\
               figlims=(10,10), negative=False):
    '''
    A simple interactive image display tool with intensity sliders and a cursor.
    istart = tuple with initial intensity lower and upper limits.
    ilims = tuple with lower limit, upper limit, and step size for the intensity sliders.
    figlims = tuple with x and y sizes of the image display.
    negative = False for white stars (photographic positive) and True for black stars (photographic negative).
    rcstart = tuple with initial position of the cursor (row, col).
    Al Harper created 190323. Changed input parameter structure and kept cursor sliders from original
    displaypic, then re-named to displaypic2.
    '''
    def showpic(vmn=istart[0],vmx=istart[1],iy=rcstart[0],ix=rcstart[1]):
        pic = image.copy()
        plt.figure(figsize = (figlims[0],figlims[1]))
        plt.title(titlestring)
        if negative == True:
            bw = 'gray_r'
        else: bw = 'gray'
        plt.imshow(pic, bw, interpolation='nearest', vmin = vmn, vmax = vmx)
        plt.colorbar(orientation = 'vertical',shrink = 0.8)
        plt.scatter(ix,iy,s=80, facecolors='none', edgecolors='g')
        print (pic[iy,ix])
    interact(showpic,vmn=(ilims[0],ilims[1],ilims[2]),vmx=(ilims[0],ilims[1],ilims[2]),\
             iy=(0,rows-1,1),ix=(0,cols-1,1))

##------------------------------------------------------------------------------------------------------------
def show_image(image,imagename,iy=100,ix=100,ydelta=100,xdelta=100,lims=(-100.,65536.,100.),linthresh=100.,linscale=0.3,\
       cursorcolors=('b','g'), starts=(3.,5.), steps = 1, plotgrid=False, whichcolor=['gray','gray_r','rainbow'],initcolor=1):
    '''Interactive display of single image.
    Inputs are:
    image-- a numpy array with dimensions (rows, columns).
    imagename-- a name string for the image to include in plot title, e.g., a filename.
    iy, ix-- row and column numbers for positions of cursor lines (horizontal, vertical).
    ydelta, xdelta-- displacements to second set of cursor lines.
    lims-- tuple with vmin, vmax, and step size for intensity scale. Note that if vmin is negative,
        LogNorm and PowerNorm may not work for some images).
    linthresh-- threshold parameter for SymLogNorm (sets range for the linear portion).
    linscale-- fraction of colorscale devoted to linear portion of SymLogNorm colorscale.
    cursorcolors-- tuple with colors of the two sets of cursors.
    starts-- tuple containing coefficients of mad_std for setting initial values of display intensity limits.
    steps-- size of steps in y and x for cursor positions.
    Outputs are:
    interactive image display.
    Area median-- the median of pixel values inside the area defined by the two cursor sets.
    Area std-- the standard deviation of pixel values inside the area defined by the cursor sets.
    Created by Al Harper, 180612.
    Al Harper modified 190311: Added 1 to ydelta and xdelta in the area integration so can integrate
    over entire image area. Note that this means the "bounded area" includes the cursor pixels.
    Al Harper modified 190318: Edit notes. Print out image median and mad_std. Add plotgrid True/False option.
    '''
    rows = image.shape[0]
    cols = image.shape[1]
    # Derive some scaling parameters.
    picmedian = np.nanmedian(image)
    picmad = mad_std(image,ignore_nan=True)
    lowstart = picmedian - starts[0]*picmad
    highstart = picmedian + starts[1]*picmad
    # An array of color scales for the interactive display slider.
    whichcolor = ['gray','gray_r','rainbow']
    def showpic(vmx=highstart,vmn=lowstart,iy=iy,ix=ix,ydelta=ydelta, xdelta=xdelta, scale=0, colorindex=initcolor):
        plt.figure(figsize = (14,14))
        if plotgrid == True:plt.grid()
        color= whichcolor[colorindex]
        print('colormap =',color)

        cx, cy = range(cols), range(rows)
        vx, vy, vdx, vdy = np.ones((rows)), np.ones((cols)), np.ones((rows)), np.ones((cols))
        vx, vy = ix * vx, iy * vy
        vdx, vdy = (ix + xdelta) * vdx, (iy + ydelta) * vdy

        plt.plot(cx,vy, cursorcolors[0], linewidth=0.4)    # Plot horizontal line at specified y value.,
        plt.plot(cx,vdy, cursorcolors[1], linewidth=0.4)   # Plot horizontal line at y + dy.
        plt.plot(vx,cy, cursorcolors[0], linewidth=0.4)    # Plot vertical line at specified x value.
        plt.plot(vdx,cy, cursorcolors[1], linewidth=0.4)   # Plot vertical line at x + dx.

        if scale == 0:
            whichscale = 'Scale = linear'
            plt.imshow(image, color, interpolation='nearest', vmin=vmn,vmax=vmx)
        if scale == 1:
            plt.imshow(image, color, interpolation='nearest', norm=LogNorm(vmin=vmn,vmax=vmx))
            whichscale = 'Scale = log'
        if scale == 2:
            plt.imshow(image, color, interpolation='nearest', norm=PowerNorm(gamma=0.5,vmin=vmn,vmax=vmx))
            whichscale = 'Scale = sqr_root'
        if scale == 3:
            vmin = -65535.
            plt.imshow(image, color, interpolation='nearest', norm=SymLogNorm(linthresh=linthresh,\
                        linscale=linscale,vmin=vmn,vmax=vmx))
            whichscale = 'Scale = sym_log'
        print( 'Area median =',np.nanmedian(image[iy:iy+ydelta+1,ix:ix+xdelta+1]),\
        '   Area std =',np.nanstd(image[iy:iy+ydelta+1,ix:ix+xdelta+1]),\
        '   Area mad_std =',mad_std(image[iy:iy+ydelta+1,ix:ix+xdelta+1],ignore_nan=True))
        plt.title(imagename + '   ' + whichscale)
        plt.colorbar(orientation = 'vertical', shrink = 0.8)
        print('median =', picmedian,'   mad_std =',picmad)
        plt.show()
    showpic()
    #     plt.scatter(ix,iy,s=80, facecolors='none', edgecolors='g')  # Positions a circular cursor at ix, iy.
    #interact(showpic,vmx=(lims[0],lims[1],lims[2]),vmn=(lims[0], lims[1],lims[2]),iy=(0,rows-1,steps),\
    #         ix=(0,cols-1,steps),ydelta=(0,rows-1,steps),xdelta=(0,cols-1,steps),scale=(0,3,1),colorindex=(0,2,1))


##------------------------------------------------------------------------------------------------------------
def blink_images(files,imagestack,refimage=0,b2=False,blink=0,blinkset=(0,1),\
          iy=10,ix=10,ydelta=10,xdelta=10, cursorcolors=('b','g'),starts=(3.,5.),\
          lims=(1.,65535,10.),linthresh=100.,linscale=0.3,addgrid=True,xystep=1):
    '''Interactive display of multiple images.
    Inputs are:
    files-- a list of filenames for the images.
    imagestack-- a numpy array with dimensions (image number, row, column).
    refimage-- index number of one of the images in the stack from which to derive autoscaling parameters.
    b2-- True to blink between two of the images, False to cycle through entire stack.
    blink-- initial value of the index to the image stack.
    blinkset-- image indices of the two images to blink.
    iy, ix-- row and column numbers for positions of cursor lines (horizontal, vertical).
    ydelta, xdelta-- displacements to second set of cursor lines.
    cursorcolors-- strings for cursor colors.
    starts-- coefficients (of -+ median absolute deviation) to set initial limits of display intensity.
    lims-- lower limit, upper limit, and stepsize for the intensity slider.
    linthresh-- threshold parameter for SymLogNorm (sets range for the linear portion).
    linscale-- fraction of colorscale devoted to linear portion of SymLogNorm colorscale.
    addgrid-- True to include grid, False not to include grid.
    xystep-- Step size for iy and ix sliders. Step size for ydelta and xdelta sliders is set to 1.
    Outputs are:
    Area median-- the median of pixel values within the area bounded by the two cursors sets.
    Area std-- the standard deviation of pixel values within the area bounded by the two cursors sets.
    Area mad-- the median absolute deviation of pixel values within the area bounded by the two cursors sets.
    Created by Al Harper, 180611
    Modified by Al Harper, 190311: Added 1 to ydelta and xdelta in the area integration so can integrate
    over entire image area. Note that this means the "bounded area" includes the cursor pixels.
    Modified by Al Harper, 190413. Changed limits on ydelta and xdelta so they can now be negative.
    '''

    refimage = refimage
    rows = imagestack[refimage].shape[0]
    cols = imagestack[refimage].shape[1]
    blinkset = blinkset
    blinkarray = np.zeros((len(files),rows,cols))
    for i in range(len(files)):
        blinkarray[i] = imagestack[i]
#     scales = ('Linear','LogNorm','')  # Don't think this line is needed. Drop if things work without it.
    # Derive some scaling parameters from one of the images.
    whichmedian = np.nanmedian(imagestack[refimage])
    whichmad = mad_std(imagestack[refimage][0:10,:],ignore_nan=True)
    lowstart = whichmedian - starts[0]*whichmad
    highstart = whichmedian + starts[1]*whichmad
    # An array of color scales for the interactive display slider.
    whichcolor = ['gray','gray_r','rainbow']
    def showpic(vmx=highstart,vmn=lowstart,iy=iy,ix=ix,ydelta=ydelta,xdelta=xdelta,scale=0,colorindex=1,blink=blink,blink2=0,b2=b2):
#         print b2
        plt.figure(figsize = (14,14))
        if addgrid == True:plt.grid()
        color= whichcolor[colorindex]

        cx, cy = range(cols), range(rows)
        vx, vy, vdx, vdy = np.ones((rows)), np.ones((cols)), np.ones((rows)), np.ones((cols))
        vx, vy = ix * vx, iy * vy
        vdx, vdy = (ix + xdelta) * vdx, (iy + ydelta) * vdy

        plt.plot(cx,vy, cursorcolors[0], linewidth=0.4)
        plt.plot(cx,vdy, cursorcolors[1], linewidth=0.4)
        plt.plot(vx,cy, cursorcolors[0], linewidth=0.4)
        plt.plot(vdx,cy, cursorcolors[1], linewidth=0.4)

        if b2 == True:
            if scale == 0:
                whichscale = 'linear'
                plt.title(files[blinkset[blink2]] + '   ' + whichscale)
                plt.imshow(blinkarray[blinkset[blink2]], color, interpolation='nearest', vmin=vmn,vmax=vmx)
            if scale == 1:
                whichscale = 'log'
                plt.title(files[blinkset[blink2]] + '   ' + whichscale)
                plt.imshow(blinkarray[blinkset[blink2]], color, interpolation='nearest', norm=LogNorm(vmin=vmn,vmax=vmx))
            if scale == 2:
                whichscale = 'square root'
                plt.title(files[blinkset[blink2]] + '   ' + whichscale)
                plt.imshow(blinkarray[blinkset[blink2]], color, interpolation='nearest', norm=PowerNorm(gamma=0.5,vmin=vmn,vmax=vmx))
            if scale == 3:
                whichscale = 'symmetric log'
                plt.title(files[blinkset[blink2]] + '   ' + whichscale)
                plt.imshow(blinkarray[blinkset[blink2]], color, interpolation='nearest', norm=SymLogNorm(vmin=vmn,\
                            vmax=vmx,linthresh=linthresh,linscale=linscale))
            print('Area median =',np.nanmedian(blinkarray[blinkset[blink2]][iy:iy+ydelta+1,ix:ix+xdelta+1]),\
            '   Area std =',np.nanstd(blinkarray[blinkset[blink2]][iy:iy+ydelta+1,ix:ix+xdelta+1]),\
            '   Area mad_std =',mad_std(blinkarray[blinkset[blink2]][iy:iy+ydelta+1,ix:ix+xdelta+1],ignore_nan=True))
        else:
            if scale == 0:
                whichscale = 'linear'
                plt.title(files[blink] + '   ' + whichscale)
                plt.imshow(blinkarray[blink], color, interpolation='nearest', vmin=vmn,vmax=vmx)
            if scale == 1:
                whichscale = 'log'
                plt.title(files[blink] + '   ' + whichscale)
                plt.imshow(blinkarray[blink], color, interpolation='nearest', norm=LogNorm(vmin=vmn,vmax=vmx))
            if scale == 2:
                whichscale = 'square root'
                plt.title(files[blink] + '   ' + whichscale)
                plt.imshow(blinkarray[blink], color, interpolation='nearest', norm=PowerNorm(gamma=0.5,vmin=vmn,vmax=vmx))
            if scale == 3:
                whichscale = 'symmetric log'
                plt.title(files[blink] + '   ' + whichscale)
                plt.imshow(blinkarray[blink], color, interpolation='nearest', norm=SymLogNorm(vmin=vmn,\
                            vmax=vmx,linthresh=linthresh,linscale=linscale))
            print('Area median =',np.nanmedian(blinkarray[blink][iy:iy+ydelta+1,ix:ix+xdelta+1]),\
            '   Area std =',np.nanstd(blinkarray[blink][iy:iy+ydelta+1,ix:ix+xdelta+1]),\
            '   Area mad_std =',mad_std(blinkarray[blink][iy:iy+ydelta+1,ix:ix+xdelta+1],ignore_nan=True))
        plt.colorbar(orientation = 'vertical', shrink = 0.8)
    interact(showpic,vmx=(lims[0],lims[1],lims[2]),vmn=(lims[0],lims[1],lims[2]),iy=(0,rows-1,xystep),\
             ix=(0,cols-1,xystep),ydelta=(-rows+1,rows-1,1),xdelta=(-cols+1,cols-1,1),scale=(0,3,1),colorindex=(0,2,1),\
             blink=(0,len(files)-1,1),blink2=(0,1,1))


##------------------------------------------------------------------------------------------------------------
def blink2images(image0,name0,image1,name1,iy=10,ix=10,ydelta=10,xdelta=10,\
                 cursorcolors=('b','g'),starts=(3.,5.),lims=(1.,65535,10.),linthresh=100.,linscale=0.3, xystep=10,addgrid = True):

    '''Interactive display of multiple images.
    Inputs are:
    image0,image1-- the two images to compare.
    name1,name2-- the names of the two images (e.g., filenames).
    iy, ix-- row and column numbers for positions of cursor lines (horizontal, vertical).
    ydelta, xdelta-- displacements to second set of cursor lines.
    cursorcolors-- tuple containing strings for cursor colors.
    starts-- coefficients (of -+ median absolute deviation) to set initial limits of display intensity.
    lims-- lower limit, upper limit, and stepsize for the intensity slider.
    linthresh-- threshold parameter for SymLogNorm (sets range for the linear portion)
    linscale-- fraction of colorscale devoted to linear portion of SymLogNorm colorscale
    addgrid-- True to include grid, False not to include grid.
    xystep-- Step size for iy and ix sliders. Step size for ydelta and xdelta sliders is set to 1.    Outputs are:
    Area median-- the median of pixel values within the area between the two cursors sets
    Area std-- the standard deviation of pixel values within the area bounded by the two cursors sets
    Area mad-- the median absolute deviation of pixel values within the area bounded by the two cursors sets
    Created by Al Harper, 180614
    Last modified by Al Harper, 190311: Added 1 to ydelta and xdelta in the area integration so can integrate
    over entire image area. Note that this means the "bounded area" includes the cursor pixels.
    '''

    rows = image0.shape[0]
    cols = image0.shape[1]
    blinkarray = np.zeros((2,rows,cols))
    blinkarray[0] = image0
    blinkarray[1] = image1
    files2blink = (name0, name1)

    # Derive some scaling parameters from one of the images.
    whichmedian = np.nanmedian(image0)
    whichmad = mad_std(image0,ignore_nan=True)
    lowstart = whichmedian - starts[0]*whichmad
    highstart = whichmedian + starts[1]*whichmad

    # An array of color scales for the interactive display slider.
    whichcolor = ['gray','gray_r','rainbow']

    def showpic(vmx=highstart,vmn=lowstart,iy=iy,ix=ix,ydelta=ydelta,xdelta=xdelta,scale=0,colorindex=1,blink=0):
        plt.figure(figsize = (14,14))
        if addgrid == True:plt.grid()
        color= whichcolor[colorindex]

        cx, cy = range(cols), range(rows)
        vx, vy, vdx, vdy = np.ones((rows)), np.ones((cols)), np.ones((rows)), np.ones((cols))
        vx, vy = ix * vx, iy * vy
        vdx, vdy = (ix + xdelta) * vdx, (iy + ydelta) * vdy

        plt.plot(cx,vy, cursorcolors[0], linewidth=0.4)
        plt.plot(cx,vdy, cursorcolors[1], linewidth=0.4)
        plt.plot(vx,cy, cursorcolors[0], linewidth=0.4)
        plt.plot(vdx,cy, cursorcolors[1], linewidth=0.4)

        if scale == 0:
            whichscale = 'linear'
            plt.title(files2blink[blink] + '   ' + whichscale)
            plt.imshow(blinkarray[blink], color, interpolation='nearest', vmin=vmn,vmax=vmx)
        if scale == 1:
            whichscale = 'log'
            plt.title(files2blink[blink] + '   ' + whichscale)
            plt.imshow(blinkarray[blink], color, interpolation='nearest', norm=LogNorm(vmin=vmn,vmax=vmx))
        if scale == 2:
            whichscale = 'square root'
            plt.title(files2blink[blink] + '   ' + whichscale)
            plt.imshow(blinkarray[blink], color, interpolation='nearest', norm=PowerNorm(gamma=0.5,vmin=vmn,vmax=vmx))
        if scale == 3:
            whichscale = 'symmetric log'
            plt.title(files2blink[blink] + '   ' + whichscale)
            plt.imshow(blinkarray[blink], color, interpolation='nearest', norm=SymLogNorm(vmin=vmn,\
                        vmax=vmx,linthresh=linthresh,linscale=linscale))
        print( 'Area median =',np.nanmedian(blinkarray[blink][iy:iy+ydelta+1,ix:ix+xdelta+1]),\
        '   Area std =',np.nanstd(blinkarray[blink][iy:iy+ydelta+1,ix:ix+xdelta+1]),\
        '   Area mad_std =',mad_std(blinkarray[blink][iy:iy+ydelta+1,ix:ix+xdelta+1],ignore_nan=True))
        plt.colorbar(orientation = 'vertical', shrink = 0.8)
    interact(showpic,vmx=(lims[0],lims[1],lims[2]),vmn=(lims[0],lims[1],lims[2]),iy=(0,rows-1,xystep),\
             ix=(0,cols-1,xystep),ydelta=(0,rows-1,1),xdelta=(0,cols-1,1),scale=(0,3,1),colorindex=(0,2,1),\
             blink=(0,1,1))
##------------------------------------------------------------------------------------------------------------------
def quickpic(image,titlestring='',madfactor=(3.,10.),mask=([0],[0]),mask2=([0],[0]) ,csize=100,edgecolor='r',\
               edgecolor2='b', figlims=(10,10), negative=True, ilims=(0.,0.)):
    '''
    A simple image display tool with initial autoscaling based on image median and mad_std.
    It also includes the option to plot circles around positions defined in one or two image masks.
    image: A 2D image.
    titlestring: A text string for the figure title. Default is empty string.
    madfactor: A tuple with multipliers for mad_std to subtract or add to median to get vmin and vmax for imshow.
    mask: A tuple of ndarrays with lists of the row and column coordinates of the circles to plot.
    mask2: Another mask array for a second set of circles with twice the radius and different color.
    csize: Sets size of circles for mask positions. Circles for mask2 positions are two times larger.
    edgecolor: The color for the mask circles.
    edgecolor2: The color for the mask2 circles.
    figlims: Tuple to determine figsize.
    negative: =True (default) for black on white background and =False for white on black background.
    Al Harper created 190818.
    '''

    pic = image.copy()
    rows, cols = pic.shape
    med, mad = np.nanmedian(pic), mad_std(pic,ignore_nan=True)
    if ilims == (0.,0.):
        vmn, vmx = med - mad * madfactor[0], med + mad * madfactor[1]
    else:
        vmn, vmx = ilims
    plt.figure(figsize = (figlims))
    plt.title(titlestring)
    if negative == True:
        bw = 'gray_r'
    else: bw = 'gray'
    plt.imshow(pic, bw, interpolation='nearest', vmin = vmn, vmax = vmx)
    plt.colorbar(orientation = 'vertical',shrink = 0.8)
    for i in mask:
        if mask[0][0] != 0 and mask[1][0] != 0:
            plt.scatter(mask[1],mask[0],s=csize, facecolors='none', edgecolors=edgecolor)
        if mask2[0][0] != 0 and mask2[1][0] != 0:
            plt.scatter(mask2[1],mask2[0],s=2*csize, facecolors='none', edgecolors=edgecolor2)

##-----------------------------------------------------------------------------------------------------------------

def getpatch(img,imgname,rowcenter,colcenter,hw,threshold=0,figlims=(20,5),patchname='',show=True):
    '''Plots subimage of img, prints some things, and returns some things.'''

    outs = (0,0,0,0,0)
    outs2 = (0,0,0,0,0)
    outs3 = -1
    patch = np.zeros((2*hw[0]+1,2*hw[1]+1))

    if rowcenter > hw[0] and colcenter > hw[1] and img.shape[0] - rowcenter > hw[0] and img.shape[1] - colcenter > hw[1]:

        patch = img[rowcenter-hw[0]:rowcenter+hw[0]+1,colcenter-hw[1]:colcenter+hw[1]+1].copy()
        patchmax = np.nanmax(patch)

        nanmask = np.isnan(patch)
        patch[nanmask] = -1e10
        nans = np.where(patch == -1e10)
        patch[nans] = 0.
        outs3 = len(nans[0])
        mask = np.where(patch < threshold)
        patch2 = patch.copy()
        patch2[mask] = 0.

        if show == True:
            plt.figure(figsize=figlims)
            print('row =',rowcenter,'   col =',colcenter)
            plt.subplot(1,3,1)
            plt.title('{}  pmax={:0.0f}  row={:d}  col={:d}'.format(patchname,patchmax,rowcenter,colcenter))
            plt.imshow(patch)
            print('number of nans =',len(nans[0]))
            if len(nans[0]) > 0:
                print('nans =',nans)
                print('')

        if patchmax > threshold:

            outs = moments(patch)
            outs2 = moments(patch2)
            psig, x, y, xw, yw = outs2

            if show == True:
                print(outs)
                print(outs2)

                rowsig = patch[int(y),:]
                rowpeak = np.max(rowsig)
                xrow = range(patch.shape[1])
                rowline = np.ones((patch.shape[1]))
                colsig = patch[:,int(x)]
                colpeak = np.max(colsig)
                xcol = range(patch.shape[0])
                colline = np.ones((patch.shape[0]))

                plt.subplot(1,3,2)
                plt.title('rowsignal,  x_hwidth ={:.2f},  peak ={:.2e}'.format(xw,rowpeak))
                plt.grid()
                plt.plot(xrow,rowsig, '-o')
                plt.plot(xrow,rowline * threshold)

                plt.subplot(1,3,3)
                plt.title('colsignal,   y_hwidth ={:.2f},  peak ={:.2e}'.format(yw,colpeak))
                plt.grid()
                plt.plot(xcol,colsig, '-o')
                plt.plot(xcol,colline * threshold)

            print('')

    else:
        print('The patch at row =',rowcenter,' and col=',colcenter,' is too close to the edge of the image.')
        print('')

    return patch, outs, outs2, outs3

##------------------------------------------------------------------------------------------------------------------
def rowplot(image, rowstart, filename, autoscale = True, ylims = (-100., 10000.), xlims = (0,1024)):
    '''Plots a single column of an image. If autoscale != 'autoscale' the y scale limits are set by the tuple ylims.'''
    def plotrow(row=rowstart, yl = ylims[0], yh = ylims[1]):
        intensity = image[row]
        x = range(image.shape[1])
        plt.figure(figsize = (18,4))
        if autoscale != True: plt.ylim(yl,yh)
        plt.xlim(xlims[0],xlims[1])
        rowmedian = np.nanmedian(intensity[xlims[0]:xlims[1]])
        rowstd = np.nanstd(intensity[xlims[0]:xlims[1]])
        rowmax = np.nanmax(intensity[xlims[0]:xlims[1]])
        rowmin = np.nanmin(intensity[xlims[0]:xlims[1]])
        plt.grid()
        plt.title(filename+'       Row ='+str(row)+'    Max ='+str(rowmax)+'    Min ='+str(rowmin)+ \
              '    Median ='+str(rowmedian)+'    Std ='+str(rowstd))
        plt.plot(x,intensity)
    interact(plotrow,row = (0,image.shape[1]-1,1), yl=(-65000.,65000.,100.), yh=(-65000.,65000.,100.))

##------------------------------------------------------------------------------------------------------------
def colplot(image, colstart, filename,  autoscale = True, ylims = (-100., 10000.), xlims = (0,1024)):
    '''Plots a single column of an image. If autoscale != 'autoscale' the y scale limits are set by the tuple ylims.'''
    def plotcol(col=colstart, yl = ylims[0], yh = ylims[1]):
        intensity = image[:,col]
        x = range(image.shape[0])
        plt.figure(figsize = (18,4))
        if autoscale != True: plt.ylim(yl,yh)
        plt.xlim(xlims[0],xlims[1])
        colmedian = np.nanmedian(intensity[xlims[0]:xlims[1]])
        colstd = np.nanstd(intensity[xlims[0]:xlims[1]])
        colmax = np.nanmax(intensity[xlims[0]:xlims[1]])
        colmin = np.nanmin(intensity[xlims[0]:xlims[1]])
        plt.grid()
        plt.title(filename+'       Col ='+str(col)+'    Max ='+str(colmax)+'    Min ='+str(colmin)+\
              '    Median ='+str(colmedian)+'    Std ='+str(colstd))
        plt.plot(x,intensity)
    interact(plotcol,col = (0,image.shape[0]-1,1))

##------------------------------------------------------------------------------------------------------------
def rowcolplot(image,filename,rowstart,colstart,ylims=(-100., 65535.),autoscale=True,xstep=10,ystep=100,\
               dots=False,figlims=(18,12),plotrow=True,plotcol=True):
    '''Plots a row and a column of an image. If autoscale != True, the y scale limits are set by the tuple ylims.
    The tuple ylims controls the y scale of both plots. The x limit row and column tuples (rxlims and cxlims)
    are set independently.
    Al Harper modified 190818: Added option to plot dots or not.
    Also changed order of parameters-- put filename second and moved autoscale after ylims.
    Al Harper modified 190319: Added parameter for figlims.
    Al Harper modified 190322: Added option to plot only row or only column.
    Al Harper modified 190322. Plot height divided by 2 if plotrow=False or plotcol=False.
    '''

    rows = image.shape[0]
    cols = image.shape[1]

    def plotrowcol(row=rowstart,col=colstart,yl=ylims[0],yh=ylims[1],rxl=0,rxh=cols,cxl=0,cxh=rows,\
                   plotrow=plotrow,plotcol=plotcol):
        if dots == True:
            dotstring = '-o'
        else:
            dotstring = ''
        if plotrow == True and plotcol == True:
            figx, figy = figlims[0], figlims[1]
        else:
            figx, figy = figlims[0], figlims[1]/2
        rowsig = image[row,:]
        colsig = image[:,col]
        rowlim = image.shape[1]
        collim = image.shape[0]
        xrow = range(rowlim)
        xcol = range(collim)
        plt.figure(figsize = (figx, figy))
        rowmedian = np.nanmedian(rowsig[rxl:rxh])
        colmedian = np.nanmedian(colsig[cxl:cxh])
        rowstd = np.nanstd(rowsig[rxl:rxh])
        colstd = np.nanstd(colsig[cxl:cxh])
        rowmadstd = mad_std(rowsig[rxl:rxh],ignore_nan=True)
        colmadstd = mad_std(colsig[cxl:cxh],ignore_nan=True)
        rowmax = np.nanmax(rowsig[rxl:rxh])
        colmax = np.nanmax(colsig[cxl:cxh])
        rowmin = np.nanmin(rowsig[rxl:rxh])
        colmin = np.nanmin(colsig[cxl:cxh])

        if plotrow==True and plotcol==True:
            plt.subplot(2,1,1)
        if plotrow == True:
            plt.grid()
            if autoscale != True: plt.ylim(yl,yh)
            plt.xlim(rxl,rxh)
            plt.title(filename+'       Row ='+str(row)+'    Max ='+str(rowmax)+'    Min ='+str(rowmin)+\
                  '    Median ='+str(rowmedian)+'    std ='+str(rowstd)+'    mad_std ='+str(rowmadstd))
            plt.xlabel('column')
            plt.plot(xrow,rowsig,dotstring)

        if plotrow==True and plotcol==True:
            plt.subplot(2,1,2)
        if plotcol == True:
            plt.grid()
            if autoscale != True: plt.ylim(yl,yh)
            plt.xlim(cxl,cxh)
            plt.title(filename+'       Col ='+str(col)+'    Max ='+str(colmax)+'    Min ='+str(colmin)+\
                  '    Median ='+str(colmedian)+'    std ='+str(colstd)+'    mad_std ='+str(colmadstd))
            plt.xlabel('row')
            plt.plot(xcol,colsig,dotstring)

    interact(plotrowcol,row = (0,image.shape[0]-1,1),  col = (0,image.shape[1]-1,1), yl = (ylims[0],ylims[1],ystep), yh = (ylims[0],ylims[1],ystep),\
            rxl = (0,cols-1,xstep), rxh = (0,cols-1,xstep), cxl = (1,rows-1,xstep), cxh = (1,rows-1,xstep))

##-----------------------------------------------------------------------------------------------------------

## Based on https://scipy-cookbook.readthedocs.io/items/FittingData.html. Didn't work correctly in my application,
## I modified this one so x is associated with column number and y with row number. Seems to work now.
## What happens if there is a negative number in col or row? Should that ever be possible? If not, would
## it be better to leave out the np.abs and just let the algorithm fail? Or put in some error handling?
## If everything is positive-definite, there should be no problems.
## For now, go with the option of NOT using the abs().
## 190415. Modified to use np.nansum() instead of xxx.sum() to work with images with nans.
def moments(data):
    """Returns (height, x, y, halfwidth_x, halfwidth_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments."""
    total = np.nansum(data)
    Y, X = np.indices(data.shape)
#     print(Y)                       # Diagnostic.
#     print(X)                       # Diagnostic.
    x = np.nansum(X*data)/total
    y = np.nansum(Y*data)/total
    col = data[:, int(x)]
#     width_y = np.sqrt(np.nansum(np.abs(((np.arange(col.size)-y)**2)*col))/np.nansum(col))
    width_y = np.sqrt(np.nansum(((np.arange(col.size)-y)**2)*col)/np.nansum(col))           # Does't use abs().
    row = data[int(y), :]
#     width_x = np.sqrt(np.nansum(np.abs(((np.arange(row.size)-x)**2)*row))/np.nansum(row))
    width_x = np.sqrt(np.nansum(((np.arange(row.size)-x)**2)*row)/np.nansum(row))           # Does't use abs().
    height = data.max()
    return height, x, y, width_x, width_y

##------------------------------------------------------------------------------------------------------------
## Probably too short to bother with. Just saves one line and hides what's going on.
# def pdata(path, whichfile):
#     """Returns a PipeData object from whichfile, where whichfile is a
#     FITS file in directory 'path'."""
#     pd = PipeData(config = config)
#     pd.load(os.path.join(datapath,whichfile))
#     return pd

# ##------------------------------------------------------------------------------------------------------------
# This one may be useful enough to keep. Saves two lines but still hides what's going on.
def head(path, filename):
    """Prints out FITS header of whichfile, where 'filename' is a
    file in directory 'path'."""
    df = DataFits()
    df.loadhead(os.path.join(path,filename))
    return df.header

##----------------------------------------------------------------------------------------------------------------

def stackfits(whichpath,files):
    '''
    Make a stack of images, a list of headers from those images, and 1D arrays for medians and mad_stds.
    whichpath:   The path to the fits files.
    files:  A list of files.
    '''

    ## Load the header of the first file in the list "files".
    df = DataFits()                                                   # Create a DataPype DataFits io object.
    df.loadhead(os.path.join(whichpath,files[0]))                # Loads just the header of the first file.
    ## Make variables for the numbers of rows and columns.
    rows, cols = df.header['naxis2'], df.header['naxis1']
    print('rows =',rows,'   cols =',cols)

    image = np.zeros((len(files), rows,cols))  # 3D numpy array to hold the stack of images.
    imedian = np.zeros((len(files)))           # 1D numpy array to hold array medians.
    imad = np.zeros((len(files)))            # 1D numpy array to hold Median absolute deviations.

    headlist = []                              # Empty list to hold the headers.
    print('image.shape =', image.shape)

    for i in range(len(files)):

    # ########################################################################
        # Use this code block if you want to work with DataFits objects
        df = DataFits()
        df.load(os.path.join(whichpath,files[i]))
        image[i] = df.imageget() * 1.0            # Load image data into numpy arrays and convert to float.
        headlist.append(df.header)
        imedian[i] = np.nanmedian(image[i])
        imad[i] = mad_std(image[i],ignore_nan=True)
        print('')
        print(i, files[i])
    # #########################################################################

    #############################################################################
    #     # Use this code block if you want to use the standard astropy.io package.
    #     fitsfilename = os.path.join(datapath, files[i])    # Full path to a fitsfile.
    #     hdulist = fits.open(fitsfilename)                  # Open a fits file as an hdulist object.
    #     hdu0 = hdulist[0]                                  # Define a fits object as the 0th hdu in the fitsfile.
    #     image[i] = hdu0.data * 1.0                         # Define image in stack as float of data in the 0th hdu.
    #     headlist.append(hdu0.header)                       # Append the header of the fits object to the header list.
    #     print('')
    #     print(i,files[i])
    ###############################################################################
    return image, headlist, rows, cols, imedian, imad

##--------------------------------------------------------------------------------------------------------

def get_stats(image2D):
    imed = np.nanmedian(image2D)
    istd = np.nanstd(image2D)
    imad = mad_std(image2D, ignore_nan=True)
    imin = np.nanmin(image2D)
    imax = np.nanmax(image2D)
    print('median =',imed,'   std =',istd,'   mad =',imad)
    print('minimum = ',imin,'   maximum =',imax)
    return imed,istd,imad,imin,imax

##-------------------------------------------------------------------------------------------------------
def count_nans(img,printout=True):
    proxy = img.copy()
    nanmask = np.isnan(proxy)
    proxy[nanmask] = -1e10
    nans = np.where(proxy == -1e10)
    number = len(nans[0])
    if printout == True:
        print('number of nans =',number)
    return number
##---------------------------------------------------------------------------------------------------
def list_dir(path):
    allfiles = [f for f in os.listdir(path)]
    allfiles = sorted(allfiles)       ## This is necessary on my Mac, may not be for others?
    for i in range(len(allfiles)):
        print( i, allfiles[i])
##-------------------------------------------------------------------------------------------------
def load_image(path,filename):
    fname = os.path.join(path,filename)
    hdulist = fits.open(fname)
    hdulist.info()
    img = hdulist[0].data * 1.          # Read in the image and convert to float.
    header = hdulist[0].header
    print('image shape =',img.shape)
    return img, header
##------------------------------------------------------------------------------------------------
## Three functions for finding pixels with intensity values above, below, or between specified limits.
## They return the number of pixels and the corresponding mask array.

def greater_than(image, threshold, printout=False):
    '''Returns the number of pixels in image with intensity greater than
    a specified threshold. Also returns the corresponding mask array.'''

    img = image.copy()
    highmask = np.where(img > threshold)
    above = len(highmask[0])
    print('number of pixels above {} = {}'.format(threshold,above))
    return above, highmask

def less_than(image, threshold, printout=False):
    '''Returns the number of pixels in image with intensity less than
    a specified threshold. Also returns the corresponding mask array.'''

    img = image.copy()
    lowmask = np.where(img < threshold)
    below = len(lowmask[0])
    print('number of pixels below {} = {}'.format(threshold,below))
    return below, lowmask


def within(image, lower, upper, printout=False):
    '''Returns the number of pixels in image with intensity between upper
    and lower limits. Also returns the corresponding mask array.'''

    img = image.copy()
    lowmask = np.where(img <= lower)
    img[lowmask] = -1e100
    highmask = np.where(img >= upper)
    img[highmask] = -1e100
    img[highmask] = -1e100
    betweenmask = np.where(img != -1e100)
    between = len(betweenmask[0])
    print('number of pixels betweeen {} and {} = {}'.format(lower,upper,between))
    return between, betweenmask

##------------------------------------------------------------------------------------------------
def combine_masks(maskA, maskB, printout=False):
    '''Returns the number of elements in the union of the two sets
    and the combined mask.'''

    a = maskA
    b = maskB
    blen = len(b[0])
    alen = len(a[0])
    clen = len(b[0]) + len(a[0])
    c0 = np.zeros((clen),dtype=int)
    c1 = np.zeros((clen),dtype=int)
    c0[0:blen] = b[0]
    c1[0:blen] = b[1]

    count = 0
    for i in range(alen):
        total = 0
        for j in range(blen):
            if a[0][i] == b[0][j] and a[1][i] == b[1][j]:
                break
            else:
                total += 1
        if total == blen:
            c0[blen + count] = a[0][i]
            c1[blen + count] = a[1][i]
            count += 1
    if count == alen:
        c = (c0,c1)
    else:
        c0 = c0[:-(alen - count)]
        c1 = c1[:-(alen - count)]
        c = (c0,c1)
    clen = len(c[0])
    if printout == True:
        print('number in union of A and B =',clen)
        print('number in intersection of A and B =', alen + blen - clen)
    return clen, c
    ##-----------------------------------------------------------------------------------------
