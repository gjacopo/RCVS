#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
title:          rcvs.py
module:         RCVS (Raster Computer Vision Simplification)

summary:        Broad, general test of the cvapplier functionality. 
description:    Print a message stderr if something wrong.  

:CONTRIBUTORS:  grazzja
:CONTACT:       jacopo.grazzini@ec.europa.eu
:SINCE:         Fri May 31 10:20:51 2013
:VERSION:       0.9
"""

import os
from os import path as os_p
import re
import itertools, collections

TESTNAME = "TEST_RCVS"
TESTRCVS_DIR=os.path.dirname(os.path.realpath(__file__))
TESTRCVS_CASE='a.tif'
TESTRCVS_DEF_IFIELD=['img']
TESTRCVS_DEF_OFIELD=['img']

import numpy as np
import scipy

try:
    import gdal, osr
except ImportError:
    from osgeo import gdal, osr
    
from rios import applier, rioserrors, imagereader
# from rios import rcvs
import rcvs

try:                            
    import skimage
    from skimage import transform, exposure, feature, color, measure, segmentation
except ImportError:                         
    rcvs.raiseImportError('skimage')
    
try:                            
    import cv2
except ImportError:                         
    rcvs.raiseImportError('cv2')

try:                            
    from matplotlib import pyplot as plt
except ImportError:                         
    rcvs.raiseImportError('matplotlib')

try:    
    #debug: just to avoid annoying messages regarding 'undefined' variables
    KEY_FUNCTION,KEY_IN,KEY_OUT,KEY_ARGS,KEY_KWARGS,KEY_BLOCKPROCESS,KEY_RETURNARGS,KEY_MARGIN,\
        KEY_BLOCKFORMAT,NO_ARGUMENT,DEF_ARGUMENT = [None]*11
    for f in ('FUNCTION','IN','OUT','ARGS','KWARGS','BLOCKPROCESS','RETURNARGS','KEY_MARGIN',\
            'BLOCKFORMAT','NO_ARGUMENT','DEF_ARGUMENT'): 
        try:    exec "KEY_"+f+"="+"getattr(rcvs,'RCVS_KEY_"+f+"')"
        except: exec f+"="+"getattr(rcvs,'RCVS_"+f+"')"
    # ps: remember that methods, classes and even modules are all methods at the end...
    ##KEY_FUNCTION, KEY_IN, KEY_OUT, KEY_ARGS, KEY_KWARGS = [getattr(cvapplier,'RCVS_KEY_'+f) \
    ##    for f in ('FUNCTION','IN','OUT','ARGS','KWARGS','BLOCKPROCESS','RETURNARGS')]
except:
    raise IOError, 'unknown key variable found'

try:
    import riostestutils
    DEF_ROWS,DEF_COLS,DEF_PIXSIZE,DEF_DTYPE,DEF_XLEFT,DEF_YTOP,DEF_EPSG =                               \
        [None]*7
    for f in ('ROWS','COLS','PIXSIZE','DTYPE','XLEFT','YTOP','EPSG'): 
        exec "DEF_"+f+"="+"getattr(riostestutils,'DEFAULT_"+f+"')"
except:
    raise ImportError, 'error when importing riostestutils'

def run(**kwargs):
    """
    Run the test
    """
    riostestutils.reportStart(TESTNAME)

    #TEST_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../downloads/','samples/')
    TEST_DIR = 's:\\Data\\CID\\Developments\\grazzja\\downloads\\samples\\'
    #TEST_CASE = 'a.tif'
    TEST_CASE = 'vierzon.tif' #'checkerboard,tif' 
    #TEST_CASE = 'lena_color.png'
    img1 = TEST_DIR + TEST_CASE
    print 'testing image', img1
    #ramp1 = 'ramp1.tif'
    #ramp2 = 'ramp2.tif'
    #riostestutils.genRampImageFile(ramp1)
    #riostestutils.genRampImageFile(ramp2, reverse=True)
    #Display(img1, **kwargs)
    #print 'test_quickshift'
    #test_quickshift(img1, controls, **kwargs)
    #return
    #print 'test_identity'
    #test_identity(img1, controls, **kwargs)
    blocksz = 50; margin = 5 # int(blocksz/10)
    blockXsz=blocksz; blockYsz = blocksz
    #kwargs.update({KEY_BLOCK:True})
    if True:   kwargs.update({KEY_BLOCKPROCESS: 'single', KEY_MARGIN:0})
    else:       kwargs.update({KEY_BLOCKPROCESS: 'serial', 'XSize': blocksz, 'YSize': blocksz,
                       KEY_MARGIN: margin})
    #
#    testOpenCVPyramid = opencv_pyramid_down(**kwargs)
#    testOpenCVPyramid(img1)
#    return
#    testOpenCVMatch = opencv_template_matching(**kwargs)
#    testOpenCVMatch(img1)
#    return
#    testNormXCorr = external_norm_xcorr(**kwargs)
#    testNormXCorr(img1)
#    kwargs.pop('win_ext')
#    return
#    print 'test_rand'
#    test_rand(img1, **kwargs)
#    return
#    print 'test_slic'
#    kwargs.update({KEY_BLOCKPROCESS: 'single', KEY_MARGIN:0})
#    test_skimage_slic(img1, **kwargs)
#    return
    print 'test_ransac'
    if False:   kwargs.update({KEY_BLOCKPROCESS: 'single', KEY_MARGIN:0})
    else:      kwargs.update({KEY_BLOCKPROCESS: 'serial', 'XSize': blockXsz, 'YSize': blockYsz,
                       KEY_MARGIN: margin})
#    print '===========run test_skimage_ransac_from_web'
#    test_skimage_ransac_from_web(img1,**kwargs)
    kwargs.update({'win_ext': 5})
    print '===========run testSkimageRansac'
    testSkimageRansac = skimage_ransac_matching(**kwargs)
    testSkimageRansac(img1)
    #testSkimageRansac()
    return
    print 'test_slic'
    #kwargs = {KEY_BLOCKPROCESS:'single'}
    skimage_slic(img1, **kwargs)
    #ok = checkResult(slicfile)
    
##    # Clean up
##    for filename in [ramp1, ramp2, outfile]:
##        os.remove(filename)
    
    return #ok

#-----------------------------------------------------------------------------#

    
    
#/****************************************************************************/
class test_identity(collections.Callable):
    
    def __init__(self): pass
        
    def __call__(fname, controls=None, **otherKwargs):
        if controls is None:    
            controls = utils_generic.create_gtiff_controls(otherKwargs.pop('XSize',None),
                                                           otherKwargs.pop('YSize',None))
        if not utils_generic.check_filexists(fname):            raise IOError
        fname, oname = [fname], [utils_generic.create_filenames('id_', controls.ext, fname)]
        # create a copy of the input image
        outfile = utils_generic.test_function(lambda x:x, fname, oname,                 \
            ifields=['img'], ofields=['out'],                                  \
            otherArgs=None, controls=controls, **otherKwargs)
        # test indeed that we stored a copy of the input image, and nothing else
        ds = gdal.Open(fname)
        ok = utils_generic.test_equality(outfile, [ds.GetRasterBand(iband+1).ReadAsArray() \
            for iband in range(ds.RasterCount)])
        del ds
        if ok:                  riostestutils.report('identity', "passed")  
        else:                   riostestutils.report('identity', "array mismatch")  
        return oname
    
    
#/****************************************************************************/
class test_randblock(collections.Callable):
    
    def __init__(self): pass
        
    def __call__(fname, controls=None, **otherKwargs):
        if controls is None:    
            controls = utils_generic.create_gtiff_controls(otherKwargs.pop('XSize',None),
                                                           otherKwargs.pop('YSize',None))
        if not utils_generic.check_filexists(fname):            raise IOError
        fname, oname = [fname], [utils_generic.create_filenames('rand_', controls.ext, fname)]
        utils_generic.test_workflow(lambda x:np.random.random(x.shape), fname, oname,   \
            ifields=['img'], ofields=['out'],                                  \
            otherArgs=None, controls=controls, **otherKwargs)
        return oname


#/****************************************************************************/
# external_norm_xcorr: N-dimensional template search by normalized cross-correlation 
# or sum of squared differences; derived from original implementation from O.Alexandrov
# available at:
#   https://github.com/oleg-alexandrov/projects/blob/master/fft_match/norm_xcorr.py
# see also original code fft_matcher for block processing available at:
#   https://github.com/oleg-alexandrov/projects/blob/master/fft_match/fft_matcher.py
class external_norm_xcorr(collections.Callable):
    mod = 'norm_xcorr'     
    web = 'https://raw.github.com/oleg-alexandrov/projects/master/fft_match/norm_xcorr.py'

    @staticmethod
    def create_template(image, win_ext=20, info=None):
        # this function create a template from a single image
        # we suppose here that the data is in Gdal format 'zyx'
        try:        ysize, xsize = info.ysize, info.xsize
        except:     ysize, xsize = image.shape if image.ndim==2 else image.shape[1:]
        # define the center of the template window (avoiding getting outside of the 
        # image's boundaries)
        r, c = [win_ext+int(np.random.random() * s) for s in (ysize-2*win_ext, xsize-2*win_ext)]
        sY, sX, sZ = np.s_[r-win_ext:r+win_ext+1], np.s_[c-win_ext:c+win_ext+1], np.s_[:]
        if image.ndim==2:              template = image[sY, sX]
        else:                          template = image[sZ, sY, sX]
        return template, (r, c)
        
    @staticmethod
    def find_match(TM, image): 
        # TM = norm_xcorr.TemplateMatch(template,method='both')
        ncc, ssd = TM(image)
        # identify the center of optimal matched window with the extrema of the 
        # estimated feature
        nccloc = np.nonzero(ncc == ncc.max())
        ssdloc = np.nonzero(ssd == ssd.min())
        return ncc, ssd, nccloc, ssdloc

    @staticmethod
    def update_results(cnt, nccloc, ssdloc, margin):
        nccloc, ssdloc = [[loc[i]-margin for i in (0,1)] for loc in (nccloc, ssdloc)]
        print 'real location:     (y,x) = (%s)' %  [cnt[i] for i in (0,1)]
        print 'estimated locations: ncc = (%s)' %  [int(nccloc[i]) for i in (0,1)]
        print '                   : ssd = (%s)' %  [int(ssdloc[i]) for i in (0,1)] 
        return nccloc, ssdloc
        
    @staticmethod
    def display_results(image, ncc, ssd, nccloc, ssdloc, cnt, template, win_ext):  
        ysize, xsize = ncc.shape
        wsize = template.shape # [2*win_ext+1]*2
        Rectangle = lambda loc: plt.Rectangle(tuple([int(loc[i]-win_ext) for i in (1,0)]), 
                                              wsize[1], wsize[0], edgecolor='r', facecolor='none')
        fig, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2,num='ND Template Search')
        plt.gray()
        for ax  in (ax1,ax3,ax4):       ax.axis((0, xsize, ysize, 0)) #ax.axis('off')
        ax1.set_title('Search image'); ax1.plot(cnt[1],cnt[0],'r+'); 
        ax1.hold(True); ax1.imshow(image, interpolation='nearest')      
        ax1.add_patch(Rectangle(cnt)); ax1.hold(False)
        ax2.set_title('Template'); ax2.plot(win_ext,win_ext,'r+'); 
        ax2.hold(True); ax2.imshow(template, interpolation='nearest')
        ax2.hold(False)
        ax3.set_title('Normalized cross-correlation'); ax3.plot(nccloc[1],nccloc[0],'r+'); 
        ax3.hold(True); ax3.imshow(ncc, interpolation='nearest')
        ax3.add_patch(Rectangle(nccloc)); ax3.hold(False)
        ax4.set_title('Sum-of-squared differences'); ax4.plot(ssdloc[1],ssdloc[0],'r+'); 
        ax4.hold(True); ax4.imshow(ssd, interpolation='nearest')
        ax4.add_patch(Rectangle(ssdloc)); ax4.hold(False) 
        plt.show()

    def __init__(self, **kwargs): 
        utils_generic.import_module(self.mod, self.web) 
        import norm_xcorr
        # debug: 'complex256' is not understood in the machine cvapplier has been developed
        # _checkffttype is redefined prior to launching norm_xcorr.TemplateMatch
        try:
            norm_xcorr._checkffttype = lambda C: C if C.dtype in ['float32','float64','complex64','complex128'] \
                else np.float64(C) 
        except:
            raise IOError, 'impossible to redefine norm_xcorr._checkffttype'
        # define variables
        self.__isBlockProcess = kwargs.pop(KEY_BLOCKPROCESS,'single')
        self.__win_ext, margin = kwargs.pop('win_ext',20), kwargs.pop(KEY_MARGIN,0)  
        self.__margin = max(margin,self.__win_ext)
        self.__XSize, self.__YSize = kwargs.pop('XSize',None), kwargs.pop('YSize',None)
        # create the different jobs used for matching
        # we introduce a job for RGB2gray reduction using skimage module
        self.orig2gray = {KEY_FUNCTION: color.rgb2gray, KEY_IN: 'image', KEY_OUT: 'image', 
            KEY_BLOCKFORMAT: 'skimage'}
        # we add one additional job for creation of a random template
        self.template = {KEY_FUNCTION: self.create_template, KEY_IN: 'image', 
            KEY_ARGS: self.__win_ext, KEY_OUT: ['template', 'cnt']}
        # for the following jobs, we have to specify that data are requested in
        # scipy 'format' (shape of arrays) with {KEY_BLOCKFORMAT: 'yxz'}, as there
        # is no format specifically defined for the module norm_xcorr; note that in
        # order to describe this format, we can use indifferently (like in the 
        # following): the module itself (scipy), a string with its name ('scipy')
        # or a string of the shape 'yxz': they all correspond to the same fmtBlock!
        # we first apply TemplateMatch like in L.57
        self.match = {KEY_FUNCTION: norm_xcorr.TemplateMatch, KEY_OUT: 'TM', 
            KEY_KWARGS: {'method':'both'}, KEY_BLOCKFORMAT: scipy}
        # define the job for correlation matching  (L.58 to 60)
        self.find = {KEY_FUNCTION: self.find_match,
            KEY_OUT: ['ncc','ssd','nccloc','ssdloc'], KEY_BLOCKFORMAT: 'yxz'}
        # job for printing the results 
        self.results = {KEY_FUNCTION: self.update_results, KEY_OUT: ['nccloc','ssdloc']}    
        # job for displaying the results (L.62 to 76)
        self.display = {KEY_FUNCTION: self.display_results, KEY_OUT: NO_ARGUMENT,
            KEY_BLOCKFORMAT: 'scipy'}
        
    ## the following static method is introduced to update (reduce) the results
    ## of the block processing 
    
    @staticmethod
    def reduce_locations(loc, val, controls, operator):
        if isinstance(val,dict):
            iblocks = val.keys()
            # take the maximum value overall
            argextrema = np.argmax if operator=='max' else np.argmin
            newargmax = argextrema([val[iblock][loc[iblock][0],loc[iblock][1]] \
                for iblock in iblocks])
            ## # equivalent approach: 
            ##extrema = np.max if operator=='max' else np.min
            ##argextrema([extrema(feat[iblock]) for iblock in iblocks])
            # retrieve the corresponding key
            maxblock = iblocks[newargmax]
            # and the corresponding block position
            yblock = maxblock[0] * controls.windowysize 
            xblock = maxblock[1] * controls.windowxsize
            # update the location 
            loc = (loc[maxblock][0]+ yblock, loc[maxblock][1]+xblock)
        return loc
    
    def __call__(self, fname, controls=None):
        # run
        if controls is None:    
            controls = utils_generic.create_gtiff_controls(self.__XSize, self.__YSize)
        if not utils_generic.check_filexists(fname):        raise IOError
        # first single block processing: we create a gray block        
        if True:     intemplate, inimage = 'temp_gray', 'img_gray'
        else:           intemplate, inimage = 'template', 'image'
        prepare = (self.template, 
                    )
        if True:         
            orig2gray = self.orig2gray.copy()
            orig2gray.update({KEY_IN: 'template', KEY_OUT: intemplate})
            prepare = prepare + (orig2gray,) 
        otherKwargs = {KEY_BLOCKPROCESS: 'single', KEY_MARGIN: 0,
                            KEY_RETURNARGS: [intemplate,'TM','cnt']} 
        o_tm, outfiles = utils_generic.test_workflow(prepare, [fname], NO_ARGUMENT, 
            ifields='image', ofields=NO_ARGUMENT, otherArgs=None, controls=controls, 
            **otherKwargs)  
        # here is the main processing        
        #self.find.update({KEY_IN: 'image', KEY_ARGS: outargs_tm['TM']})
        mapper = (self.match.update({KEY_IN: NO_ARGUMENT, KEY_ARGS: o_tm[intemplate]}) or self.match, 
                  self.find.update({KEY_IN: ['TM',inimage]}) or self.find
                    )
        if True:         
            orig2gray = self.orig2gray.copy()
            orig2gray.update({KEY_IN: 'image', KEY_OUT: inimage})
            mapper = (orig2gray,) + mapper
        ifields, ofields = ['image'], ['ncc', 'ssd']
        oname = [utils_generic.create_filenames(n+'_', controls.ext, fname) \
            for n in ofields]
        # setup and run processing, possibly through block divide-and-conquer jobs
        otherKwargs.update({KEY_BLOCKPROCESS: self.__isBlockProcess, 
            KEY_RETURNARGS: self.find[KEY_OUT], KEY_MARGIN: self.__margin})     
        o, outfiles = utils_generic.test_workflow(mapper, [fname], oname, 
            ifields=ifields, ofields=ofields, otherArgs=None, controls=controls, 
            **otherKwargs)  
        # !!! to garantee the consistency of the results, the normalization of the SSD
        # (between 0 and 1) in the original norm_xcorr function should be deleted, hence
        # the following lines should be commented:
        #           #ssd -= ssd.min()
        #           #ssd /= ssd.max()
        # otherwise the values of the SSD computed over the different blocks won't be
        # comparable
        # Reducer: compute the max/min of NCC and SSD over the whole image
        if self.__isBlockProcess!='single':    # first update all the coordinates 
            o.update({'nccloc': self.reduce_locations(o['nccloc'], o['ncc'], controls, 'max')})
            o.update({'ssdloc': self.reduce_locations(o['ssdloc'], o['ssd'], controls, 'min')})
        # display the results using the stored variables previously returned as output
        present = (self.results.update({KEY_IN: NO_ARGUMENT, 
                    KEY_ARGS: [o_tm['cnt'],o['nccloc'],o['ssdloc'], self.__margin]})
                    or self.results,
                   self.display.update({KEY_IN: ['image','ncc', 'ssd','nccloc','ssdloc'],   
                    KEY_ARGS: [o_tm['cnt'], o_tm[intemplate], self.__win_ext]})
                    or self.display)
        otherKwargs.update({KEY_BLOCKPROCESS: 'single', KEY_MARGIN: 0}) 
        fname = [fname] + oname
        ifields += ofields 
        utils_generic.test_workflow(present, fname, NO_ARGUMENT, 
            ifields=ifields, ofields=NO_ARGUMENT, 
            otherArgs=NO_ARGUMENT, controls=controls, **otherKwargs)
        return


#/****************************************************************************/
# OpenCV based template matching derived from documentation available at:
#    http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
# see also OpenCV generic documentation and examples in python available at:
#    http://docs.opencv.org/trunk/doc/py_tutorials/py_tutorials.html
class opencv_template_matching(collections.Callable):
    mod = 'cv2'     
    web = 'http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html'

    @staticmethod
    def match_and_find(image, template, operators, methods): 
        feat = dict(zip(methods,[None] * len(methods)))
        val = feat.copy(); loc = feat.copy()
        for method in methods:
            meth = eval('cv2.' + method)
            # cv2.matchTemplate: see http://docs.opencv.org/modules/imgproc/doc/object_detection.html?highlight=matchtemplate#cv2.matchTemplate
            feat[method] = cv2.matchTemplate(image, template, meth)
            # y,x = np.unravel_index(result.argmax(), result.shape)
            # cv2.minMaxLoc: see http://docs.opencv.org/modules/core/doc/operations_on_arrays.html?highlight=minmaxloc#cv2.minMaxLoc
            mval, Mval, mloc, Mloc = cv2.minMaxLoc(feat[method])
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if operators[method]=='min': val[method], loc[method] = mval, mloc
            else:                       val[method], loc[method] = Mval, Mloc
        return feat, val, loc

    @staticmethod
    def update_results(cnt, loc, win_ext, margin, methods):
        print 'real location:     (y,x) = (%s)' %  [cnt[i] for i in (0,1)]
        for method in methods:
            # note that cnt is of the form (r,c), ie. (y,x) while cv2.minMaxLoc
            # returns the locations in (x,y) coordinates (hence '(1,0)' below)
            loc[method] = tuple([loc[method][i]-margin for i in (1,0)])
            # note that cv2.matchTemplate optimizes the location of the top left 
            # corner of the window to match (hence '+win_ext' below)
            print 'estimated locations: %s = (%s)' %  (method,[int(loc[method][i]+win_ext) for i in (0,1)])
        return loc

    @staticmethod
    def display_results(image, loc, template, cnt, result, win_ext, methods):  
        Rectangle = lambda loc: plt.Rectangle(loc, w, h, edgecolor='r', facecolor='none')
        h, w = template.shape if template.ndim==2 else template.shape[:2] 
        ysize, xsize = image.shape if image.ndim==2 else image.shape[:2]
        nsub = len(methods)
        fig, axes = plt.subplots(2,int(nsub/2)+1, num='Template Matching')
        axes = list(itertools.chain.from_iterable(axes))
        for ax  in axes:       ax.axis((0, xsize, ysize, 0)) and ax.axis('off')
        axes[0].set_title('Search image', fontsize=8);  axes[0].plot(cnt[1],cnt[0],'r+')
        axes[0].hold(True), axes[0].imshow(image, interpolation='nearest')
        axes[0].add_patch(Rectangle(tuple([cnt[j]-win_ext for j in (1,0)])))
        axes[4].set_title('Template', fontsize=8); axes[4].axis((0, w, h, 0)); axes[4].plot(win_ext,win_ext,'r+')   
        axes[4].hold(True), axes[4].imshow(template, interpolation='nearest')
        #print 'arriva qui', result
        for i in xrange(len(methods)):
            method = methods[i]
            topL = loc[method][::-1]
            cnt_method = [int(topL[j]+win_ext) for j in (0,1)]
            # the location marked by the red point is the one with the highest value, 
            # so that location (the rectangle formed by that point as a corner and 
            # width and height equal to the patch image) is considered the match.
            i = i+1 if i<3 else i+2
            axes[i].hold(True);  axes[i].plot(cnt_method[0],cnt_method[1],'r+') 
            # bottomR = (topL[0] + w, topL[1] + h)
            # cv2.rectangle(result[method], topL, bottomR, 255, 2) 
            axes[i].imshow(result[method], interpolation='nearest',cmap = 'gray'), 
            axes[i].add_patch(Rectangle(topL)); axes[i].hold(False)
            axes[i].set_title(method, fontsize=8)
        plt.show()        
        
    def __init__(self, **kwargs): 
        utils_generic.import_module(self.mod) # we won't go any further 
        # define variables
        self.__isBlockProcess = kwargs.pop(KEY_BLOCKPROCESS,'single')
        self.__win_ext, margin = kwargs.pop('win_ext',20), kwargs.pop(KEY_MARGIN,0)  
        self.__margin = max(margin,self.__win_ext)
        self.__XSize, self.__YSize = kwargs.pop('XSize',None), kwargs.pop('YSize',None)
        self.__methods = ['TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR',
                    'TM_CCORR_NORMED', 'TM_SQDIFF', 'TM_SQDIFF_NORMED']
        self.__operator = {'TM_CCOEFF':'max', 'TM_CCOEFF_NORMED':'max', 'TM_CCORR':'max',
                    'TM_CCORR_NORMED':'max', 'TM_SQDIFF':'min', 'TM_SQDIFF_NORMED':'min'}
        # define methods
        self.template = {KEY_FUNCTION: external_norm_xcorr.create_template, KEY_IN: 'image', 
            KEY_ARGS: self.__win_ext, KEY_OUT: ['template', 'cnt']}
        self.matchandfind = {KEY_FUNCTION: self.match_and_find, KEY_OUT: ['feat','val','loc'], 
            KEY_BLOCKFORMAT: 'cv2'}
        self.results = {KEY_FUNCTION: self.update_results, KEY_OUT: 'loc'}
        self.display = {KEY_FUNCTION: self.display_results, KEY_OUT: NO_ARGUMENT}
        return

    @staticmethod  # see external_norm_xcorr.reduce_locations
    def reduce_locations(loc, val, controls, operators, methods):
        newloc = None
        if isinstance(loc[loc.keys()[0]], dict):
            iblocks = loc.keys()
            newloc = dict(zip(methods,[None] * len(methods)))
            for method in methods:
                # newloc.update({method: dict([(k,v[method]) for k, v in loc.items()])})    
                argextrema = np.argmax if operators[method]=='max' else np.argmin
                newargmax = argextrema([val[iblock][method] for iblock in iblocks])
                maxblock = iblocks[newargmax]
                yblock = maxblock[0] * controls.windowysize 
                xblock = maxblock[1] * controls.windowxsize
                # note that cv2.minMaxLoc returns the locations in (x,y) coordinates
                newloc.update({method: (loc[maxblock][method][0]+ xblock, 
                                        loc[maxblock][method][1]+yblock)})
        return newloc or loc

    @staticmethod 
    def reduce_features(feat, controls, win_ext, margin, methods):
        newfeat = None
        pad = win_ext - margin
        if isinstance(feat[feat.keys()[0]], dict):
            iblocks = feat.keys()
            ytotalblocks, xtotalblocks = [max([iblock[i] for iblock in iblocks])+1 for i in (0,1)]
            newfeat = dict(zip(methods,[None] * len(methods)))
            for method in methods:
                x = np.vstack(tuple([np.hstack(tuple([feat[(y,x)][method]
                                    for x in xrange(xtotalblocks)]))
                            for y in xrange(ytotalblocks)]))
                if pad>0:
                    try:    
                        newfeat[method] = np.pad(x, win_ext, 'constant', constant_values=0)
                    except: 
                        newfeat[method] = np.zeros(np.add(x.shape,2*win_ext))
                        newfeat[method][win_ext:-win_ext,win_ext:-win_ext] = x
                else:
                    newfeat[method] = x
        return newfeat or feat
                                                 
    def __call__(self, fname, controls=None):
        if controls is None:    
            controls = utils_generic.create_gtiff_controls(self.__XSize, self.__YSize)
        if not utils_generic.check_filexists(fname):        raise IOError
        otherKwargs = {KEY_BLOCKPROCESS: 'single', KEY_MARGIN: 0,
            KEY_RETURNARGS: ['cnt','template']}
        o_tm, outfiles = utils_generic.test_workflow(self.template, [fname], NO_ARGUMENT, 
            ifields=['image'], ofields=NO_ARGUMENT, otherArgs=None, controls=controls, 
            **otherKwargs) 
        # the output template is in 'gdal' format ('zyx'); we convert it in 'cv2'
        # format ('yxz') for use by the cv2 derived functions
        o_tm['template'] = rcvs.Format.toAxis(o_tm['template'], 'yxz')
        otherKwargs.update({KEY_BLOCKPROCESS: self.__isBlockProcess,
            KEY_RETURNARGS: self.matchandfind[KEY_OUT], KEY_MARGIN: self.__margin})
        self.matchandfind.update({KEY_IN: 'image', 
            KEY_ARGS: (o_tm['template'], self.__operator,self.__methods)})
        o, outfiles = utils_generic.test_workflow(self.matchandfind, [fname], NO_ARGUMENT, 
            ifields=['image'], ofields=NO_ARGUMENT, otherArgs=None, controls=controls, 
            **otherKwargs)  
        if self.__isBlockProcess!='single':    # first update all the coordinates 
            o['loc'] = self.reduce_locations(o['loc'], o['val'],  controls, self.__operator, self.__methods)
            o['feat'] = self.reduce_features(o['feat'], controls, self.__win_ext, self.__margin, self.__methods)
            #o['feat'].update({method: self.reduce_feature(o['feat'][method])})
        self.results.update( {KEY_IN: NO_ARGUMENT, 
            KEY_ARGS: (o_tm['cnt'], o['loc'], self.__win_ext, self.__margin, self.__methods)})
        self.display.update({KEY_IN: ['image','loc'],   
            KEY_ARGS: [o_tm['template'],o_tm['cnt'], o['feat'], self.__win_ext, self.__methods],
            KEY_BLOCKFORMAT: 'cv2'})
        otherKwargs.update({KEY_BLOCKPROCESS: 'single', KEY_MARGIN: 0}) 
        utils_generic.test_workflow((self.results,self.display), [fname], NO_ARGUMENT, 
            ifields=['image'], ofields=NO_ARGUMENT, 
            otherArgs=NO_ARGUMENT, controls=controls, **otherKwargs)

#/****************************************************************************/
# Robust matching using RANSAC - example from:
class skimage_ransac_matching(object):
    mod = 'skimage'     
    web = 'http://scikit-image.org/docs/dev/auto_examples/plot_matching.html'

    # define the function for checkerboard generation that correspond to L.38 to 
    # 45 of original code
    @staticmethod
    def generate_synthetic(): 
        from skimage import data
        checkerboard = skimage.util.img_as_float(data.checkerboard())
        img_orig = np.zeros(list(checkerboard.shape) + [3])
        img_orig[..., 0] = checkerboard
        gradient_r, gradient_c =                                               \
            np.mgrid[0:img_orig.shape[0],0:img_orig.shape[1]] / float(img_orig.shape[0])
        img_orig[..., 1], img_orig[..., 2] = gradient_r, gradient_c
        return img_orig 

    # define the find_correspondences based on the 'on-the-fly' calculations of the
    # original code: a static function created from copy/paste of L.65 to 102
    @staticmethod
    def find_correspondences(orig, warped, coords, coords_orig_subpix, \
                             coords_warped, coords_warped_subpix, win_ext=5):
        # use the functions gaussian_weights and match_corner already defined in 
        # the original code (L.65 to 92)
        def gaussian_weights(win_ext, sigma=1):
            y, x = np.mgrid[-win_ext:win_ext+1, -win_ext:win_ext+1]
            g = np.zeros(y.shape, dtype=np.double)
            g[:] = np.exp(-0.5 * (x**2 / sigma**2 + y**2 / sigma**2))
            g /= 2 * np.pi * sigma * sigma
            return g
        def match_corner(orig, warped, coords, coords_warped, coords_warped_subpix, 
                         Gaussian, win_ext):
            global test
            r,c = [int(v) for v in np.round(coords)]
            win_orig = orig[r-win_ext:r+win_ext+1,c-win_ext:c+win_ext+1, :]
            while np.prod(win_orig.shape[:2])!=(2*win_ext+1)**2:
                win_ext -= 1
                win_orig = orig[r-win_ext:r+win_ext+1,c-win_ext:c+win_ext+1, :]
            if win_ext not in Gaussian: Gaussian.update({win_ext: gaussian_weights(win_ext, 3)})
            weights = Gaussian[win_ext]
            weights = np.dstack((weights, weights, weights))
            SSDs, SSD = [], 0
            for cr, cc in coords_warped:
                win_warped = warped[cr-win_ext:cr+win_ext+1,cc-win_ext:cc+win_ext+1, :]
                if win_orig.shape!=win_warped.shape:        continue
                SSD = np.sum(weights * (win_orig - win_warped)**2)
                SSDs.append(SSD)
            # we add here some furhter testing in the case the set of matched points
            # is empty
            if SSDs!=[]:                return coords_warped_subpix[np.argmin(SSDs)]
            else:                       return []
        # apply those functions for finding correspondences like in L.95 to 103
        Gaussian = {} # introduced to avoid computing several times the same thing...
        src, dst = [], []        
        for coords in coords_orig_subpix:# we added some control wrt the original code
            if any(np.isnan(coords)):                       continue
            matches = match_corner(orig, warped, coords, coords_warped, coords_warped_subpix, 
                                   Gaussian, win_ext)
            if matches==[] or any(np.isnan(matches)):       continue 
            src.append(coords)
            dst.append(matches)
        return np.array(src), np.array(dst)
        
    # print the results like in L.118 to 121
    @staticmethod
    def update_results(tform, model, model_robust):
        # compare "true" and estimated transform parameters
        print(tform.scale, tform.translation, tform.rotation)
        print(model.scale, model.translation, model.rotation)
        print(model_robust.scale, model_robust.translation, model_robust.rotation)
        return
    
    # create a static display function following L.24 to 148
    @staticmethod
    def display_results(img_orig_gray, img_warped_gray, coords, coords_warped, src, dst, inliers, outliers):  
        if img_orig_gray.ndim>img_warped_gray.ndim:     img_orig_gray = img_orig_gray[:,:,1]
        elif img_warped_gray.ndim>img_orig_gray.ndim:   img_warped_gray = img_warped_gray[:,:,1]
        ysize, xsize = img_orig_gray.shape if img_orig_gray.ndim==2 else img_orig_gray.shape[:2]
        img_combined = np.concatenate((img_orig_gray, img_warped_gray), axis=1)
        fig, ax = plt.subplots(nrows=3, ncols=1, num='Ransac matching')
        if img_warped_gray.ndim==2:         plt.gray()
        ax[0].set_title('key points', fontsize=8), ax[0].imshow(img_combined, interpolation='nearest')   
        ax[0].plot(coords[:, 1], coords[:, 0], '.', markersize=10, color='r')
        ax[0].plot(coords_warped[:, 1] + xsize, coords_warped[:, 0], '.', markersize=10, color='r')
        ax[1].set_title('correct correspondences', fontsize=8), ax[1].imshow(img_combined, interpolation='nearest')   
        ax[2].set_title('faulty correspondences', fontsize=8), ax[2].imshow(img_combined, interpolation='nearest')
        for idx, a in enumerate(ax): 
            ax[idx].axis('off'); ax[idx].axis((0, 2*xsize, ysize, 0))
        for idx, (m, c) in enumerate(((inliers, 'g'), (outliers, 'r'))):
            ax[idx+1].plot((src[m, 1], dst[m, 1] + xsize), (src[m, 0], dst[m, 0]), '-', color=c)
            ax[idx+1].plot(src[m, 1], src[m, 0], '.', markersize=10, color=c)
            ax[idx+1].plot(dst[m, 1] + xsize, dst[m, 0], '.', markersize=10, color=c)
        plt.show() # visualize correspondences

    ## define the basic jobs

    def __init__(self, **kwargs): 
        utils_generic.import_module(self.mod) 
        # define variables
        self.__isBlockProcess = kwargs.pop(KEY_BLOCKPROCESS,'single')
        self.__win_ext, self.__margin = kwargs.pop('win_ext',5), kwargs.pop(KEY_MARGIN,0)  
        #if self.__isBlockProcess!='single': 
        self.__margin = max(self.__margin,self.__win_ext)
        self.__XSize, self.__YSize = kwargs.pop('XSize',None), kwargs.pop('YSize',None)
        self.__scale, self.__rotation, self.__translation = \
            kwargs.pop('scale',(0.9, 0.9)), kwargs.pop('rot',0.2), kwargs.pop('trans',(20, -10))
        # define a function for intensity rescaling of the input image; this ensures
        # further appropriate calculations 
        self.rescale ={KEY_FUNCTION: lambda x: exposure.rescale_intensity(1.*x), 
                       KEY_IN: 'img_orig', KEY_OUT: 'img_orig'}
        # define the process for gray image generation: L.46 of original code
        self.orig2gray = {KEY_FUNCTION: color.rgb2gray, KEY_IN: 'img_orig', KEY_OUT: 'img_orig_gray'}
        # note: no need to specify the use of the skimage format ('yxz') as it will recognise the module
        # from the function prototype        
        self.__tform = transform.AffineTransform(scale=self.__scale, rotation=self.__rotation, 
                                          translation=self.__translation)
        # define the warping procedure: L.48 to 50
        #warped = {KEY_FUNCTION: transform.warp, KEY_IN: 'img_orig', KEY_OUT: 'img_warped', 
        #    KEY_ARGS: tform.inverse}
        self.warp = {KEY_FUNCTION: lambda x,t: skimage.transform.warp(x, t.inverse), 
            KEY_IN: ['img_orig'], KEY_ARGS: self.__tform, KEY_OUT: 'img_warped', KEY_BLOCKFORMAT: 'skimage'}          
        # ibid for gray conversion: L.51
        self.warp2gray = {KEY_FUNCTION: color.rgb2gray, KEY_IN: 'img_warped', KEY_OUT: 'img_warped_gray'}        
        # note that instead of defining both warped and warp2gray processes, it is
        # possible to define one single process warpedgray; in that case, it is necessary
        # to specify the format used by the method (key KEY_BLOCKFORMAT) as this can't be 
        # recognised from the lambda function (key KEY_FUNCTION) 
        self.__warpgray = {KEY_IN: 'img_orig', KEY_OUT: 'img_warped_gray', 
            KEY_FUNCTION: lambda x,*a,**kw: skimage.color.rgb2gray(skimage.transform.warp(x,*a,**kw)),
            KEY_ARGS: self.__tform.inverse, KEY_BLOCKFORMAT: 'skimage' # force the use of skimage format
            }
        # define the process for extracting the coordinates of feature points (L.53 to 62)
        # as a sequence of jobs (op_* repeated over both original and warped images
        # (v_orig and v_warp)
        harris = {KEY_FUNCTION: feature.corner_harris, KEY_IN: None, KEY_OUT: None}
        peak = {KEY_FUNCTION: feature.corner_peaks, KEY_IN: None, KEY_OUT: None,
            KEY_KWARGS: {'threshold_rel': 0. if self.__margin else 0.001,'min_distance': 5}} 
        subpix = {KEY_FUNCTION: feature.corner_subpix, KEY_IN: None, KEY_OUT: None,
            KEY_KWARGS: {'window_size': 9}}
        v_orig = ('img_orig_gray','coords','coords_orig_subpix')
        v_warp = ('img_warped_gray','coords_warped','coords_warped_subpix')
        self.features = list([harris.update({KEY_IN:i, KEY_OUT:c})                              \
           or peak.update({KEY_IN:c, KEY_OUT:c}) or  subpix.update({KEY_IN:[i,c], KEY_OUT:s})   \
           or (harris.copy(),peak.copy(),subpix.copy()) for i,c,s in (v_orig,v_warp)])
        # correspondences process
        self.correspondences = {KEY_FUNCTION: self.find_correspondences,
            KEY_IN: ['img_orig','img_warped','coords','coords_orig_subpix','coords_warped','coords_warped_subpix'],
            KEY_OUT: ['src', 'dst'], KEY_KWARGS: {'win_ext': self.__win_ext},
            KEY_BLOCKFORMAT: 'skimage'} # we obviously need input Array in 'skimage' format
        # both coords and correspondences methods are obviously the most time consuming 
        # jobs in performing image matching: those should be performed through a
        # multithreading approach
        # set the model (L.106 to 107): no input argument is explicitely set as a (tuple
        # of) empty tuple ((),)
        self.model = {KEY_FUNCTION: transform.AffineTransform, KEY_IN: NO_ARGUMENT, KEY_OUT: 'model',
            KEY_ARGS: NO_ARGUMENT}
        # perform standard fitting likewise L.109 to 110                     
        self.estimate = {KEY_FUNCTION: lambda x, *a: x.estimate(*a)}
        # we could also have written:
        ## @staticmethod
        ## def estimate(model,src,dst):  return model.estimate(src,dst)
        ## estimate = {KEY_FUNCTION: estimate.__func__, KEY_IN: ['model','src', 'dst']} 
        # note that the item {KEY_BLOCKFORMAT: KEY_NOFORMAT} could be added for clarity, but it is
        # in fact unnecessary here as this is the default rule 
        # perform robust RANSAC fitting likewise L.113 to 115                     
        self.ransac = {KEY_OUT: ['model_robust', 'inliers'],
            KEY_FUNCTION: lambda src, dst, *a, **kw: skimage.measure.ransac((src, dst), *a, **kw),
            KEY_KWARGS: {'min_samples': 3,'residual_threshold': 2, 'max_trials': 100}}
        # define the output outliers
        self.outliers = {KEY_FUNCTION: lambda x: x == False, KEY_IN: 'inliers', KEY_OUT: 'outliers'}
        # displaying the results
        self.results = {KEY_FUNCTION: self.update_results, KEY_OUT: NO_ARGUMENT}
        self.display = {KEY_FUNCTION: self.display_results, KEY_OUT: NO_ARGUMENT,
            KEY_BLOCKFORMAT: 'skimage'}

    @staticmethod
    def reduce_locations(coord, controls):
        if isinstance(coord,dict):
            for iblock in coord.keys():                
                if coord[iblock].size==0:       continue # with next iteration
                yblock = iblock[0] * controls.windowysize
                xblock = iblock[1] * controls.windowxsize
                coord[iblock] += [yblock,xblock]
            # create a single set of updated coordinates, discarding the non-matched
            coord = reduce(lambda x, y: np.vstack((x,y)), 
                           [c for c in coord.values() if c.size!=0])
        return coord

    def __call__(self, fname=None, controls=None): 
        # set controls if None
        controls = controls or utils_generic.create_gtiff_controls(self.__XSize, self.__YSize)
        # initialise the input/output names, input/output fields
        # check the input original image
        if fname is None:
            fname = utils_generic.create_filenames('checkerboard', controls.ext, '')
        if not utils_generic.check_filexists(fname):
            utils_generic.create_testfile(fname, self.generate_synthetic())
        # the file contained in fname is the input file of the whole process that must exist:
        # see the original; all other data can be derived from it
        fname, oname = [fname], [utils_generic.create_filenames('warp_', controls.ext, fname)] 
        ifields, ofields = ['img_orig'], ['img_warped']
        otherKwargs = {}
        # intialise the workflow as a list of processes for creating or loading the input
        # original and warped images
        workflow = [self.rescale, self.warp]
        # check if the warp image already exists or if it needs to be created (hence 
        # the job for its creation cancelled from the workflow or not)
        if not utils_generic.check_filexists(oname[0]):
            # note that in the case  isBlockProcess=='single', all jobs could be
            # in fact added already to the workflow and processed at once; we divide the 
            # processing in different phases for clarity
            # specify some of the arguments calculated that are later needed for computation 
            otherKwargs.update({KEY_BLOCKPROCESS: self.__isBlockProcess, KEY_MARGIN: self.__margin})
            # first single block processing     
            utils_generic.test_workflow(workflow, fname, oname, ifields=ifields, ofields=ofields, 
                otherArgs=None, controls=controls, **otherKwargs)  
        # further process
        # first add the warped image to the list of input data 
        fname += oname; ifields += ofields
        oname = [utils_generic.create_filenames(n+'_', controls.ext, fname[0])     \
            for n in ('gray', 'warpgray')]
        ofields = ['img_orig_gray', 'img_warped_gray']
        # main core processing: this is where the workflow can benefit from block processing
        workflow = rcvs.Format.flattenIterable([self.orig2gray, 
                                                     self.warp2gray])
        # update the input/output data and the workflow depending on what exists already
        i = 0
        print 'first', fname, oname
        print ifields, ofields
        while i<len(oname): 
            if not utils_generic.check_filexists(oname[i]):                 i += 1
            else: fname += [oname.pop(i)]; ifields += [ofields.pop(i)]; workflow.pop(i)
        if oname!=[]:
            print fname, oname
            print ifields, ofields
            otherKwargs.update({KEY_BLOCKPROCESS: 'single', KEY_MARGIN: 0})                         
            utils_generic.test_workflow(workflow, fname, oname, ifields=ifields, ofields=ofields, 
                otherArgs=None, controls=controls, **otherKwargs)  
        # main core processing: this is where the workflow can benefit from block processing
        fname += oname; ifields += ofields
        workflow = rcvs.Format.flattenIterable([self.rescale,
                                                     self.features, 
                                                     self.correspondences])
        otherKwargs.update({KEY_BLOCKPROCESS: self.__isBlockProcess, KEY_MARGIN: self.__margin,     \
            KEY_RETURNARGS: ['src','dst','coords','coords_warped']})                         
        outargs, dummy = utils_generic.test_workflow(workflow, fname, NO_ARGUMENT, 
            ifields=ifields, ofields=NO_ARGUMENT, otherArgs=None, controls=controls, 
            **otherKwargs)  
        # reducing operation: update the output coords in the case block processing has
        # been adopted
#        print 'OUTPUT'
#        print [outargs[key] for key in ('src','dst','coords','coords_warped')]
        if self.__isBlockProcess!='single':                                                                                 
            [outargs.update({key: self.reduce_locations(outargs[key], controls)})   \
                for key in ('src','dst','coords','coords_warped')]
        # further process (analyze) in the case of block processing: this is to adjust
        # from the block processing consequences
        self.estimate.update({KEY_IN: 'model', KEY_ARGS: (outargs['src'], outargs['dst'])})
        self.ransac.update({KEY_IN: NO_ARGUMENT,                              \
            KEY_ARGS: (outargs['src'], outargs['dst'], transform.AffineTransform)})
        # final estimation: the results need to be analysed all together   
        workflow = rcvs.Format.flattenIterable([self.model, 
                                                       self.estimate, 
                                                       self.ransac,
                                                       self.outliers
                                                       ])
        # reset the processing format to 'single' in otherKwargs
        otherKwargs.update({KEY_BLOCKPROCESS: 'single',                         \
            KEY_RETURNARGS: ['model', 'model_robust','inliers', 'outliers']})                         
        #otherKwargs.update({'debug':True})                   
        outargs_model, dummy = utils_generic.test_workflow(workflow, 
            NO_ARGUMENT, NO_ARGUMENT, ifields=NO_ARGUMENT, ofields=NO_ARGUMENT, 
            otherArgs=NO_ARGUMENT, controls=controls, **otherKwargs)  
        # display the results using the stored variables previously returned as output
        self.results.update({KEY_IN: NO_ARGUMENT,
            KEY_ARGS: [self.__tform, outargs_model['model'], outargs_model['model_robust']]})
        self.display.update({KEY_ARGS: [outargs['coords'], outargs['coords_warped'],
            outargs['src'], outargs['dst'], outargs_model['inliers'], outargs_model['outliers']],
            KEY_IN: ['img_orig_gray','img_warped_gray']})
        otherKwargs.update({KEY_BLOCKPROCESS: 'single'}) 
        # note that the following update can be in fact 
        fname, ifields = list(fname+oname)[-2:], ['img_orig_gray','img_warped_gray']       
        utils_generic.test_workflow((self.results,self.display), fname, NO_ARGUMENT, 
            ifields=ifields, ofields=NO_ARGUMENT, 
            otherArgs=NO_ARGUMENT, controls=controls, **otherKwargs)
        return oname


#/****************************************************************************/
# SLIC segmentation - example from:
#   
class skimage_slic(collections.Callable):
    mod = 'skimage'     
    web = 'http://scikit-image.org/docs/dev/auto_examples/plot_segmentations.html'
    
    def __init__(self): 
        # define the different required steps of the SLIC segmentation algorithm
        self.asfloat = {KEY_FUNCTION:'img_as_float', KEY_IN:'img', KEY_OUT:'img'}
        self.asrgb = {KEY_FUNCTION: lambda x: np.dstack((x,x,x)) if x.ndim==2 else (x[:,:,:3] if x.shape[2]>3 else x),
            KEY_IN:'img', KEY_OUT:'img'}  
        self.sample = {KEY_FUNCTION: lambda x: x[::2, ::2], KEY_IN:'img', KEY_OUT:'img'}
        self.slic = {KEY_FUNCTION: skimage.segmentation.slic, KEY_IN:'img',  KEY_OUT:'label', 
            KEY_KWARGS:{'compactness':10, 'n_segments':250, 'sigma':1.}}
        # note that the prototype for the slic function is:
        #       skimage.segmentation.slic(image, n_segments=100, compactness=10.0,
        #               max_iter=10, sigma=None, spacing=None, multichannel=True, 
        #               convert2lab=True, ratio=None)
        # it was also possible to write it using:
        # from skimage import segmentation
        # slic = {KEY_FUNCTION: segmentation.slic, KEY_IN:'img',  KEY_OUT:'label', 
        #              KEY_KWARGS:{'compactness ':10, 'n_segments':250, 'sigma':1.}}    
        self.boundaries = {KEY_FUNCTION: 'skimage.mark_boundaries',
            KEY_IN:['img', 'label'], KEY_OUT:'out'}
        self.imshow = {KEY_FUNCTION: 'matplotlib.imshow', KEY_IN:'out', KEY_OUT: NO_ARGUMENT}
        return
        
    def __call__(self, fname, controls=None, **otherKwargs):
        if controls is None:    
            controls = utils_generic.create_gtiff_controls(otherKwargs.pop('XSize',None),
                                                           otherKwargs.pop('YSize',None))
        if not utils_generic.check_filexists(fname):        raise IOError
        # define the name of the output file
        oname = [utils_generic.create_filenames('slic_', controls.ext, fname)]
        fname = [fname]
        # defrine the whole processing workflow
        workflow = (# first transform the data in the correct float format to be processed
                    self.asfloat, 
                    # ensure that a 3-bands RGB image is passed as input 
                    self.asrgb,
                    # perform simple subsampling to reduce complexity
                    self.sample,
                    # then perform the actual segmentation
                    self.slic,
                    # mark the segments' boundaries over the input image
                    self.boundaries,
                    # display the results
                    self.imshow
                    )
        # apply the workflow on the desired image
        outargs, outfiles = utils_generic.test_workflow(workflow, fname, oname, \
            ifields=['img'], ofields=['out'],                          \
            otherArgs=None, controls=controls, **otherKwargs)  
        return oname


#/****************************************************************************/
class skimage_canny(collections.Callable):
    
    def __init__(self): pass
        
    def __call__(fname, controls, **kwargs):
        from skimage import filter
        oname = utils_generic.create_filenames('canny_', controls.ext, fname)
        workflow = {KEY_FUNCTION:filter.canny, KEY_KWARGS:{'sigma':1.}}
        utils_generic.test_workflow(workflow, fname, oname, controls, **kwargs)
        return oname

#/****************************************************************************/
# OpenCV based multiresolution Gaussian pyramid decomposition available at:
#    http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_pyramids/py_pyramids.html#py-pyramids
# see also OpenCV generic documentation and examples in python available at:
#    http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_table_of_contents_imgproc/py_table_of_contents_imgproc.html
class opencv_pyramid_down(collections.Callable):
    mod = 'cv2'
    web = 'http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_pyramids/py_pyramids.html'
        
    @staticmethod 
    def prepare_pyramid(nlevels, info):   
        ysize, xsize = info.ysize, info.xsize
        # first adjust the nlevels
        size = min(ysize, xsize)
        nlevels = min(nlevels, (size-1).bit_length())
        
    @staticmethod 
    def compute_pyramid(image, nlevels):   # generate Gaussian pyramid
        # start building the pyramid
        gpA = []
        for i in xrange(nlevels):
            image = cv2.pyrDown(image)
            gpA.append(image)
        return tuple(gpA)

    @staticmethod
    def display_pyramid(image, nlevels, *args): 
        nlevels = min(min(nlevels,len(args)),5)
        fig, axes = plt.subplots(1, nlevels+1,num='Pyramid Downdecomposition')
        args = (image,) + args
        for i, ax  in enumerate(axes):    
            image = args[i]
            ysize, xsize = image.shape if image.ndim==2 else image.shape[:2]
            ax.axis((0, xsize, ysize, 0)) and ax.axis('off')
            if i>0:     axes[i].set_title('level '+str(i), fontsize=8)
            else:       axes[0].set_title('original', fontsize=8)
            axes[i].imshow(image, interpolation='nearest')
        plt.show()        
        
    def __init__(self, **kwargs): 
        utils_generic.import_module(self.mod) # we won't go any further 
        # define variables
        self.__isBlockProcess = kwargs.pop(KEY_BLOCKPROCESS,'single')
        self.__margin = kwargs.pop(KEY_MARGIN,0)  
        self.__nlevels = kwargs.pop('nelevels',2)  
        self.__XSize, self.__YSize = kwargs.pop('XSize',None), kwargs.pop('YSize',None)
        self.pyrdown = {KEY_FUNCTION: self.compute_pyramid, KEY_IN: 'image', 
            KEY_ARGS: self.__nlevels, KEY_BLOCKFORMAT: 'cv2'}
        self.display = {KEY_FUNCTION: self.display_pyramid, KEY_IN: 'image', 
            KEY_BLOCKFORMAT: 'cv2'}
                                                 
    def __call__(self, fname, controls=None):
        if controls is None:    
            controls = utils_generic.create_gtiff_controls(self.__XSize, self.__YSize)       
        if not utils_generic.check_filexists(fname):        raise IOError
        ofields = ['pyr'+str(n) for n in range(self.__nlevels)]
        oname = [utils_generic.create_filenames(o+'_', controls.ext, fname) \
            for o in ofields]
        otherKwargs = {KEY_BLOCKPROCESS: 'single', KEY_MARGIN: 0,
            KEY_RETURNARGS: ofields}
        # note the use of controls.setXYsize(-1,-1): as the size of the output data
        # is not known in advance (though it could be estimated easily), setting
        # the (xsize,ysize) sizes to -1 will force the estimation of the desired output 
        # image: this will be calculated as the sizes of the first block output by
        # the calculation and given to write multtiplied by the number of processing
        # blocks
        controls.setXYsize(-1,-1)
        # update the pyrdown attribute
        self.pyrdown.update({KEY_OUT: ofields})
        o, outfiles = utils_generic.test_workflow(self.pyrdown, [fname], oname, 
            ifields=['image'], ofields=ofields, otherArgs=None, controls=controls, 
            **otherKwargs) 
        # display
        self.display.update({KEY_ARGS: [self.__nlevels] + [o[f] for f in ofields]})
        o, outfiles = utils_generic.test_workflow(self.display, [fname], NO_ARGUMENT, 
            ifields=['image'], ofields=NO_ARGUMENT, otherArgs=None, controls=controls, 
            **otherKwargs) 
                       
#/****************************************************************************/
def test_display(fname, controls, **kwargs):
    import cv2
    print 'test matplotlib.imshow'
    utils_generic.test_function('matplotlib.imshow', fname, ''
    , controls, **kwargs)
    print 'cv2 matplotlib.imshow'
    utils_generic.test_function('cv2.imshow', fname, '', controls, **kwargs)
    cv2.waitKey()
    return

#/****************************************************************************/
def test_PIL(fname, controls, **kwargs):
    oname = None
    image =  {KEY_FUNCTION: lambda x: Image.frombuffer(x), KEY_OUT: 'pilimg'}
    histogram =  {KEY_FUNCTION: lambda x: x.histogram(), KEY_IN: 'pilimg'}

#/****************************************************************************/
def test_watershed():
    # example taken from:
    #   http://scikit-image.org/docs/dev/auto_examples/plot_marked_watershed.html#example-plot-marked-watershed-py
    import skimage
    import scipy
    from skimage import morphology, filter  
    oname = gen_create_filenames('watershed_', controls.ext, fname)
    image = img_as_ubyte(data.camera())
    # denoise image
    denoised = filter.rank.median(image, disk(2))
    # find continuous region (low gradient) --> markers
    markers = rank.gradient(denoised, disk(5)) < 10
    markers = ndimage.label(markers)[0]
    #local gradient
    gradient = rank.gradient(denoised, disk(2))
    # process the watershed
    labels = watershed(gradient, markers)


def test_subpix():
    corner = peak_local_max(corner_harris(img), num_peaks=1)
    subpix = corner_subpix(img, corner)
                       
#/****************************************************************************/
# utils_generic: a class with a bunch of utility functions
class utils_generic():
        
    @staticmethod
    def import_module(module, page=None):
        if page is None:
            try:                    exec('import ' + module)
            except  ImportError:    raise IOError, '%s module failed import' % module
        else:     
            filename = module + '.py'
            path = os_p.join(TESTRCVS_DIR,filename)
            if not utils_generic.check_filexists(path):   
                # open the webpage
                try:                            import urllib2
                except:                         raise IOError, 'urrllib2 module is required'
                try:                            
                    response = urllib2.urlopen(page) 
                except urllib2.HTTPError as e:  
                    raise urllib2.HTTPError, e.code
                else:   
                     print '!!!%s will be imported and installed in your directory (%s)!!!'  \
                        % (filename,path) 
                    #response.info()
                # read the file of the opened webpage to the designated location
                with open(path, 'w') as outfile:   
                    outfile.write(response.read())
            # try import
            try:                    exec('import ' + module)
            except  ImportError:    raise IOError, '%s module failed import' % module 
        return
        
    @staticmethod
    def create_gtiff_controls(XSize=None, YSize=None):
        controls = rcvs.ApplierControls()
        controls.setOutputDriverName('GTiff') #controls.drivername='GTiff'
        if XSize is None:   XSize = imagereader.DEFAULTWINDOWXSIZE
        if YSize is None:   YSize = imagereader.DEFAULTWINDOWXSIZE
        controls.setWindowXsize(XSize); controls.setWindowYsize(YSize)
        controls.ext='.tif'
        return controls
    
    @staticmethod
    def check_filexists(filepath):  # case insensitive file existence checker
        if os.name=='nt':   return os_p.exists(filepath) # Windows is case insensitive 
        path, name = os_p.split(os.path.abspath(filepath))
        for f in os.listdir(path):
            if re.search(f, name, re.I):    return True
        return False    
    
    @staticmethod
    def create_testfile(fname, data, dtype=None, epsg=28355,
                            xLeft=DEF_XLEFT, yTop=DEF_YTOP, xPix=DEF_PIXSIZE, yPix=DEF_PIXSIZE):
        # see createTestFile and genRampArray in riostestutils
        driver, creationOptions = gdal.GetDriverByName('GTiff'), [] # ['COMPRESS=YES']
        if data.ndim==3:            numRows, numCols, numBands = data.shape
        elif data.ndim==2:          numRows, numCols, numBands = data.shape, 1
        if dtype is None:       dtype = rcvs.IOFormat.npy2gdt[data.dtype.name]
        elif dtype<0:           dtype = DEF_DTYPE
        if dtype in rcvs.IOFormat.GdalTypes:   dtype = eval('gdal.GDT_'+dtype)
        ds = driver.Create(fname, numCols, numRows, numBands, dtype, creationOptions)
        if ds is None:      raise rioserrors.ImageOpenError('Cannot create an image')
        ds.SetGeoTransform((xLeft, xPix, 0, yTop, 0, -yPix))
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(epsg)
        ds.SetProjection(sr.ExportToWkt())    
        for iband in range(numBands):
            band = ds.GetRasterBand(iband+1)
            if numBands>1:      band.WriteArray(data[:,:,iband])
            else:               band.WriteArray(data)
        del ds
        return #ds
    
    @staticmethod
    def create_filenames(gname, ext, fname):
        if not isinstance(ext,str):     raise IOError, 'string of file extension expected'
        extbasename = lambda x: os_p.basename(os_p.splitext(x)[0]) + ext
        if isinstance(fname,(list,tuple)):   
            oname = map(extbasename, fname)
            if isinstance(gname,str) and gname!='':oname = [gname+on for on in oname]
            elif isinstance(gname,(list,tuple)):   oname = map(lambda x,y:x+y, gname, oname)
        elif isinstance(gname,str):                                       
            if isinstance(gname,str):              oname = gname+extbasename(fname)
            else:                                  raise IOError, 'string of generic name expected'
        else:                           raise IOError
        return oname
    
    @staticmethod
    def initialize_files(ifnames, ofnames, ifields, ofields):
        infile, outfile = NO_ARGUMENT, NO_ARGUMENT
        emptyList = ((),[],None)
        # create/initialize the infile with input filenames
        if not (ifnames in emptyList or ifields in emptyList):
            infile = applier.FilenameAssociations()
            ifnames, ifields = [[v] if isinstance(v,str) else v for v in (ifnames, ifields)]
        # create/initialize the outfile with output filenames if it is requested
        if not (ofnames in emptyList or ofields in emptyList):
            outfile = applier.FilenameAssociations()
            ofnames, ofields = [[v] if isinstance(v,str) else v for v in (ofnames, ofields)]
        # set the fields
        if not (ifnames in emptyList or ifields in emptyList):
            for (ifield, iname) in itertools.izip(ifields,ifnames):
                setattr(infile, ifield, iname) #[iname]
        else:
            pass
        if not (ofnames in emptyList or ofields in emptyList):
            for (ofield, oname) in itertools.izip(ofields,ofnames):
                setattr(outfile, ofield, oname)
        else:
            pass
        return infile, outfile
        
    @staticmethod
    def test_workflow(workflow, fname, oname=NO_ARGUMENT,
            ifields=TESTRCVS_DEF_IFIELD, ofields=TESTRCVS_DEF_OFIELD,
            otherArgs=None, controls=None, **otherKwargs):
        if isinstance(oname,str):           oname = [oname]
        if not oname in ((),[],None):       oname = [os_p.join(TESTRCVS_DIR, o) for o in oname]
        else:                               ofields, oname = NO_ARGUMENT, NO_ARGUMENT
        infile, outfile = utils_generic.initialize_files(fname, oname, ifields, ofields)        
        outargs = rcvs.apply(workflow, infile, outfile, otherArgs=otherArgs, 
                                  controls=controls, **otherKwargs)
        return outargs, outfile
         
    @staticmethod
    def test_equality(infile, tarray, test='value'):
        ok = True
        inputfile = infile.__dict__.values()
        if isinstance(inputfile,list) and len(inputfile)>1:
            return all([utils_generic.test_equality(ifile, ta, test=test)    \
                for ifile, ta in zip(inputfile, tarray)])
        else:
            if isinstance(inputfile,list):  inputfile = inputfile[0]
            ds = gdal.Open(inputfile)
            if ds is None:
                raise IOError, 'loading error when testing input file' % inputfile
            elif ds.RasterCount!=len(tarray):
                raise IOError, 'wrong input parameters dimensions'    
            #tarray = np.expand_dims(tarray,axis=0)
            for iband in range(ds.RasterCount):
                iarray = ds.GetRasterBand(iband+1).ReadAsArray()
                if (test=='value' and not np.array_equal(iarray, tarray[iband]))   \
                  or (test=='shape' and iarray.shape!=tarray[iband].shape):
                    ok = False; break
                else:   
                    pass
            del ds        
        return ok
