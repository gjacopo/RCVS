#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
title:          rcvs.py
module:         RCVS (Raster Computer Vision Simplification)

summary:        Automatisation of geo-processing workflow over raster and vector data   
description:    Python basic tools for applying a geo-processing workflow over raster and
                vector data:
                    - using the input/output utility functions of RIOS (Raster Input Output
                    Simplification) module, itself based on Gdal module,
                    - using external Computer Vision and Image Processing processing (CVIP) 
                    algorithms provided (when installed independently) by modules like PIL, 
                    OpenCV, skimage, matplotlib and/or scipy.ndimage.
                This way, a 'simple' definition of processing workflow is possible

usage:          see main()/test()

:CONTRIBUTORS:  grazzja
:CONTACT:       jacopo.grazzini@ec.europa.eu
:SINCE:         Fri May 31 10:20:51 2013
:VERSION:       0.9

:REQUIRES:      gdal, rios, numpy, scipy
                Queue, multiprocessing
                math, re, inspect, operator, itertools, collections           
:OPTIONAL:      cv2, skimage, PIL, matplotlib, vigra, mahotas
                pathos
"""

# EUPL LICENSE
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.


#==============================================================================
# PROGRAM METADATA
#==============================================================================

__contributor__ = ['grazzja']
__license__     = 'EUPL'
__date__        = 'Fri May 31 10:20:51 2013'
__version__     = '0.9'

__all__         = ['apply']


#==============================================================================
# IMPORT STATEMENTS
#==============================================================================

## import standard common modules
import os, sys, inspect, re
import warnings
import math
import itertools, collections

def raiseMessage(error, message):           raise error, message
def raiseImportError(mod,web):
    message = str('%s module is required' % mod)
    if web not in ('',None):        message += str('; visit %s for install' % web)
    raise ImportError, message
def warnMessage(message):                   warnings.warn(message)
def warnImportError(mod,web):
    message = str('missing %s module' % mod)
    if web not in ('',None):        message += str('; visit %s for install' % web)
    warnMessage(message)

try:
    numpy_mod, numpy_web = 'NumPy', 'http://www.numpy.org/'
    import numpy as np
    Array=np.ndarray
    memArray=np.memmap
except ImportError:
    raiseImportError(numpy_mod,numpy_web)

## import specific CVIP modules used herein
# last visit of websites: 15/01/14

## import GDAL module just to ensure it is available
try:
    gdal_mod, gdal_web = 'GDAL', 'https://pypi.python.org/pypi/GDAL/'
    import gdal
except ImportError:
    try:                from osgeo import gdal#analysis:ignore
    except ImportError: raiseImportError(gdal_mod,gdal_web)

# define the RCVS_LIST_MODULES constant as the list of available(imported) CV modules
RCVS_LIST_MODULES = []
gdal__name__='gdal' # used instead of gdal.__name__ that may return 'osgeo.gdal'
RCVS_LIST_MODULES.append(gdal__name__) # RIOS format

## import RIOS module (submodules) used herein
try:
    rios_mod, rios_web = 'RIOS', 'https://bitbucket.org/chchrsc/rios'
    # import rios
    from rios import imageio, imagereader, imagewriter, vectorreader
    from rios import applier, rioserrors
    # from rios import calcstats
except (ValueError, ImportError):
    raiseImportError(rios_mod,rios_web)

try:
    scipy_mod, scipy_web = 'SciPy', 'http://docs.scipy.org/doc/scipy/reference/ndimage.html'
    import scipy
    #from scipy import ndimage
    RCVS_LIST_MODULES.append(scipy.__name__)
except ImportError:
    warnImportError(scipy_mod, scipy_web)
    class scipy: pass
    #class ndimage: pass

try:
    matplotlib_mod, matplotlib_web = 'matplotlib', 'http://matplotlib.org/'
    import matplotlib
    #from matplotlib import pyplot
    RCVS_LIST_MODULES.append(matplotlib.__name__)
except ImportError:
    warnImportError(matplotlib_mod,matplotlib_web)
    class matplotlib: pass
    #class pyplot: pass

try:
    cv2_mod, cv2_web = 'OpenCV', 'http://opencv.org/' # last visited 15/01/14
    import cv2
    #from cv2 import cv as cv
    RCVS_LIST_MODULES.append(cv2.__name__)
except ImportError:
    warnImportError(cv2_mod,cv2_web)
    class cv2: pass
    #class cv(): iplimage = type('DUM_CV_IMAGE_TYPE',(object,),{})

try:
    skimage_mod, skimage_web = 'scikits-image', 'http://scikits.appspot.com/scikits-image'
    import skimage
    #from skimage.io import _io as skim_io
    RCVS_LIST_MODULES.append(skimage.__name__)
except ImportError:
    warnImportError(skimage_mod,skimage_web)
    class skimage: pass
    #class skim_io(): Image = type('DUM_SKIM_IMAGE_TYPE',(object,),{})

try:
    pil_mod, pil_web = 'PIL', 'http://www.pythonware.com/products/pil/, http://effbot.org/zone/pil-index.htm'
    import PIL
    #from PIL import Image as pil_im
    RCVS_LIST_MODULES.append(PIL.__name__)
except ImportError:
    # define dummy class
    warnImportError(pil_mod,pil_web)
    class PIL: pass
    # define dummy type as an instance of a dummy class
    #class pil_im(): Image = type('DUM_PIL_IMAGE_TYPE',(object,),{})

try:
    vigra_mod, vigra_web = 'vigra', 'http://ukoethe.github.io/vigra/doc/vigranumpy/index.html'
    import vigra
    VigraArray = vigra.VigraArray
    RCVS_LIST_MODULES.append(vigra.__name__)
except ImportError:
    warnImportError(vigra_mod, vigra_web)
    class vigra: pass
    class VigraArray(): Image = type('DUM_VIGRA_IMAGE_TYPE',(object,),{})
    #class pyplot: pass

try:
    mahotas_mod, mahotas_web = 'mahotas', 'http://luispedro.org/software/mahotas' # last visited 15/02/14
    import mahotas
    #from cv2 import cv as cv
    RCVS_LIST_MODULES.append(mahotas.__name__)
except ImportError:
    warnImportError(mahotas_mod,mahotas_web)
    class mahotas: pass

## check the availability and import other modules used in processing 

# Python doesn't pickle method instance by default
# http://stackoverflow.com/questions/1816958/cant-pickle-type-instancemethod-when-using-pythons-multiprocessing-pool-ma
import copy_reg
import types
def _pickle_method(method): # identify the fields of the method
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)
def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:                    func = cls.__dict__[func_name]
        except KeyError:        pass
        else:                   break
    return func.__get__(obj, cls)
# http://matthewrocklin.com/blog/work/2013/12/05/Parallelism-and-Serialization/
# Note that functions (built-in and user-defined) that are mapped for multiprocessing  
# using the original multiprocessing module are pickled by "fully qualified" name
# reference, not by value. This means that only the function name is pickled, along 
# with the name of the module the function is defined in. Neither the function's code, 
# nor any of its function attributes are pickled. Thus the defining module must be 
# importable in the unpickling environment, and the module must contain the named 
# object, otherwise an exception will be raised.
# Similarly, classes are pickled by named reference, so the same restrictions in the
# unpickling environment apply. Note that none of the class's code or data is pickled.
# Similarly, when class instances are pickled, their class's code and data are not
# pickled along with them. Only the instance data are pickled. This is done on purpose, 
# so you can fix bugs in a class or add methods to the class and still load objects
# that were created with an earlier version of the class. 
copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

import Queue
import multiprocessing
RCVS_CPU_NODES = multiprocessing.cpu_count()

pathos_mod, pathos_web = 'pathos', 'https://github.com/uqfoundation/pathos/'
#if re.search('win',sys.platform):   
#    warnImportError(pathos_mod, pathos_web) # platform.system()
#else:    
try:                            
    import pathos
except ImportError:             
    warnImportError(pathos_mod, pathos_web)
    class pathos:   pass
else:
    from pathos.multiprocessing import ProcessingPool
    pool = ProcessingPool(ncpus=1)               
    try: 
        pool.imap(lambda x: x**2, (0,), chunksize=1).next(timeout=0.1)
        RCVS_LIST_MODULES.append(pathos.__name__)
    except: # (multiprocessing.TimeoutError,cPickle.PicklingError):
        warnMessage('pathos multiprocessing not available with the current system')
    else:
        warnMessage('pathos multiprocessing will be used')       
# most of the multiprocessing pool operations performed herein are based on the 
# module pathos, not the original multiprocessing module itself

#==============================================================================
# GLOBAL VARIABLES/METHODS
#==============================================================================

## global constants used herein
RCVS_KEY_IN                = 'in'
RCVS_KEY_OUT               = 'out'
RCVS_KEY_BLOCKBYTE         = 'dtype' # reading option
RCVS_KEY_BLOCKFORMAT       = 'fmtBlock'
RCVS_KEYDICT_BLOCKFORMAT   =                                                           \
                              dict([(fmtDir,fmtDir+RCVS_KEY_BLOCKFORMAT[0].capitalize()\
                                     +RCVS_KEY_BLOCKFORMAT[1::])                       \
                                    for fmtDir in (RCVS_KEY_IN,RCVS_KEY_OUT)])
RCVS_KEY_INAXIS            = RCVS_KEYDICT_BLOCKFORMAT[RCVS_KEY_IN]
RCVS_KEY_OUTAXIS           = RCVS_KEYDICT_BLOCKFORMAT[RCVS_KEY_OUT]

## additional keys used for defining the processing workflow
RCVS_KEY_MODULE            = 'module'
RCVS_KEY_FUNCTION          = 'call'
RCVS_KEY_ARGS              = 'args'
RCVS_KEY_KWARGS            = 'kwargs'

## final list of keys used for configuring the processing workflow
RCVS_KEYLIST_WORK          =                                      \
                       (RCVS_KEY_MODULE, RCVS_KEY_FUNCTION,       \
                        RCVS_KEY_IN, RCVS_KEY_OUT,                \
                        RCVS_KEY_ARGS, RCVS_KEY_KWARGS,           \
                        RCVS_KEY_BLOCKFORMAT)

## keys used for defining the processing
RCVS_LIST_BLOCKPROCESS     = ('single', 'serial', 'pool', 'pthread')
RCVS_KEY_BLOCKSINGLE       = 'singleBlock'
RCVS_KEY_BLOCKPROCESS      = 'procBlock'

RCVS_KEY_BLOCKNUM          = 'numBlock'
RCVS_KEY_RETURNARGS        = 'returnArgs'

RCVS_KEY_INFO              = 'info'
RCVS_KEY_MARGIN            = 'margin' # 'overlap'

## additional variables used
RCVS_NO_ARGUMENT           = None
RCVS_DEF_ARGUMENT          = set((None,())).difference(set((RCVS_NO_ARGUMENT,))).pop() # ()
# note that RCVS_DEF_ARGUMENT corresponds to a 'neutral' argument, ie. nothing
# has been passed for a given key, then it is left to the default value in the
# program
RCVS_EMPTY_ARGUMENT        = () # ((),)
# a NO_ARGUMENT key will have its value set to EMPTY_ARGUMENT

RCVS_DEF_BLOCKRANGE        = 5

RCVS_NONE2_LIST            = (None,None)

#==============================================================================
# METHODS
#==============================================================================

## new functions/classes defined for cvapplier utility

#/****************************************************************************/
# Format
#/****************************************************************************/
class Format(object):
    """Class of utility methods for data format/structure manipulation.
    
    Numpy does not provide any means to attach semantics to axes, but relies purely
    on the convention that the most important axis is last, as in :literal:`array[y, x]` 
    or :literal:`array[z, y, x]` ("C-order").
    
    However, there is no way to enforce this convention in a program, since arrays
    can be transposed outside of the user's control (e.g. when saving data to a 
    file). Moreover, many imaging libraries use the opposite convention where the
    :data:`x`-axis comes first, as in :literal:`array[x, y]` or 
    :literal:`array[x, y, z]`.
    """

    #/************************************************************************/
    # Data format (or shape): sets the axis arrangement (order) of the (numpy)
    # Array

    KEY_NOFORMAT = RCVS_NO_ARGUMENT # None
    # ensure in particular that KEY_NOFORMAT!=RCVS_KEY_DEFARGUMENT

    mod2format = {
        gdal__name__:           'zyx',      gdal.__name__:          'zyx',
        PIL.__name__:           'yxz',
        cv2.__name__:           'yx-z', # in cv2, RGB images are read as BGR
        skimage.__name__:       'yxz',
        matplotlib.__name__:    'yxz',  #'-yxz'
        scipy.__name__:         'yxz',
        np.__name__:            'yxz',
        vigra.__name__:         'xyz', # we refer here to the RGB format of vigra
        mahotas.__name__:       'yxz',
        KEY_NOFORMAT:           '' # no change will be made
        }
    """Define the format (shape) of the Arrays used for processing by the various 
    CVIP modules considered herein (say it otherwise, the axis arrangement of the 
    matrices used by those modules.
    """

    DataBlockFormat = \
        list(set(mod2format.keys()).union(set(mod2format.values())))
        # list(set(mod2format.keys()).difference((None,)).union(set(mod2format.values())))
    """Define what a string used as a block format entry should be: both the name 
    of the module used and the description of the format itself are accepted.
    """
    # see also RCVS_KEY_BLOCKFORMAT) 

    #/************************************************************************/
    # Data type (byte format) dictionaries that can be used for various type
    # conversions

    # Python pack types names
    PPTypes = ['B', 'b', 'H', 'h', 'I', 'i', 'f', 'd']
    # http://www.python.org/doc//current/library/struct.html
    # http://docs.python.org/2/library/array.html

    # NumPy types names
    NumPyTypes = [np.dtype(__n).name for __n in PPTypes+['l','L','c']]

    # Python pack types <-> Numpy
    ppt2npy = dict([(__n,np.dtype(__n).name) for __n in PPTypes])
    # http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
    # http://docs.scipy.org/doc/numpy/reference/arrays.scalars.html
    # note regarding np.dtype:
    #   np.dtype('B')           -> dtype('uint8')
    #   np.dtype('uint8')       -> dtype('uint8')
    #   np.dtype(np.uint8)      -> dtype('uint8')
    npy2ppt = dict(zip(ppt2npy.values(),ppt2npy.keys()))

    # GDAL Data Type names
    # http://www.gdal.org/gdal_datamodel.html
    __ngdtypes = 11 # currently 11...
    GdalTypes = [__n for __n in [gdal.GetDataTypeName(__i) for __i in range(1,__ngdtypes+1)] if __n]
    # note: there is no 'b' type ('int8') in Gdal!
    # to retrieve thelist of GDT values, we can use the following:
    #   gdalGdalTypes = [eval('gdal.GDT_'+dtype) for dtype in GdalTypes]

    # GDAL <-> Python pack types
    gdt2ppt = dict(zip(GdalTypes,[__n for __n in PPTypes if __n!='b']))
    ppt2gdt = dict(zip(gdt2ppt.values(),gdt2ppt.keys()))
    ppt2gdt.update({'b': 'Byte'})

    # GDAL <-> Numpy
    gdt2npy = dict([(__i,ppt2npy[__n]) for (__i,__n) in gdt2ppt.items()             \
                    if __n in ppt2npy.keys()])      
    # add the complex types
    gdt2npy.update({'CInt16':'complex64', 'CFloat32':'complex64','CInt32':'complex64', 'CFloat64':'complex128'})
    npy2gdt =   dict(zip(gdt2npy.values(),gdt2npy.keys()))
    # since several things map to complex64 (see gdt2npy update above), we ensure that only
    # one match is selected (http://gdal.org/python/osgeo.gdal_array-pysrc.html)
    npy2gdt.update({'complex64':'CFloat32'}) # it could be the case already, but make it safe
    npy2gdt.update({'int8':'Byte', 'bool':'Byte'}) # ({'single':'Float32', 'float':'Float64'})
    # see also imageio.dataTypeMapping
    
    # OpenCV Data Type names
    # obviously order matters, as the order in OCVTypes should correspond to
    # that of PPTypes
    OCVTypes = ['8U', '8S', '16U', '16S', '32S', 'i', '32F', '64F']
    # note: there is no 'i' type ('int32') in OpenCV
    try:
        # OpenCV(cv2) <-> Python pack types
        ocv2ppt = dict([(eval('cv2.IPL_DEPTH_'+__n),__i) for (__n,__i) in zip(OCVTypes,PPTypes) \
                        if __n!='i' and __i!='i'])
        ppt2ocv = dict(zip(ocv2ppt.values(),ocv2ppt.keys()))
        ppt2ocv.update({'i': cv2.IPL_DEPTH_32S})
        # OpenCV(cv2) <-> Numpy
        ocv2npy = dict([(__i,ppt2npy[__n]) for (__i,__n) in ocv2ppt.items() \
                        if __n in ppt2npy.keys()])
        npy2ocv = dict(zip(ocv2npy.values(),ocv2npy.keys()))
        npy2ocv.update({'int32': cv2.IPL_DEPTH_32S})
    except:
        pass

    # PIL Data Type names
    PILTypes = ['1', 'L', 'P', 'RGB', 'RGBA', 'CMYK', 'YCbCr', 'I', 'F']
    # http://effbot.org/imagingbook/concepts.htm
    __pilppt = ['B', 'B', 'B', 'B',   'B',    'B',    'B',     'i', 'f']
    # PIL -> Python pack types
    pil2ppt = dict(zip(PILTypes,__pilppt))
    # PIL -> Numpy
    pil2npy = dict(zip(PILTypes,[ppt2npy[__n] for __n in __pilppt]))

    # Skimage Data Type names
    # http://scikit-image.org/docs/dev/user_guide/data_types.html
    SkimTypes = NumPyTypes
    # ski2npy = dict(zip(SkimTypes,NumPyTypes)) # not used here
    # npy2ski, ski2ppt, ppt2ski = ski2npy, npy2ppt, ppt2npy

    # Pyplot Data Type names
    # http://matplotlib.org/users/image_tutorial.html
    PPlTypes = NumPyTypes
    # ppl2npy = dict(zip(PPlTypes,NumPyTypes))
    # npy2ppl, ppl2ppt, ppt2ppl = ppl2npy, npy2ppt, ppt2npy
    # note that "Matplotlib plotting can handle float32 and uint8, but image reading/writing
    # for any format other than PNG is limited to uint8 data." However, the type of the data
    # used being a Numpy array, we keep it open for type conversions.

    # obviously...
    SciPyTypes = NumPyTypes

    # define all possible (unique) strings for data type description
    DataBlockTypes = list(set(PPTypes+GdalTypes+OCVTypes+PILTypes+NumPyTypes))

    #/************************************************************************/
    @classmethod
    def isCVModule(cls, module):
        """Check that a module is one of the various loaded CVIP modules.
        
            >>> resp = Format.isCVModule(module)
        """
        if module is None:                  return False
        elif inspect.ismodule(module):      return cls.isCVModule(module.__name__)
        elif not isinstance(module,str):    raise IOError, 'unexpected argument'
        return any([re.search(m,module) for m in RCVS_LIST_MODULES])

    #/************************************************************************/
    @classmethod
    def whichCVModule(cls, module):
        """Retrieve the name of a CVIP module when loaded.
        
            >>> name = Format.whichCVModule(module)
        """
        if module is None:                  return None
        elif inspect.ismodule(module):      return cls.whichCVModule(module.__name__)       
        elif not isinstance(module,str):    raise IOError, 'unexpected argument'
        res = [re.search(m,module) for m in RCVS_LIST_MODULES]
        if any(res):    
            return RCVS_LIST_MODULES[[i for (i,s) in enumerate(res) if s][0]]
        else:           
            return None

    #/************************************************************************/
    @classmethod
    def checkTypeBlock(cls, dType):
        """Check that a given type is accepted for processing.
        
            >>> resp = Format.checkTypeBlock(dType)
        """
        if not (dType is None or dType in cls.DataBlockTypes):
            raise IOError, 'wrong block type: must be None or any string in %s' % cls.DataBlockTypes
        else:
            return dType

    #/************************************************************************/
    @classmethod
    def toByte(cls, x, dType=None, oMod=None): # dType is a string defining the type of the data
        """Convert some data to the desired byte format, taking into account both
        the type and the format.
        
            >>> new = Format.toByte(x, dType=None, oMod=None)
        
        Arguments
        ---------
        x : scalar,np.ndarray,vigra.VigraArray,dict,list,tuple
            container storing some data to convert.
        dType : str,type
            container providing the type to convert the data to.
        oMod : str
            name of the module to which the data will be sent. 
            
        Returns
        -------
        new : scalar,np.ndarray,vigra.VigraArray,dict,list,tuple
            well formatted data representing the same data as those conveyed by
            :data:`x`\ .
        """
        try:                np.ndarray([1,2,2]).astype('int8',copy=False)
        except TypeError:   asType = lambda x, dtype: x.astype(dtype)
        else:               asType = lambda x, dtype: x.astype(dtype,copy=False)
        tobyteArray = {
            False:          lambda x, dtype: asType(x,dtype) if (dtype is not None and x.dtype!=dtype)      \
                                else x,
            True:           lambda x, dtype: skimage.img_as_ubyte(x) if re.search('int8',dtype)             \
                                else (skimage.img_as_int(x) if re.search('int16',dtype)                     \
                                else (skimage.img_as_uint(x) if re.search('uint16',dtype)                   \
                                else (skimage.img_as_float(x) if re.search('float',dtype)                   \
                                else (asType(x,dtype) if (dtype is not None and x.dtype!=dtype)             \
                                else x))))
            }
        tobyteScalar =      lambda x, dtype: int(x) if any([re.search(t,dtype) for t in ('int8','int16')])  \
                                else (long(x) if re.search('int32', dtype) \
                                else (float(x) if re.search('float', dtype) else x))
        if oMod==cv2.__name__ and (dType is None or not any([re.search(t,dType) for t in ('int8','uint8')])):
            dType = 'uint8'
        elif oMod==vigra.__name__ and dType is None:
            dType = 'float32' # unless noted, vigra functions expect/create numpy.float32 arrays
        if dType is None:                       
            return x
        elif np.isscalar(x):                    
            return tobyteScalar(x,dType)
        if isinstance(x,(Array,VigraArray)):    
            return tobyteArray[oMod==skimage.__name__ ](x,dType)
        elif isinstance(x,dict):                
            return dict(zip(x.keys(),map(cls.toByte, x.values(), [dType]*len(x), [oMod]*len(x))))
        elif isinstance(x,(list,tuple)):        
            return map(cls.toByte, x, [dType]*len(x), [oMod]*len(x))

    #/************************************************************************/
    @classmethod
    def checkFormatBlock(cls, formatBlock):
        """Check and retrieve a format for describing data.
        
            >>> fmt, mod = Format.checkFormatBlock(fmt)
        
        Arguments
        ---------
        fmt : str, module
            either the (name of a) module or a string describing the shape of the array
            (i.e. the order of axis in the array) itself (nothing to do then).
            
        Returns
        -------
        fmt : str
            string describing the shape of an array for a given module (and not 
            anymore the name of the module itself).
        mod : str
            identified module name if any.
        """
        # in case a module itself has been passed, retry
        if inspect.ismodule(formatBlock): return cls.checkFormatBlock(formatBlock.__name__)
        # first check that the string describing the format is acceptable: i
        if formatBlock not in cls.DataBlockFormat:
            raise IOError, 'wrong block format: must be any string in %s' % cls.DataBlockFormat
        # ...then ensure it is a string describing the shape of the array (and not
        # the name of a module)
        elif formatBlock in cls.mod2format.keys(): # formatBlock is the name of a module
            return cls.mod2format[formatBlock], formatBlock
        # else return: it is already in cls.mod2format.values()
        else:
            return formatBlock, None # we have no clue about the module

    #/************************************************************************/
    # toAxis: method for rearranging the array data
    @classmethod
    def toAxis(cls, x, oAxis, iAxis='zyx'):
        try: # keyword 'axis' new in version 1.7.0
            np.squeeze(np.ndarray([1,2,2]), axis=(0,))
        except TypeError:
            toaxisSqueeze = lambda x, axis: x if (len(x.shape)!=3 or x.shape[axis]!=1) else np.squeeze(x)
        else:
            toaxisSqueeze = lambda x, axis: x if (len(x.shape)!=3 or x.shape[axis]!=1) else np.squeeze(x, axis=axis)
        toaxisExpand =      lambda x, axis: x if len(x.shape)==3 else np.expand_dims(x, axis=axis)
        toaxisTranspose =   lambda x, axes: x.transpose(axes) # np.transpose(x,axes=axes)
        toaxisInvert = lambda x, lidx: x if lidx in (None,()) \
                        else x[tuple((slice(None,None,-1) if i in lidx else slice(None) for i in range(x.ndim)))]
        #toaxisInvert = lambda x, idx: np.swapaxes(np.swapaxes(x, 0, idx)[::-1], 0, idx)
        toaxis = { # vigra is assumed to be 'xyz'
            'zyx':  {'yxz': lambda x, lidx: toaxisTranspose(toaxisInvert(toaxisExpand(x,2),lidx),[2, 0, 1]),
                     'zyx': lambda x, lidx: toaxisInvert(x,lidx),
                     'vigra':lambda x, lidx: toaxisInvert(Array(x.asRGB(normalize=False).transposeToOrder('F')), [lidx[2]]+lidx[:2])
                     },
            'yxz':  {'zyx': lambda x, lidx: toaxisInvert(toaxisSqueeze(toaxisTranspose(x,[1, 2, 0]),2),lidx),
                     'yxz': lambda x, invert: toaxisInvert(x,lidx),
                     'vigra':lambda x, lidx: toaxisInvert(Array(x.transposeToOrder('C')), lidx)
                     },
            'vigra':{'yxz': lambda x, lidx: VigraArray(toaxisInvert(toaxisExpand(x,2),lidx),order='C'),
                     'zyx': lambda x, lidx: VigraArray(toaxisInvert(toaxisExpand(x,2),lidx),order='F'),
                     'vigra':lambda x, lidx: x
                     }
            # the configurations above are the only possible with the considered modules
            }
        # check if no change is requested
        if oAxis=='' or iAxis=='':              return x
        # otherwise define the list of axis that need to be inverted:  lidx will return
        # the index of the axis in (y,x,z) order
        lidx = [] # will stored the indices in (y,x,z) of the axis that change orientation
        [lidx.append(i) for (i,straxis) in enumerate(['y','x','z']) \
            if (re.search('-'+straxis,iAxis) is None)^(re.search('-'+straxis,oAxis) is None)]
        # about (re.search('-'+straxis,iAxis) is None)^(re.search('-'+straxis,oAxis) is None): an axis
        # is inverted if and only if there is a change in 'sign' in between iAxis and oAxis, eg.:
        #   - if iAxis='yx-z' and oAxis='zyx', then lidx=[2] as the z-axis needs to be inverted,
        #   - if iAxis='-yxz' and oAxis='yx-z', then lidx=[0,2] as both y- and z- need to be inverted,
        #   - if iAxis='-yxz' and oAxis='z-yx' (non existing example), then lidx=[] as there is no
        #     axis inversion needed.
        # clean the axis definitions
        iAxis, oAxis = iAxis.replace('-',''), oAxis.replace('-','')
        # apply the matrix transformation
        if isinstance(x,Array):                 return toaxis[oAxis][iAxis](x,lidx)
        else:                                   return x

    #/************************************************************************/
    @classmethod
    def formatContainer(cls, blockContainer, **kwargs):
        # define a utility function for manipulating the block container
        defineNumpyType =  lambda dtype: dtype if dtype in cls.NumPyTypes               \
                      else (cls.ppt2npy[dType] if dtype in cls.PPTypes                  \
                      else (cls.gdt2npy[dtype] if dtype in cls.GdalTypes                \
                      else (cls.ocv2npy[dtype] if dtype in cls.OCVTypes                 \
                      else (cls.pil2npy[dtype] if dtype in cls.PILTypes                 \
                      else None))))
        # we assume here that non CV module use the same array format as Gdal ('zyx'); if this
        # not the case, such module should be added to the RCVS_LIST_MODULES list and a specific
        # format transformation should be implemented herein (through toAxis and toByte methods)
        formatBlock =       lambda block, oAxis, iAxis, dType, oMod: cls.toByte(cls.toAxis(block,oAxis,iAxis), dType, oMod)
        # set/check the byte format of the data
        dType = kwargs.pop(RCVS_KEY_BLOCKBYTE,None)
        if not (dType is None or isinstance(dType,str)):    raise IOError, 'unknown byte format %g' % dType
        else:                                               dType = defineNumpyType(dType)
        # set/check the input/output format type (array arrangement) of the data
        # and the module, when it exists
        oMod = kwargs.pop(RCVS_KEY_MODULE,None)
        iAxis, oAxis = [kwargs.pop(key,cls.mod2format[gdal__name__])                    \
            for key in (RCVS_KEY_INAXIS,RCVS_KEY_OUTAXIS)]
        iAxis = iAxis if iAxis==vigra.__name__ else cls.checkFormatBlock(iAxis)[0]
        oAxis = vigra.__name__ if (oMod==vigra.__name__ or oAxis==vigra.__name__)       \
            else cls.checkFormatBlock(oAxis)[0]
        # define the complex formating function as a case handler
        formatCase = lambda bC: [formatBlock(b, oAxis, iAxis, dType, oMod) for b in bC] if isinstance(bC,(list,tuple))            \
            else (dict(zip(bC.keys(),[formatBlock(b, oAxis, iAxis, dType, oMod) for b in bC.values()])) if isinstance(bC,dict)    \
            else formatBlock(bC, oAxis, iAxis, dType, oMod))
        # return the formatted block container
        return formatCase(blockContainer)

    #/************************************************************************/
    # totype: method for converting the data to the desired type (structure)
    @classmethod
    def toType(cls, x, fmtOut=Array):
        totypeList =        lambda x: list(x) if isinstance(x,(tuple,Array)) \
                     else (x.values() if isinstance(x,dict) else x)
        totypeTuple =       lambda x: tuple(x) if isinstance(x,list) \
                      else (tuple(list(x)) if isinstance(x,Array) \
                            else (tuple(x.values()) if isinstance(x,dict) else x))
        totypeArray =       lambda x: Array(x) if isinstance(x,(list,tuple)) \
                      else (x.values()[0] if (isinstance(x,dict) and len(x)==1) \
                            else (x.values() if isinstance(x,dict) else x))
        totypeDict =        lambda x: dict(enumerate(x)) if isinstance(x,(list,tuple)) \
                     else (dict(enumerate((x,))) if isinstance(x,Array) else x)
        return {list:           totypeList,
                tuple:          totypeTuple,
                Array:          totypeArray,
                dict:           totypeDict,
                None:           lambda x:x
                }[fmtOut](x)

    # utility function for 'reducing' an iterable
    @classmethod
    def flattenIterable(cls, iterable):
        #return reduce(lambda l, i: l + cls.flattenIterable(i) if isinstance(i, (list, tuple)) \
        #  else l + (i,), iterable, ())
        if iterable is None:                            return ((),)
        elif not isinstance(iterable, (list, tuple)):   raise IOError
        flatten = lambda iterable: reduce(lambda l, i: l + flatten(i) if isinstance(i, (list, tuple)) \
            else l + (i,), iterable, ())
        if isinstance(iterable,list):                   return list(flatten(iterable))
        else:                                           return flatten(iterable)

    @classmethod
    def definePicklable(cls, arg, unpick=False, block=False):
        arg_cls_name = arg.__class__.__name__.lower()
        if re.search('list',arg_cls_name):  
            pickle, unpickle = lambda arg, x: arg.append(x),lambda arg: arg.pop()                    
        elif re.search('queue',arg_cls_name):   
            pickle, unpickle = lambda arg, x: arg.put(x),   lambda arg: arg.get(block=block)                   
        elif re.search('tuple',arg_cls_name):  # tuples are immutable
            pickle, unpickle = lambda arg, x: arg + (x,),   lambda arg: arg                
        else:                                   
            raise IOError, 'unexpected picklable structure'
        if unpick is False:             return pickle
        else:                           return unpickle
        
#/****************************************************************************/
# ImageReader overriding class
#/****************************************************************************/
class ImageReader(imagereader.ImageReader):

    # variable set to True when one single block (ie. the whole image) is loaded
    __singleBlock = False
    # variable setting the array format (shape) of the blocks: any of the format
    # used by the accepted CV modules
    __formatBlock, __module = None, None
    # define the type (byte) of the blocks ot read
    __dtypeBlock = None
    # variable for 'memorizing' the actual window size when it is decided to read
    # one entire block
    __windowxsize, __windowysize = None, None

    #/************************************************************************/
    def __init__(self, *args, **kwargs):
        """Retrieve specific keyword arguments not to be passed to the superclass.

            >>> x = ImageReader(*args, **kwargs)
        """
        self.__formatBlock, self.__module =                                         \
            Format.checkFormatBlock(kwargs.pop(RCVS_KEY_BLOCKFORMAT,None)    \
            or kwargs.pop(RCVS_KEY_OUTAXIS,gdal__name__))
        self.__dtypeBlock = Format.checkTypeBlock(kwargs.pop(RCVS_KEY_BLOCKBYTE,None))
        self.__singleBlock = kwargs.pop(RCVS_KEY_BLOCKSINGLE, False)
        # run the super class method
        imagereader.ImageReader.__init__(self, *args, **kwargs)
        # store the desired window(x,y)size in case we modify it later when
        # reading one block only
        self.__windowxsize, self.__windowysize = self.windowxsize, self.windowysize
        # at that stage, we may not know yet what will be exactly the size of the
        # analyzing window as self.info has not been 'prepared' for sure; we force
        # it however
        if self.info is None:           self.prepare(**kwargs)

    #/************************************************************************/
    def prepare(self, workingGrid=None, **kwargs):
        """
            >>> x.prepare(workingGrid=None, **kwargs)
        """
        # possibly create if not done already
        if self.info is None:
            imagereader.ImageReader.prepare(self, workingGrid=workingGrid)
        # update the info field: in the case the whole image is to be read all
        # at once (hence, one block only) reset the 'windowing' dimensions
        if self.__singleBlock or kwargs.pop(RCVS_KEY_BLOCKSINGLE, False):
            self.windowxsize, self.windowysize = self.info.xsize, self.info.ysize
            self.info.windowxsize, self.info.windowysize = self.info.xsize, self.info.ysize
            self.info.xtotalblocks, self.info.ytotalblocks = 1, 1
        else: # reset if it has possibly been modified earlier
            self.windowxsize, self.windowysize = self.__windowxsize, self.__windowysize
        return

    #/************************************************************************/
    def readBlock(self, *nblock, **kwargs):
        """
            >>> blockInfo, blockContainer = x.readBlock(*nblock, **kwargs)
        """
        if nblock in ((),None):         nblock=0
        else:                           nblock=nblock[0]
        # prepare and/or update
        self.prepare(**kwargs)
        # we still need to raise an OutsideImageBoundsError for the key counter
        # (same condition as in imagereader.ImageReader.readBlock) when iterating
        # over the reader
        if nblock >= reduce(lambda x,y:x*y, self.info.getTotalBlocks()):
            raise rioserrors.OutsideImageBoundsError()
        # update the desired output format within kwargs, otherwise used the default
        # one defined in the constructor
        fmtOut, module = Format.checkFormatBlock(kwargs.pop(RCVS_KEY_BLOCKFORMAT,None)   \
            or kwargs.pop(RCVS_KEY_OUTAXIS,self.__formatBlock))
        # perform the reading
        if fmtOut==Format.mod2format[gdal__name__]:
            # use the super class method (ibid with pool processing)
            return imagereader.ImageReader.readBlock(self,nblock)
            #return imagereader.ImageReader.readBlock(self,nblock)
        else:
            # the method otherwise is essentially the same as imagereader.ImageReader.readBlock,
            # modulo the possible output data formatting
            info, blockContainer = imagereader.ImageReader.readBlock(self,nblock)
            dType = Format.checkTypeBlock(kwargs.pop(RCVS_KEY_BLOCKBYTE,self.__dtypeBlock))
            module = module or self.__module # still possibly None
            return info, Format.formatContainer(blockContainer,
                **{RCVS_KEY_INAXIS: Format.mod2format[gdal__name__],
                   RCVS_KEY_OUTAXIS: fmtOut,
                   RCVS_KEY_BLOCKBYTE: dType,
                   RCVS_KEY_MODULE: module })


#/****************************************************************************/
# VectorReader overriding class
#/****************************************************************************/
class VectorReader(vectorreader.VectorReader):
    __formatBlock, __module = None, None
    __dtypeBlock = None

    #/************************************************************************/
    def __init__(self, arg, **kwargs):
        """Retrieve specific keyword arguments not to be passed to the superclass.

            >>> x = VectorReader(*args, **kwargs)
        """
        self.__formatBlock, self.__module =                                                 \
            Format.checkFormatBlock(kwargs.pop(RCVS_KEY_BLOCKFORMAT,None)              \
            or kwargs.pop(RCVS_KEY_OUTAXIS,gdal__name__))
        self.__dtypeBlock = Format.checkTypeBlock(kwargs.pop(RCVS_KEY_BLOCKBYTE,None))
        vectorreader.VectorReader.__init__(self, arg, **kwargs)

    #/************************************************************************/
    def rasterize(self, info, **kwargs):
        """Retrieve specific keyword arguments not to be passed to the superclass.

            >>> r = rasterize(info, **kwargs)
        """
        fmtOut, module = Format.checkFormatBlock(kwargs.pop(RCVS_KEY_BLOCKFORMAT,None) \
            or kwargs.pop(RCVS_KEY_OUTAXIS,self.__formatBlock))
        dType = Format.checkTypeBlock(kwargs.pop(RCVS_KEY_BLOCKBYTE,self.__dtypeBlock))
        module = module or self.__module
        return Format.formatContainer(vectorreader.VectorReader.rasterize(self, info), 
            **{RCVS_KEY_INAXIS: Format.mod2format[gdal__name__],
               RCVS_KEY_OUTAXIS: fmtOut,
               RCVS_KEY_BLOCKBYTE: dType,
               RCVS_KEY_MODULE: module})


#/****************************************************************************/
# ImageWriter overriding class
#/****************************************************************************/
class ImageWriter(imagewriter.ImageWriter):
    # additional attributes w.r.t. original class for use in data format/shape
    # conversion
    __formatBlock = None
    __dtypeBlock = None
    __filename, __drivername, __creationoptions = None, None, None
    __xsize, __ysize, __projection, __transform = None, None, None, None        
    __nbands, __gdaldatatype = None, None

    #/************************************************************************/
    def __init__(self, arg, **kwargs):
        """
            >>> x = ImageWriter(arg, **kwargs)
        """
        if arg is None: 
            raise IOError, 'missing output filename'
        else:
            self.__filename = arg
            self.ds = None    
            self.blocknum = 0 # start writing at the first block
        # retrieve info and firstblock (note that those keywords are defined in 
        # imagewriter.ImageWriter))
        info, firstblock = kwargs.pop('info',None), kwargs.get('fisrtblock')
        # retrieve and store the passed parameters
        self.__drivername = kwargs.pop('drivername',imagewriter.DEFAULTDRIVERNAME)
        self.__creationoptions = kwargs.pop('creationoptions',imagewriter.DEFAULTCREATIONOPTIONS) or []
        # extract information from provided info if not None, possibly overwritting it
        # (no error raising when 'conflictual' parameters are passed contrary to the
        # original implementation)
        self.__xsize, self.__ysize = kwargs.pop('xsize',None), kwargs.pop('ysize',None)
        self.windowxsize, self.windowysize = kwargs.pop('windowxsize',None), kwargs.pop('windowysize',None)
        if self.__xsize is None: 
            self.__xsize = info and info.getTotalSize()[0]
            self.windowxsize = self.windowxsize or (info and info.getWindowSize()[0])
        if self.__ysize is None:   
            self.__ysize = info and info.getTotalSize()[1]
            self.windowysize = self.windowysize or (info and info.getWindowSize()[1])
        self.__transform = kwargs.pop('transform',None) or (info and info.getTransform())
        self.__projection = kwargs.pop('projection',None) or (info and info.getProjection())
        self.overlap = kwargs.pop('overlap',None) or (info and info.getOverlapSize()) or 0 # shouldn't be None
        self.xtotalblocks, self.ytotalblocks = (None,None) if info is None else info.getTotalBlocks()
        self.__nbands, self.__gdaldatatype = kwargs.pop('nbands',None), kwargs.pop('gdaldatatype',None) 
        # format related information
        self.__formatBlock, _ = Format.checkFormatBlock(kwargs.pop(RCVS_KEY_BLOCKFORMAT,None)  \
            or kwargs.pop(RCVS_KEY_INAXIS,gdal__name__))
        self.__dtypeBlock = Format.checkTypeBlock(kwargs.pop(RCVS_KEY_BLOCKBYTE,None))
        if firstblock is not None: 
            # we still check some consistency
            if imagewriter.anynotNone([self.__nbands,self.__gdaldatatype]):
                raise rioserrors.ParameterError('Must pass one either firstblock or nbands and gdaltype, not all of them')
            if self.__formatBlock!=Format.mod2format[gdal__name__]:
                # possibly transform the first block passed to the Writer instance
                firstblock = Format.formatContainer(firstblock,
                        **{RCVS_KEY_INAXIS: self.__formatBlock,
                           RCVS_KEY_OUTAXIS: Format.mod2format[gdal__name__],
                           RCVS_KEY_BLOCKBYTE: self.__dtypeBlock,
                           RCVS_KEY_MODULE: gdal__name__})
            # RIOS only works with 3-d image arrays, where the first dimension is 
            # the number of bands. Check that this is what the user gave us to write. 
            if len(firstblock.shape) != 3:
                raise rioserrors.ArrayShapeError(
                    "Array dimension must be 3D instead of shape %s"%repr(firstblock.shape))
        # prepare already the dataset in the case enough parameters have been set
        self.prepare(firstblock=firstblock)       
        # if we have a first block then write it
        if firstblock is not None:          self.write(firstblock)
        return

    #/************************************************************************/
    def prepare(self, firstblock=None):
        """
            >>> x.prepare(firstblock=None)
        """
        if firstblock is not None:
            if self.windowxsize is None:    self.windowxsize = firstblock.shape[-2] - 2*self.overlap
            if self.windowysize is None:    self.windowysize = firstblock.shape[-1] - 2*self.overlap
            if self.__xsize<=0:
                if self.xtotalblocks is None: raise rioserrors.ParameterError('Missing number of blocks')  
                self.__xsize = self.windowxsize * self.xtotalblocks
            if self.__ysize<=0:
                if self.ytotalblocks is None: raise rioserrors.ParameterError('Missing number of blocks')  
                self.__ysize = self.windowysize * self.ytotalblocks
            
            (self.__nbands,y,x) = firstblock.shape # get the number of bands out of the block
            self.__gdaldatatype = imageio.NumpyTypeToGDALType(firstblock.dtype) # and the datatype
        # possibly update if it was not set before
        if self.xtotalblocks is None:   self.xtotalblocks = int(math.ceil(float(self.__xsize) / self.windowxsize))
        if self.ytotalblocks is None:   self.ytotalblocks = int(math.ceil(float(self.__ysize) / self.windowysize))
        if not imagewriter.allnotNone((self.__nbands, self.__gdaldatatype))     \
            or not imagewriter.allnotNone((self.__xsize, self.__ysize))         \
            or self.__xsize<=0 or self.__ysize<=0:
            return # do nothing, not enough parameters have been set
        # else: create the output dataset
        driver = gdal.GetDriverByName(self.__drivername)
        # check that specified driver has gdal create method and go create
        if driver is None:                              raise IOError, 'wrong driver definition'
        else:                                           metadata = driver.GetMetadata()
        if not metadata.has_key(gdal.DCAP_CREATE) or metadata[gdal.DCAP_CREATE]=='NO':
            raise TypeError,  'GDAL %s driver does not support Create(Copy) methods' % driver
        elif not imagewriter.allnotNone((self.__xsize, self.__ysize)): 
            raise rioserrors.ParameterError('Must pass information about output image size')                       
        if not isinstance(self.__filename,str):         raise IOError, 'wrong filemane definition'
        try:    
            self.ds = driver.Create(str(self.__filename), self.__xsize, self.__ysize, 
                                self.__nbands, self.__gdaldatatype, self.__creationoptions)
        except:
            raise rioserrors.ImageOpenError('missing internal parameters')
        if self.ds is None:
            raise rioserrors.ImageOpenError('Unable to create output file %s' % self.__filename)
        if self.__projection is not None:  
            self.ds.SetProjection(self.__projection)
        if self.__transform is not None:  
            self.ds.SetGeoTransform(self.__transform)
        return

    #/************************************************************************/
    def write(self, block, **kwargs):
        """
            >>> x.write(block, **kwargs)
        """
        # first shape the block container
        fmtIn, _ = Format.checkFormatBlock(kwargs.pop(RCVS_KEY_BLOCKFORMAT,None)   \
            or kwargs.pop(RCVS_KEY_INAXIS,self.__formatBlock))
        # we now adapt the write method to be able to perform multithread processing
        blockNum = kwargs.pop(RCVS_KEY_BLOCKNUM, None) # tell us where to write
        if blockNum is None:    
            blockNum = self.blocknum
        elif not (isinstance(blockNum,int) and blockNum>=0): 
            raise IOError, 'wrong block index %s' % blockNum
        dType = Format.checkTypeBlock(kwargs.pop(RCVS_KEY_BLOCKBYTE,self.__dtypeBlock))
        if fmtIn!=Format.mod2format[gdal__name__] or dType is not None:
            block = Format.formatContainer(block,
                **{RCVS_KEY_INAXIS: fmtIn,
                   RCVS_KEY_OUTAXIS: Format.mod2format[gdal__name__],
                   RCVS_KEY_BLOCKBYTE: dType,
                   RCVS_KEY_MODULE: gdal__name__})
        # possibly initialize the dataset with block considered as firstblock
        if self.ds is None:     self.prepare(firstblock=block)       
        # from here, it is copy/paste from write method
        yblock, xblock = blockNum // self.xtotalblocks, blockNum % self.xtotalblocks
        xcoord, ycoord = xblock * self.windowxsize,   yblock * self.windowysize
        self.writeAt(block, xcoord, ycoord)
        # we keep the internal blocknum increment (in the case blockNum was None)
        self.blocknum += 1 # use for serial writing

    
#/****************************************************************************/
# Job: class of methods for job definition and running
#/****************************************************************************/
class Job(collections.Callable):

    #/************************************************************************/
    def __getitem__(self, key):
        if isinstance(key,str) and key in RCVS_KEYLIST_WORK:
            try:        return self.job[key]
            except:     warnMessage('key not set in designed job %s' % key)
        else:
            raiseMessage(IOError,'unexpected job key')
        return
    #def __repr__(self):             return "<%s instance at %s>" % (self.__class__.__name__, id(self))
    def __str__(self):              return str(self.job)

    # properties
    @property
    def function(self):         return self.job[RCVS_KEY_FUNCTION]
    @property
    def module(self):           return self.job[RCVS_KEY_MODULE]
    @module.setter#analysis:ignore
    def module(self,module):    self.job.update({RCVS_KEY_MODULE: module})
    @property
    def inargs(self):           return self.job[RCVS_KEY_IN]
    @inargs.setter#analysis:ignore
    def inargs(self,inargs):    self.job.update({RCVS_KEY_IN: inargs})
    ##@property
    ##def outargs(self):          return self.job[RCVS_KEY_OUT]
    ##@outargs.setter
    ##def outargs(self,outargs):  self.job.update({RCVS_KEY_OUT: outargs})
    @property
    def args(self):             return self.job[RCVS_KEY_ARGS]
    @args.setter#analysis:ignore
    def args(self,args):        self.job.update({RCVS_KEY_ARGS: args})
    @property
    def kwargs(self):           return self.job[RCVS_KEY_KWARGS]
    @kwargs.setter#analysis:ignore
    def kwargs(self,kwargs):    self.job.update({RCVS_KEY_KWARGS: kwargs})
    @property
    def format(self):           return self.job[RCVS_KEY_BLOCKFORMAT]
    @format.setter#analysis:ignore
    def format(self,fmt):       self.job.update({RCVS_KEY_BLOCKFORMAT: fmt})
    @property
    def nin(self):              return len(self.job[RCVS_KEY_IN])
    @property
    def nout(self):             return len(self.job[RCVS_KEY_OUT])
    @property
    def nargs(self):            return len(self.job[RCVS_KEY_ARGS])
    @property
    def nkwargs(self):          return len(self.job[RCVS_KEY_KWARGS].items())
    
    #/************************************************************************/
    def __init__(self, *args):
        """
            >>> x.Job(*args)
        """
        # define a job as a dictionary
        self.job = dict(itertools.izip(RCVS_KEYLIST_WORK,args)) # {}
        ## self.job.update({RCVS_KEY_MODULE:userModule, RCVS_KEY_FUNCTION:userFunction, \
        ##                   RCVS_KEY_IN:inkeys, RCVS_KEY_OUT:inkeys,                  \
        ##                   RCVS_KEY_ARGS:otherArgs, RCVS_KEY_KWARGS:otherKwargs,
        ##                   RCVS_KEY_BLOCKFORMAT: fmtBlock})
        if args not in ((),None):               self.prepare(*args)
        
    #/************************************************************************/
    @staticmethod
    def find(userFunction):
        """
            >>> module, userFunction = x.find(userFunction)
        """
        def recfind(module, nUserFunction): # recursively look for module tree for method
            if hasattr(module, nUserFunction):
                return module
            for nmod, submod in inspect.getmembers(module, inspect.ismodule):
                if submod.__name__.startswith(module.__name__): # must be a specific submodule
                    result = recfind(submod, nUserFunction)
                    if isinstance(result,list):     result = result[0]
                    if result:                      return result, module # return the first method that matches
            return None
        if isinstance(userFunction,(list,tuple)):   return zip(*[Job.find(uf) for uf in userFunction])
        elif isinstance(userFunction,str):          nUserFunction = userFunction
        elif callable(userFunction):                nUserFunction = userFunction.__name__
        else:                                       nUserFunction = None
        if nUserFunction is not None:
            m = [re.match(mod,nUserFunction) for mod in RCVS_LIST_MODULES]
            if any(m):
                i = max([s.end() for s in m if s and s.start()==0])
                nUserFunction, mod = nUserFunction[i:], nUserFunction[:i]
                if nUserFunction.startswith('.'):   nUserFunction = nUserFunction[1:]
                module = recfind(__import__(mod), nUserFunction)
            else:
                module = [m for m in [recfind(__import__(mod), nUserFunction) for mod in RCVS_LIST_MODULES] if m]#analysis:ignore
                if module!=[]:      module = module[0] # we take the first matching module found
                else:               module = None
            if isinstance(module,tuple):            submodule, module = module
            else:                                   submodule = module
            if module is not None and callable(getattr(submodule,nUserFunction)):
                userFunction = getattr(submodule,nUserFunction)
        else:
            module = RCVS_NO_ARGUMENT # inspect.getmodule(userFunction)
        if module is RCVS_NO_ARGUMENT and not callable(userFunction):       raise IOError #TypeError
        return module, userFunction

    #/************************************************************************/
    def prepare(self, userWork, inkeys=None, *otherArgs, **otherKwargs):
        """
            >>> x.prepare(userWork, inkeys=None, *otherArgs, **otherKwargs)
        """
        # go through the userWork passed in input
        if isinstance(userWork,(list,tuple)) and len(userWork)==1:
            userWork = userWork[0]
        if callable(userWork) or isinstance(userWork,str):
            userModule, userFunction = self.find(userWork)
            # note that by default the output of a function will be overwritten over the
            # input, hence 'RCVS_KEY_IN:inkeys and RCVS_KEY_OUT:inkeys'
            self.set(userModule, userFunction, inkeys, inkeys, otherArgs, otherKwargs)
        elif isinstance(userWork,(list,tuple)):
            if isinstance(userWork,tuple):                          userWork = list(userWork)
            if not inspect.ismodule(userWork[0]):
                try:
                    self.module, shft = __import__(userWork[0]), 0
                except (ImportError,TypeError):
                    self.module, self.job[RCVS_KEY_FUNCTION] = self.find(userWork[0])
                    shft = +1
            # if still no module: problem
            if self.module is not None and not inspect.ismodule(self.module):
                raise TypeError, 'unrecognised user function %s' % self.module
            for i in range(1,len(userWork)):
                if isinstance(userWork[i],str) or callable(userWork[i]):
                    try:                                            userModule, userFunction = self.find(userWork[i])
                    except TypeError:                               self.job[RCVS_KEYLIST_WORK[i+shft]] = [userWork[i]] # pass
                    else:
                        if self.function is not None and userFunction!=self.function:
                            raise TypeError, 'list of functions provided'
                        elif userModule!=self.module:
                            warnings.warn('different module identified')
                        self.module, self.job[RCVS_KEY_FUNCTION] = userModule, userFunction
                elif isinstance(userWork[i],(list,tuple, dict)):    self.job[RCVS_KEYLIST_WORK[i+shft]] = userWork[i] #pass
                else:                                               self.job[RCVS_KEYLIST_WORK[i+shft]] = [userWork[i]]
        elif isinstance(userWork,dict):
            self.job = userWork
            # check and fill the missing fields with None
            if not set(self.job.keys()).issubset(set(RCVS_KEYLIST_WORK)):    
                raise TypeError, 'wrong key argument in work dictionary'
            else:          
                [self.job.update({key: RCVS_DEF_ARGUMENT})                            \
                for key in list(set(RCVS_KEYLIST_WORK).difference(set(self.job.keys())))]
            # possibly update the function and/or module
            if self.function in ((),None):
                raise TypeError, 'no method found'
            else:
                try:
                    self.module, self.job[RCVS_KEY_FUNCTION] = self.find(self.job[RCVS_KEY_FUNCTION])
                except TypeError:
                    raise TypeError, 'unrecognised user function %s' % self.job
        else:
            raise TypeError, 'unexpected input user function %s' % userWork
        # further checking
        if not (isinstance(self.kwargs,dict) or self.kwargs in (RCVS_NO_ARGUMENT,RCVS_DEF_ARGUMENT)):
            raise IOError, 'unexpected kwargs argument'
        elif not(isinstance(self.module,str) or inspect.ismodule(self.module) \
            or self.module in (RCVS_NO_ARGUMENT,RCVS_DEF_ARGUMENT)):
            raise IOError, 'unexpected module argument'
        elif not callable(self.function):
            raise IOError, 'unexpected function argument'
        # return the final dictionary defining the processing worflow
        return # self.job

    #/************************************************************************/
    def update(self, *otherArgs, **otherKwargs):
        """
            >>> x.update(*otherArgs, **otherKwargs)
        """
        # possibly set all the keys of RCVS_KEYLIST_WORK as default None value
        # in the dictionary
        [self.job.update({key: RCVS_DEF_ARGUMENT}) for key in RCVS_KEYLIST_WORK if key not in self.job]
        # at this stage, we possibly anticipate the format (shape, i.e. array axis arrangement)
        # of the block containers used when processing
        if Format.isCVModule(self.module):
            self.format = Format.mod2format[self.module.__name__]
        elif self.format==RCVS_DEF_ARGUMENT:
            # the default format conversion is: Id, ie. no (format/shape) modification of
            # the block container (see function Format.formatContainer) is performed:
            # the input argument is used as it is
            self.format = Format.KEY_NOFORMAT # Format.mod2format[gdal__name__]
            # in particular, it implicitely assumes that any function passed in the
            # userWork used the same array format/shape as Gdal ('zyx'); if this is not
            # the case:
            #   - either the  module should be added to the RCVS_LIST_MODULES list
            #   and a specific format transformation should be implemented (e.g., through
            #   Format.toAxis and Format.toByte methods),
            #   - or RCVS_KEY_BLOCKFORMAT should be explicitely defined
        elif self.format!=Format.KEY_NOFORMAT:   # leave as it has been passed, but check
            self.format, _ = Format.checkFormatBlock(self.format)
        # we finally update the module
        self.module = Format.whichCVModule(self.module)
        # update the different keys
        updateKeyWithDefault =  lambda key, val: self.job.update({key:val}) if self.job[key]==RCVS_DEF_ARGUMENT \
            else None
        updateKeyAsTuple =      lambda key:  self.job.update({key:tuple(self.job[key])}) if isinstance(self.job[key],(tuple,list))   \
            else (self.job.update({key:(self.job[key],)}) if self.job[key] not in (None,(),RCVS_NO_ARGUMENT)                    \
            else (self.job.update({key:RCVS_EMPTY_ARGUMENT}) if self.job[key]==RCVS_NO_ARGUMENT else None))
            #else (job.update({key:(job[key],)}) if isinstance(job[key],str) else None)
        for key, val in ((RCVS_KEY_IN,     RCVS_NO_ARGUMENT ), # RCVS_EMPTY_ARGUMENT
                         (RCVS_KEY_OUT,    RCVS_NO_ARGUMENT), # RCVS_EMPTY_ARGUMENT
                         (RCVS_KEY_ARGS,   otherArgs),
                         (RCVS_KEY_KWARGS, otherKwargs)):
            updateKeyWithDefault(key,val) # set with default if empty
        for key in (RCVS_KEY_IN,RCVS_KEY_OUT,RCVS_KEY_ARGS):
            updateKeyAsTuple(key) # make it a tuple
        return

    #/************************************************************************/
    def run(self, iBlocks, info, kwargs): # dummy function for dummy call
        """
            >>> res = x.run(iBlocks, info, kwargs)
        """
        return self.__call__(iBlocks, info, kwargs=kwargs)

    #/************************************************************************/
    def __call__(self, iBlocks, info, kwargs={}):
        updateContainer =   lambda b, **kw: b if not isinstance(b,Array) \
            else Format.formatContainer(b,**kw)
        updateArgs =        lambda args, bC, **kw: (tuple([updateContainer(bC[k],**kw) for k in range(len(bC))]),) + args \
            if isinstance(bC,(tuple,list))                                                                                  \
            else (updateContainer(bC,**kw),) + args
        # note that 'kw.update(...})' always returns None, so the second part of updateArgs (after
        # the 'or') will always be evaluated
        fmtOut = (self.format if self.format in Format.DataBlockFormat   \
                  else Format.mod2format[gdal__name__]) or kwargs.pop(RCVS_KEY_INAXIS,None)
        # if fmtOut is None, it is of the same format than previous calculation
        # configure the input arguments and keyword arguments
        kwargs.update({RCVS_KEY_OUTAXIS: fmtOut,
                      RCVS_KEY_MODULE: self.module})
        iargs, ikwargs = self.args, self.kwargs
        if ikwargs in (None, ()):           ikwargs = {}
        try:
            # read the list of input arguments starting from the end
            for key in self.inargs[::-1]: # possibly empty
                if key in ([],(),None):                         continue
                elif key not in iBlocks.__dict__.keys():         raise IOError, 'unknown parameter %s' % key
                kwargs.update({RCVS_KEY_INAXIS: iBlocks.__format__[key]})
                iargs = updateArgs(iargs, iBlocks.__dict__[key], **kwargs)
        except:
            raise IOError, 'wrong parameters setting'
        # run the function with the previously defined arguments
        try:    # through try/except with different arrangements of arguments
            try:
                argOut = self.function(*iargs, **ikwargs)
            except:
                ikwargs.update({RCVS_KEY_INFO: info})
                argOut = self.function(*iargs, **ikwargs)
        except (TypeError,ValueError): # retry while inverting the order of the arguments
            if self.nin==0:              raise IOError # no further testing
            iargs = iargs[self.nin:] + iargs[:self.nin] # reorder, but do not re-update!
            try:
                argOut = self.function(*iargs, **ikwargs)
            except:
                ikwargs.pop(RCVS_KEY_INFO)
                argOut = self.function(*iargs, **ikwargs)
        # we should have used sthg more 'pythonic' here, likewise 'with' statement
        return argOut

def pool_eval(f_args):
    """Takes a tuple of a function and args, evaluates and returns result
    
        >>> r = pool_eval(f_args)    
    """
    return f_args[0](*f_args[1:]) 
    
#/****************************************************************************/
# Workflow: class of methods for workflow definition and data processing
#/****************************************************************************/
class Workflow(collections.MutableSequence):

    #/************************************************************************/
    def __init__(self, *args):
        """
            >>> x.Workflow(*args)
        """
        # define a workflow as a list of job dictionaries
        self.workflow = [] # [Job(),]
        if len(args)==1 and isinstance(args[0],Job):   
            self.workflow = [args[0],]
        elif args not in ((),None):                   
            self.prepare(*args)
        
    #/************************************************************************/
    # http://docs.python.org/2/reference/datamodel.html#emulating-container-types
    def __getitem__(self, key):
        if isinstance(key,(int,slice)):
            if isinstance(key,int) and key < 0:     key = key + len(self)
            try:                return self.workflow[key]
            except:             raiseMessage(IndexError,'wrong job index in workflow')
        elif isinstance(key,str) and key in RCVS_KEYLIST_WORK:
            try:                return [job[key] for job in self.workflow]
            except:             warnMessage(IOError,'key not set in designed workflow')
        else:
            raiseMessage(TypeError,'unexpected key for workflow')
        return
    def __setitem__(self, position, job):
        if not isinstance(job,Job):     raiseMessage(IOError,'unexpected job for workflow')
        elif position<len(self):        self.workflow[position] = job
        elif position==len(self):       self.workflow.append(job)
        else:                           raiseMessage(IndexError,'unexpected position in workflow')
    def __delitem__(self, position):
        if not isinstance(position,int):raiseMessage(IOError,'unexpected job for workflow')
        elif position<len(self):        del self.workflow[position]
        else:                           raiseMessage(IndexError,'unexpected position in workflow')
    def __len__(self):                  return len(self.workflow)
    def __contains__(self, value):      return value in self.workflow
    def __iter__(self):
        # for wf in self.workflow):       yield wf
        return iter(self.workflow)  # self.workflow.__iter__()
    def __str__(self):
        return '\n'.join(['job #%s: %s' % (str(i),str(job)) for i,job in enumerate(self.workflow)])
    
    def insert(self, position, job):
        if not isinstance(job,Job):     raiseMessage(IOError,'unexpected job for workflow')
        elif position<len(self):        self.workflow.insert(position,job)
        elif position==len(self):       self.workflow.append(job)
        else:                           raiseMessage(IOError,'unexpected position in workflow')

    #/************************************************************************/
    # properties
    @property
    def njobs(self):                                return len(self) # len(self.workflow)
    @property
    def modules(self):  
        return tuple([mod for mod in self[RCVS_KEY_MODULE]           \
                    if mod not in ((),None,RCVS_NO_ARGUMENT)])
        
    #/************************************************************************/
    def prepare(self, workflow, infiles=None, outfiles=None, *otherArgs, **otherKwargs):
        """Define the list of all input/output arguments set in the input/output files.
        
            >>> x.prepare(workflow, infiles=None, outfiles=None, *otherArgs, **otherKwargs)
        """
        try:                                        inkeys = tuple(infiles.__dict__.keys())
        except AttributeError:                      inkeys = ()
        try:                                        outkeys = tuple(outfiles.__dict__.keys())
        except AttributeError:                      outkeys = ()
        # update the workflow dictionary
        try:
            workflow = list(workflow) if isinstance(workflow,tuple)                     \
                else ([workflow] if isinstance(workflow,str) else workflow) # make it a list
            while workflow:
                self.workflow.insert(0,Job(workflow.pop(), inkeys, *otherArgs, **otherKwargs))
            # this will raise a TypeError in the case workflow is a single dictionary
            # (as pop expects at least 1 argument; KeyError when there is an argument
            # with wrong key), and also a TypeError in the case workflow is a list 
            # describing a single work
        except TypeError: # KeyError
            workflow = Job(workflow, inkeys, *otherArgs, **otherKwargs)
            workflow.update(*otherArgs, **otherKwargs)
            self.workflow = (workflow,) # make it a list with single job
        except:
            raiseMessage(IOError, 'wrong parameter setting in workflow %s')
        else: # end of 'try' processing...
            # update accordingly
            [job.update(*otherArgs, **otherKwargs) for (i,job) in enumerate(self.workflow)]
        # define the list of all input/output arguments produced throughout the processing
        # workflow
        inargs, outargs = [Format.flattenIterable(reduce(lambda x,y: (x,)+(y,), [uw[key] for uw in self.workflow])) \
            for key in (RCVS_KEY_IN, RCVS_KEY_OUT)]
        # we assume that if the input and output have not been set anywhere in the processing workflow,
        # they must be added to the first and the last process respectively
        [self.workflow[pos].update({key: values})                                                       \
            if values!=RCVS_EMPTY_ARGUMENT and set(args).intersection(set(values))==set([])        \
            else None                                                                                   \
            for key, values, args, pos in ((RCVS_KEY_IN,inkeys,inargs,0),                          \
                                           (RCVS_KEY_OUT,outkeys,outargs,-1))
                                           ]
        return #self.workflow

    #/************************************************************************/
    def run(self, reader, vecreader, writer, controls, outfiles, outdata):
        """
            >>> res = x.run(reader, vecreader, writer, controls, outfiles, outdata)
            
        See also
        --------            
        http://learn-gevent-socketio.readthedocs.org/en/latest/general_concepts.html
        """
        progress, lock = {'pct':0, 'num':0}, None
        if controls.process=='pool':  # cpu is used for faster processing
            from multiprocessing import Manager, Lock
            try:                                
                # import dill, pathos
                from pathos.multiprocessing import ProcessingPool as Pool
            except ImportError: 
                raiseMessage(IOError,'ProcessingPool module import error')
            # we allocate the pool for as many threads as there are cores
            blockRange = RCVS_CPU_NODES-1
            pool = Pool(RCVS_CPU_NODES)  
            manager = Manager()
            progress, lock = manager.dict(progress), Lock()
            inBlockArgs, outBlockArgs = manager.Queue(), manager.Queue()  # multiprocessing.Queue() # create a shared queue
            outdata = manager.dict(outdata)
        elif controls.process=='pthread': # pool of threads
            try:                from multiprocessing.pool import ThreadPool  
            except ImportError: raiseMessage(IOError,'ThreadPool module import error')
            # note that the GIL single lock inside of the Python interpreter prevents
            # multiple threads from being executed in parallel, even on multi-core or
            # multi-CPU systems, hence no use for Tread
            blockRange = RCVS_CPU_NODES-1
            pool = ThreadPool(RCVS_CPU_NODES)  
            #from multiprocessing import Manager
            #manager = Manager(); outArgs =  manager.Queue()
            inBlockArgs, outBlockArgs = [], Queue.Queue() 
        ## elif controls.process=='ipool':
        ##     raiseMessage(IOError,'IPython not implemented')
        ##     try:    from IPython.parallel import Client
        ##     except: raiseMessage(IOError,'IPython module import error')
        ##     try:        pool = Client()[:]
        ##     except:     raiseMessage(IOError,'IPython: start the cluster first')
        ##     pool.use_dill()
        elif controls.process in ('single','serial'):
            blockRange = RCVS_DEF_BLOCKRANGE if controls.process=='serial' else 1
            inBlockArgs, outBlockArgs = [], []
            pass
        # define 'shortcut functions
        #---------------------------------------------------------------------#      
        def loadBlock(*args):
            readerIter, inBlockArgs = args
            return self.readBlock(readerIter, vecreader, inBlockArgs, blockRange=blockRange)
        #---------------------------------------------------------------------#
        def runBlock(*args):
            inBlockArgs, outBlockArgs = args
            ## try:
            # main processing function
            ninfo = self.applyBlock(inBlockArgs, self.workflow, outfiles, outdata, outBlockArgs)
            # update the progress bar
            if controls.progress and ninfo:   self.updateProgress(ninfo, controls, progress, lock)
            ## except:
            ##     [operator.close for operator in (reader,vecreader,writer)\
            ##               if operator not in (None,(),(RCVS_NONE2_LIST,))]
            ##     raise IOError, 'error in block processing'
            ## else:
            return 0
        #---------------------------------------------------------------------#
        def saveBlock(*args):
            outBlockArgs, outfiles, outdata = args # write in outfiles and outdata
            return self.writeBlocks(outBlockArgs, writer, outfiles, outdata)            
        #---------------------------------------------------------------------#      
        # play with progress
        if controls.progress is not None:
            controls.progress.setProgress(0)
            controls.progress.setTotalSteps(100) 
        # create an ImageIterator to read through
        if reader==(RCVS_NONE2_LIST,): readerIter = imagereader.ImageIterator(None)
        else:                               readerIter = reader.__iter__()
        # run the block processing function defined in blockProcess over the different
        # blocks in a single, sequential or parallel (either CPU or GPU based) procedure
        if controls.process in ('single','serial'):
            # sequentially perform:
            isReaderIterable = True
            while readerIter.nblock<len(reader) and isReaderIterable:
                # note that alternative solution can be possibly implemented using 
                # itertools.imap, itertools.izip, itertools.repeat...
                isReaderIterable = loadBlock(readerIter, inBlockArgs) == blockRange
                runBlock(inBlockArgs, outBlockArgs)
                saveBlock(outBlockArgs, outfiles, outdata) 
        elif controls.process in ('pool','pthread'): 
            # create an initial worker for writing on the files whenever the output 
            # queue is filled (as concurrent access by multiple processes to the same
            # file may actually slow down the reading of the file)
            ## pool.apply_async(self.writeBlocks, args=(outArgs, writer, outfiles, outdata))
            pool.apply_async(saveBlock, args=(outBlockArgs, outfiles, outdata))
            # create workers for loading the data and running the processing
            if inBlockArgs is not None:      
                unpickle = Format.definePicklable(inBlockArgs, unpick=True, block=False)
            while readerIter.nblock<len(reader):
                # provide the pool with one worker for loading a bunch of data, 
                # and run it asap
                ## blockNum = pool.apply_async(self.readBlock, 
                ##                             args=(readerIter, vecreader, inArgs, blockRange)).get()
                blockNum = pool.apply_async(loadBlock, args=(readerIter, inBlockArgs)).get()
                # provide the pool with the jobs to run as soon as a thread is available
                # after the input queue has been filled
                ## results = [pool.apply_async(self.applyBlock, 
                ##                             args=(unpickle(inArgs), self.workflow, outfiles, outdata, outArgs)) 
                ##                 for i in range(blockNum)]
                results = [pool.apply_async(runBlock, args=(unpickle(inBlockArgs), outBlockArgs)) 
                                for i in range(blockNum)]
                # collect results from the workers through the pool results queue
                [job.get(timeout=None) for job in results]
            # close properly the queue with None
            if outBlockArgs is not None:    Format.definePicklable(outBlockArgs)(outBlockArgs,None)
            # close and join the pool
            pool.close()
            pool.join()
        if controls.progress is not None: 
            controls.progress.setProgress(100)
        return outdata

    #/************************************************************************/
    # readBlock method for sequential reading of the input file(s)
    @staticmethod
    def readBlock(reader, vecreader, inBlockArgs=None, blockRange=None):
        """
            >>> blockRange = x.readBlock(reader, vecreader, inBlockArgs=None, blockRange=None)
        """
        if reader is None:                  blockRange = 1
        elif isinstance(reader,ImageReader):blockRange, reader = None, reader.__iter__()# read it entirely      
        elif not isinstance(reader, imagereader.ImageIterator):
            raise IOError, 'wrong input reader iterator in readBlock'
        if blockRange is None:          blockRange = len(reader.reader) # read it till the end
        # set the args where to store the blocks
        if inBlockArgs is None:          inBlockArgs = []
        pickle = Format.definePicklable(inBlockArgs)
        # the block are sequentially read at stage
        for i in range(blockRange):
            iBlock = BlockAssociations()
            try:                        
                blockNum = reader.nblock
                blockInfo, blockContainer = reader.next() # reader.nblock is incremented at this stage only
            except AttributeError:    #  AttributeError when reader is None
                # we still provide with a block because we are dealing here with 
                # functions that don't take any image argument
                pickle(inBlockArgs, (0, None, iBlock)) # dummy
                return i
            except StopIteration:    # StopIteration when reader there is nothing more to read
                return i # nothing more to read 
            # initialise the input blocks
            if blockContainer not in (None,()):
                iBlock.__dict__.update(blockContainer)
                # create a dictionary with the intrinsic format/shape of the considered arrays
                iBlock.formatUpdate(blockContainer.keys(),Format.mod2format[gdal__name__])
            if vecreader not in (None,(),(RCVS_NONE2_LIST,)):
                vecBlock = vecreader.rasterize(blockInfo)
                iBlock.__dict__.update(vecBlock)
                iBlock.formatUpdate(vecBlock.keys(),Format.mod2format[gdal__name__])
                vecBlock = None
            # get rid of elements that cannot be serialized
            if blockInfo not in (None,()):
                blockInfo.blocklookup = None
                blockInfo.loggingstream = None
            # the block is added to the queue/list
            pickle(inBlockArgs, (blockNum, blockInfo, iBlock))
        return blockRange

    #/************************************************************************/
    # applyBlock: single block processing method
    @staticmethod
    def applyBlock(inBlockArgs, *args):
        """
            >>> blockNum, blockInfo, oBlock = x.applyBlock(inBlockArgs, *args)
        """
        if inBlockArgs in ((),[],None):             
            return RCVS_NONE2_LIST
        elif isinstance(inBlockArgs,tuple): 
            blockNum, blockInfo, iBlock = inBlockArgs
        else:   # then we expect to have a queue/list of arguments
            try:            unpickle = Format.definePicklable(inBlockArgs,unpick=True)
            except (IndexError,IOError): raise IOError, 'wrong arguments in applyBlock'
            while True:   
                try:            blockArgs = unpickle(inBlockArgs)
                except (IndexError,Queue.Empty):                    break
                else:           res = Workflow.applyBlock(blockArgs, *args)
            return res
        if len(args)==3:        args += (None,)  
        elif len(args)!=4:      raise IOError, 'wrong number of input arguments'  
        workflow, outfiles, outdata, outBlockArgs = args
        # internal lambda
        updateBlock =       lambda block, key, value: block.__dict__.update({key: value})
        # internal variables
        if not(outfiles in (None,RCVS_NO_ARGUMENT) or outfiles.__dict__.keys() in ([],None,RCVS_NO_ARGUMENT)):
            outputKeys = outfiles.__dict__.keys()
        else:   outputKeys = []
        if not(outdata in (None,RCVS_NO_ARGUMENT) or outdata.keys() in ([],None,RCVS_NO_ARGUMENT)):    
            returnKeys = outdata.keys()
        else:   returnKeys = []
        if blockInfo is not None and blockInfo.loggingstream is None:     
            blockInfo.loggingstream = sys.stdout
        inkeys = [uw[RCVS_KEY_IN] for uw in workflow] # list (of list) of input arguments
        kw = {RCVS_KEY_INAXIS: Format.mod2format[gdal__name__]}
        nJobs = len(workflow)
        # initialise the output blocks
        oBlock = BlockAssociations()
        [oBlock.__dict__.update({key:RCVS_NO_ARGUMENT}) for key in outputKeys]
        # loop over the different provided functions to perform block processing
        for iJob in range(nJobs):
            # run the job
            argOut = workflow[iJob].run(iBlock, blockInfo, kw) # same as workflow[iJob](iBlock, info, kw)
            # store the output
            if argOut is None:
                continue # nothing is returned by the function: go to next work
            elif workflow[iJob][RCVS_KEY_OUT] in (RCVS_NO_ARGUMENT,RCVS_EMPTY_ARGUMENT):
                continue # nothing is requested to be returned: go to next work
            elif isinstance(argOut,tuple):
                argOut = list(argOut)
                if len(argOut)!=len(workflow[iJob][RCVS_KEY_OUT]):
                    raise IOError, 'wrong number of output arguments for %s' % workflow[iJob][RCVS_KEY_FUNCTION].__name__
                for key in workflow[iJob][RCVS_KEY_OUT][::-1]:
                    testing = iJob<nJobs-1 or oBlock is None # key not in oBlock.__dict__.keys() or oBlock is None
                    updateBlock(iBlock if testing else oBlock, key, argOut.pop())
                    (iBlock if testing else oBlock).formatUpdate(key, kw[RCVS_KEY_OUTAXIS])
            else:
                if len(workflow[iJob][RCVS_KEY_OUT])!=1:
                    raise IOError, 'wrong number of output arguments for %s' % workflow[iJob][RCVS_KEY_FUNCTION].__name__
                key = workflow[iJob][RCVS_KEY_OUT][0]
                testing = iJob<nJobs-1 or oBlock is None # key not in oBlock.__dict__.keys() or oBlock is None
                updateBlock(iBlock if testing else oBlock, key, argOut)
                (iBlock if testing else oBlock).formatUpdate(key, kw[RCVS_KEY_OUTAXIS])
            # update
            kw[RCVS_KEY_INAXIS] = kw[RCVS_KEY_OUTAXIS]
            # clean
            if inkeys not in ([],()):                           inkeys.pop(0)
            # do some cleaning
            for key in workflow[iJob][RCVS_KEY_IN]:
                if key not in Format.flattenIterable(inkeys+returnKeys+outputKeys): #oBlock.__dict__.keys()
                    iBlock.__dict__.pop(key)
        # update the output oBlock before writing after the last processing by
        # retrieving variables that have been temporarly stored in iBlock  as an 
        # intermediate result
        if oBlock is not None:
            # - either an output arg
            [updateBlock(oBlock, key, iBlock.__dict__.pop(key)) or oBlock.formatUpdate(key, iBlock.__format__.pop(key)) \
                  for (key,value) in oBlock.__dict__.items() if value is None and key in iBlock.__dict__]
            # - or a returned arg 
            [updateBlock(oBlock, key, iBlock.__dict__.pop(key)) or oBlock.formatUpdate(key, iBlock.__format__.pop(key)) \
                  for key in returnKeys if (key not in oBlock.__dict__ and key in iBlock.__dict__)]
            # note the use of pop to avoid deep copying    
        iBlock = None
        # serialisation issue:
        # note that qpickle(queue,(outfiles, oBlock, controls, info, blockNum)) with standard
        # multiprocessing module (instead of pathos) leads to cPickle.PicklingError; indeed
        # info is a ReaderInfo object that cannot be passed as it is because it is not pickeable:
        # it has in its __dict__ various proxy of <Swig Object of Type 'GDALDatasetShadow'> 
        # fields, namely:
        #    <osgeo.gdal.Dataset; proxy of <Swig Object of type 'GDALDatasetShadow>>)
        if outBlockArgs is not None:
            try:                        pickle = Format.definePicklable(outBlockArgs)
            except (IndexError,Queue.Empty):    raise IOError, 'unexpected argument'
            pickle(outBlockArgs,(blockNum, blockInfo, oBlock))
            return (blockNum, blockInfo) # avoid overload, we do not pass oBlock
            # note that nested classes also make pickle fail, since it relies on the path
            # of the object inside the application
            # https://www.frozentux.net/2010/05/python-multiprocessing/
        else:
            return (blockNum, blockInfo, oBlock)           

    #/************************************************************************/
    # writeBlocks overidding method writeOutputBlocks enabling for writing or saving:
    #   - of computed intermediary outputs,
    #   - in parallel processes.
    @staticmethod
    def writeBlocks(outBlockArgs, *args):
        """
            >>> x.writeBlocks(outBlockArgs, *args)
            
        Note
        ----
        With respect to original :meth:`writeOutputBlocks`, it enables for writing
        or saving:
        - of computed intermediary outputs,
        - in parallel processes.
       
       """
        #if len(args)==1 and args[0] is None:    return
        if outBlockArgs is None:                                    return
        elif isinstance(outBlockArgs,tuple):
            blockNum, blockInfo, oBlock = outBlockArgs
        else:   # then we expect to have a queue/list of arguments
            try:            unpickle = Format.definePicklable(outBlockArgs,unpick=True,block=True)
            except (IndexError,IOError): raise IOError, 'wrong arguments in writeBlocks'
            while True:   
                try:                blockArgs = unpickle(outBlockArgs) 
                except (IndexError,Queue.Empty):                    break
                if blockArgs is None:                               break
                else:               Workflow.writeBlocks(blockArgs, *args)
            return
        writer, outfiles, outdata = args
        # first write the blocks if not empy into the disk
        if writer=={} or outfiles is None or outfiles.__dict__=={}          \
                or oBlock is None or oBlock.__dict__.values() in ([],None):
            pass
        else:
            # the final conversion into gdal format is done right before writing
            updateContainer = lambda b, **kw: b if not isinstance(b,Array) else Format.formatContainer(b,**kw)
            kw = {RCVS_KEY_INAXIS: None, RCVS_KEY_OUTAXIS: gdal__name__}
            # from here, it is similar to the original applier.writeOutputBlocks
            for name in outfiles.__dict__.keys(): # outputKeys
                outfileName = getattr(outfiles, name)
                if outfileName in ([],[None],None,RCVS_NO_ARGUMENT):    continue # most likely one of the 'returnKeys'
                if name not in oBlock.__dict__:
                    # the output is not an array, most likely the workflow returned the final 
                    # process status as an output.
                    # still, this should be handled by the user in order to avoid this type of error
                    raise rioserrors.KeysMismatch('Output key %s not found in output blocks' % name)
                if hasattr(oBlock,'__format__'): # to ensure interoperability with original BlockAssociations blocks
                    kw.update({RCVS_KEY_INAXIS: oBlock.__format__[name]})
                else:
                    kw.update({RCVS_KEY_INAXIS: gdal__name__})
                block = updateContainer(oBlock.__dict__[name], **kw)
                if name not in writer:
                    raise rioserrors.ImageOpenError('writer not defined for key %s' % name)   
                if isinstance(outfileName, list):
                    numFiles, numBlocks = len(outfileName), len(block) #oBlock.__dict__[name])
                    if numBlocks != numFiles:
                        raise rioserrors.MismatchedListLengthsError(("Output '%s' writes %d files, "+
                            "but only %d blocks given")%(name, numFiles, numBlocks))
                    # store the block index
                    for i in range(numFiles):
                        # here lies another difference with original applier.writeOutputBlocks
                        writer[name][i].write(block[i], **{RCVS_KEY_BLOCKNUM: blockNum})
                else:
                    # ibid as previous comment
                    writer[name].write(block, **{RCVS_KEY_BLOCKNUM: blockNum})
        # second, save the possible desired outptuts
        if outdata in ({},None,RCVS_NO_ARGUMENT) or outdata.keys()==[]:    
            pass
        else:
            if blockInfo is not None:
                try:        xnblocks,ynblocks = blockInfo.getTotalBlocks()
                except:     xnblocks,ynblocks = 1, 1
            else:
                try:                        key = outdata.keys()[0] # returnKeys[0]
                except IndexError:          return # outdata is empty
                except:                     raise IOError, 'unexpected argument for output dictionary'
                else:                       keyBlocks = outdata[key].keys()
                try:    xnblocks, ynblocks = [max([k[i] for k in keyBlocks])+1 for i in (0,1)]
                except: xnblocks,ynblocks = 1, 1 # non 'block-dependent' outputs
            if xnblocks*ynblocks==1:                                                pass
            else:                           blockNum = (blockNum//xnblocks, blockNum%xnblocks)
            if re.search ('Proxy', outdata.__class__.__name__):
                for key in outdata.keys():  # to ensure compatibility with Manager  
                    # Manager proxy objects are unable to propagate changes made to mutable objects
                    # inside a container. In the case of a manager.dict() object, any changes to the
                    # managed dict itself are propagated to all the other processes. But if there is 
                    # a dict inside a key, any changes to the inner dict are not propagated, because 
                    # the manager has no way of detecting the change. 
                    o = outdata[key]
                    o.update({blockNum: oBlock.__dict__[key]}) 
                    outdata[key] = o
            else:
                [outdata[key].update({blockNum: oBlock.__dict__[key]}) for key in outdata.keys()]
        return
    
    #/************************************************************************/
    # updateProgress: overriding method
    @staticmethod
    def updateProgress(arg, controls=None, progress=None, lock=None):
        """
            >>> x.updateProgress(arg, controls=None, progress=None, lock=None)
        """
        try:            blockNum, blockInfo = arg[:2]
        except:         return
        if controls is None or blockInfo is None or progress in (None,{})   \
            or not re.search('dict',progress.__class__.__name__.lower()):      
            return
        elif controls.process!='pool' or progress['num'] is None or lock is None:
            if progress['num'] is not None:   progress.update({'num':progress['num']+1})
            progress.update({'pct':applier.updateProgress(controls, blockInfo, progress['pct'])})
        else:
            with lock:
                progress.update({'num':progress['num']+1})
                pct = int(float(progress['num'])/float(blockInfo.xtotalblocks * blockInfo.ytotalblocks) * 100)
                if pct != progress['pct']:      controls.progress.setProgress(pct)
                progress.update({'pct':pct})
        return
        

#/****************************************************************************/
# ApplierControls overriding class
#/****************************************************************************/
class ApplierControls(applier.ApplierControls):

    def __init__(self):
        """
            >>> c = ApplierControls()
        """
        applier.ApplierControls.__init__(self)
        # retrieve/set other useful variable
        # variables considered for reading or processing
        self.process, self.dType, self.returnArgs = 'serial', None, []
        # variables considered for writing
        self.xsize, self.ysize = None, None      
        self.nbands, self.gdaldatatype = None, None
        self.transform, self.projection = None, None

    def setProcess(self, method):
        """
            >>> c.setProcess(method)
        """
        if isinstance(method,str) and method in RCVS_LIST_BLOCKPROCESS:
            self.process = method or 'serial'
        elif method is not None:        raise IOError, 'wrong process method: %s' % method
    def setGdalDType(self, gdaldtype):
        """
            >>> c.setGdalDType(gdaldtype)
        """
        if isinstance(gdaldtype,str) and gdaldtype in Format.GdalTypes: 
            gdaldtype = eval('gdal.GDT_'+gdaldtype)
        if isinstance(gdaldtype,int) and gdaldtype in [dt[1] for dt in imageio.dataTypeMapping]:
                self.gdaldatatype = gdaldtype
        elif gdaldtype is not None:     raise IOError, 'wrong Gdal data type: %s' % gdaldtype
    def setDType(self, dType):
        """
            >>> c.setDType(dType)
        """
        if isinstance(dType,str) and dType in Format.DataBlockTypes:
            self.dType = dType
        elif dType is not None:         raise IOError, 'wrong data type: %s' % dType
    def setNbands(self, nbands):
        """
            >>> c.setNbands(nbands)
        """
        if isinstance(nbands,int):        
            self.nbands = nbands
        elif nbands is not None:        raise IOError, 'wrong nbands variable: %s' % nbands
    def setXYsize(self, xsize, ysize):              
        """
            >>> c.setXYsize(xsize, ysize)
        """
        if (xsize is not None and not isinstance(xsize,int))            \
            or (ysize is not None and not isinstance(ysize,int)):
                raise IOError, 'wrong size variables for setXYsize: %s' % (xsize, ysize)
        self.xsize, self.ysize = xsize, ysize
    def setReturnArgs(self, returnArgs):            
        """
            >>> c.setReturnArgs(returnArgs)
        """
        self.returnArgs = returnArgs
    def setTransform(self, transform):              
        """
            >>> c.setTransform(transform)
        """
        if (isinstance(transform,(tuple,list)) and len(transform)!=6):
            self.__transform = transform
        elif transform is not None:  raise IOError, 'wrong transform variable: %s' % transform
    def setProjection(self, projection):            
        """
            >>> c.setProjection(projection)
        """
        self.projection = projection


#/****************************************************************************/
# BlockAssociations overriding class
#/****************************************************************************/
class BlockAssociations(applier.BlockAssociations): 
    # introduce an attribute to store the intrinsic format (shape) of the data
    # contained in the block(s)
    __format__ = {}
    def formatUpdate(self, keys, fmt):
        """
            >>> BlockAssociations.formatUpdate(keys, fmt)
        """
        if isinstance(keys,str):                        keys = [keys]
        elif not isinstance(keys,list):                 keys = list(keys)
        self.__format__.update(dict((k,fmt if isinstance(self.__dict__[k],Array) else None)    \
            for k in keys))
        return


#/****************************************************************************/
# apply overriding method
#/****************************************************************************/
def apply(userFunction, infiles, outfiles=None, otherArgs=None, controls=None, **otherKwargs):
    """
        >>> apply(userFunction, infiles, outfiles=None, otherArgs=None, controls=None, **otherKwargs)
                
    **Note**
              
    :meth:`rcvs.cvapplier` is a set of functions/methods/classes aiming at overriding the
    original |RIOS| :meth:`apply` function for chain processing, while also overriding
    other classes from that module so that they can be used transparently. Hence, it is 
    essentially a wrapper as most of the original IO functionalities have been preserved.

    """
    # note that we keep 'apply parametrisation order and further add keyword arguments
    
    # get default controls object if none given
    if controls is None:                    controls = ApplierControls()

    # update controls variable with internal cvapplier specifications for input
    # reading and processing
    controls.setProcess(otherKwargs.pop(RCVS_KEY_BLOCKPROCESS, None))
    controls.setDType(otherKwargs.pop(RCVS_KEY_BLOCKBYTE,None))
    controls.setReturnArgs(otherKwargs.pop(RCVS_KEY_RETURNARGS, None))
    # update controls variable with cvapplier specifications for output writing
    # in the case of 'overlap', the variable passed in otherKwargs may overwrite
    # the one possibly passed in controls (for compatibility with original code)
    controls.setOverlap(otherKwargs.pop(RCVS_KEY_MARGIN, controls.overlap))
    ## controls.setGdalDType(otherKwargs.pop('gdaldatatype',controls.gdaldatatype))
    ## controls.setNbands(otherKwargs.pop('nbands', controls.nbands))
    ## controls.setXYsize(otherKwargs.pop('xsize', controls.xsize),otherKwargs.pop('ysize', controls.ysize))       
    ## controls.setTransform(otherKwargs.pop('transform', controls.transform))              
    ## controls.setProjection(otherKwargs.pop('projection', controls.projection))
    
    # check that the multiprocessing can be performed as requested
    if controls.process=='pool' and pathos.__name__ not in RCVS_LIST_MODULES:
        controls.process = 'pthread'
        warnMessage('processing method reset to %s' % controls.process)
    elif controls.process in ('pool','pthread') and RCVS_CPU_NODES==1:
        controls.process = 'serial'
        warnMessage('processing method reset to %s' % controls.process)
    
    # define the workflow adopted
    if otherArgs is None:                               otherArgs = ()

    # define the Workflow instance to store all the different jobs
    if isinstance(userFunction, Workflow):                          workflow = userFunction
    else:   workflow = Workflow(userFunction, infiles, outfiles, *otherArgs, **otherKwargs)
    # at that stage, workflow is a Workflow of processes
    
    if len(workflow)==1: #isinstance(workflow,dict):
        module = workflow[0][RCVS_KEY_MODULE]
        if False and controls.process=='serial' and controls.returnArgs is None                                                                                       \
           and (module is None or not Format.isCVModule(module.__name__)):
            try:        applier.apply(userFunction, infiles, outfiles,
                                      otherArgs=otherArgs, controls=controls)
            except:     raise IOError, 'unrecognised method %s' % userFunction
            else:       return
        else:
            pass

    #/************************************************************************/
    # separateVectors overidding method
    def separateVectors(infiles):
        def filexists(filepath):  # case insensitive file existence checker
            if os.name=='nt':   return os.path.exists(filepath) # Windows is case insensitive
            path, name = os.path.split(os.path.abspath(filepath))
            for f in os.listdir(path):
                if re.search(f, name, re.I):    return True
            return False
        # make a test on the input list of filenames (this should have been done in
        # InputCollection, but we avoid redefining that one as well)
        if infiles.__dict__!={}:                     files = infiles.__dict__.values()
        else:   return (applier.FilenameAssociations(), applier.FilenameAssociations())
        reduceliststrings = lambda x,y: ((x,) if isinstance(x,str) else x) + ((y,) if isinstance(y,str) else y)
        if len(files)>1:                    files = reduce(reduceliststrings, files)
        if files in ([],None):
            raise IOError, 'empty list of string filenames in list of arguments %s' % files
        elif not all([isinstance(f,str) and filexists(f) for f in files]):
            raise IOError, 'one or more non existing file in list of arguments %s' % files
        return applier.separateVectors(infiles)
    #---------------------------------------------------------------------#
    # readers definition method
    def readerDefine(infiles, controls):
        reader, vecreader = RCVS_NO_ARGUMENT, RCVS_NO_ARGUMENT
        # use separateVectors defined herein for checking the existence of the
        # input files
        if infiles in ((),[],RCVS_NO_ARGUMENT):
            return reader, vecreader
        (imagefiles, vectorfiles) = separateVectors(infiles)
        # use ImageReader defined herein for passing possible 'singleBlock' option
        kw = {RCVS_KEY_BLOCKSINGLE: controls.process=='single',
              RCVS_KEY_BLOCKBYTE: controls.dType}
        reader = ImageReader(imagefiles.__dict__, controls.footprint,
            controls.windowxsize, controls.windowysize, controls.overlap,
            controls.statscache, loggingstream=controls.loggingstream, **kw)
        #controls.windowxsize, controls.windowysize = reader.windowxsize, reader.windowysize
        if len(vectorfiles) > 0:
            # take into account possible request made into controls
            kw.update({RCVS_KEY_BLOCKBYTE: controls.vectordatatype})
            vectordict = applier.makeVectorObjects(vectorfiles, controls)
            # use VectorReader defined herein
            vecreader = VectorReader(vectordict, progress=controls.progress, **kw)
        # resampling option: no change, this is handled 'outside' by gdalwarp
        applier.handleInputResampling(imagefiles, controls, reader)
        return reader, vecreader
    #---------------------------------------------------------------------#
    def writerDefine(outfiles, controls, info):
        writer = {}
        if outfiles is None or outfiles.__dict__=={}:           return writer
        if info is not None: # serialize info
            info.blocklookup, info.loggingstream = None, None
        kwargs = {'info': info}
        for name in outfiles.__dict__.keys():
            kwargs.update({'drivername': controls.getOptionForImagename('drivername', name),
                          'creationoptions': controls.getOptionForImagename('creationoptions', name),
                          'nbands': controls.getOptionForImagename('nbands', name),
                          'gdaldatatype': controls.getOptionForImagename('gdaldatatype', name),
                          'xsize': controls.getOptionForImagename('xsize', name),
                          'ysize': controls.getOptionForImagename('ysize', name),
                          'transform': controls.getOptionForImagename('transform', name),
                          'projection': controls.getOptionForImagename('projection', name)})
            outfileName = getattr(outfiles, name)
            if outfileName in ([],[None],None,RCVS_NO_ARGUMENT):    continue # most likely one of the 'returnArgs'
            if isinstance(outfileName, list):
                writer[name], numFiles = [], len(outfileName)
                for i in range(numFiles):
                    filename = outfileName[i]
                    # use the newly defined ImageWriter
                    singlewriter = ImageWriter(filename, **kwargs)
                    writer[name].append(singlewriter)
                    if controls.getOptionForImagename('thematic', name):    singlewriter.setThematic()
                    layernames = controls.getOptionForImagename('layernames', name)
                    if layernames is not None:                  singlewriter.setLayerNames(layernames)
            else:
                # ibid as previous comment
                singlewriter = ImageWriter(outfileName, **kwargs)
                writer[name] = singlewriter
                if controls.getOptionForImagename('thematic', name):        singlewriter.setThematic()
                layernames = controls.getOptionForImagename('layernames', name)
                if layernames is not None:                      singlewriter.setLayerNames(layernames)
        return writer
   #---------------------------------------------------------------------#
        
    # define the readers
    reader, vecreader = readerDefine(infiles, controls)
    
    # initialize the writer dictionary
    writer = writerDefine(outfiles, controls, None if reader is None else reader[0][0])
    
    #---------------------------------------------------------------------#
    def resultDefine():
        outdata = {}
        # determine if any intermediary (e.g. non image) output should be returned
        # by the program
        if controls.returnArgs is None:                     return outdata
        elif isinstance(controls.returnArgs,str):           controls.returnArgs = (controls.returnArgs,)
        elif not isinstance(controls.returnArgs,tuple):     controls.returnArgs = tuple(controls.returnArgs)
        if reader!=RCVS_NO_ARGUMENT:   inargs = tuple(reader.imageContainer.keys())
        else:                               inargs = ()
        outargs = Format.flattenIterable(reduce(lambda x,y: (x,)+(y,), [uw[RCVS_KEY_OUT] for uw in workflow]))
        # return only those arguments that have been calculated in the workflow
        controls.returnArgs = list(set(controls.returnArgs).intersection(set(outargs+inargs)))
        if reader not in ((),None,RCVS_NO_ARGUMENT,):
            xnblocks, ynblocks = reader.info.getTotalBlocks() #nblocks = len(reader)
            yxblocks = [0] if ynblocks*xnblocks==1 else [(y,x) for x in range(xnblocks) for y in range(ynblocks)]
        for key in controls.returnArgs:
            try:    outdata.update({key: dict(itertools.izip_longest(yxblocks, [None]))})
            ##outdata.update({key: np.empty([1] if ynblocks*xnblocks==1 else [ynblocks,xnblocks],
            ##                                      dtype=object)})
            except: outdata.update({key: dict()}) #[RCVS_NO_ARGUMENT]
        return outdata
    #---------------------------------------------------------------------#

    # define a special result variable outdata to be returned with items storing
    # all other desired outputs
    outdata = resultDefine()

    # case the input image is not to be read yet
    if reader==RCVS_NO_ARGUMENT:               reader = (RCVS_NONE2_LIST,) # or((None,),)
    # case no vector file provided
    if vecreader==RCVS_NO_ARGUMENT:        vecreader = (RCVS_NONE2_LIST,) #((None,),)

    # main block processing
    workflow.run(reader, vecreader, writer, controls, outfiles, outdata)

    # close the files
    [operator.close for operator in (reader,vecreader)                       \
            if operator not in (None,(),(RCVS_NONE2_LIST,))] 
    reader, vecreader = None, None
    
    # update/write the output files
    if not (outfiles is None or writer in ({},None)):
        try:    applier.closeOutputImages(writer, outfiles, controls)
        except: raise IOError, 'error closing output files'
    writer = None
    
    # possibly reduce outdata
    if controls.returnArgs is not None and outdata is not None:
        [outdata.update({key: value[0]}) for (key,value) in outdata.items()  \
           if len(value)==1]
    # exit program and possibly return intermediary outputs
    
    return outdata
