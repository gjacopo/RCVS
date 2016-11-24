RCVS
====

Module for automatization of geo-processing workflow over raster and vector data.
---

**About**

Inspired by/based on `RIOS` module for *Raster Computer Vision Simplification*.

*credits*:      `grazzja <jacopo.grazzini@ec.europa.eu>`

*version*:      0.9

*since*:        Fri May 31 10:20:51 2013

**Description**
   
Python basic tools for geo-processing workflow of raster and vector data:
    - using the input/output utility functions of [`RIOS`][RIOS] (Raster Input Output
      Simplification) module, itself based on [`gdal`][gdal] module,
    - using external Computer Vision and Image Processing processing (CVIP) 
      algorithms provided (when installed independently) by modules like [`PIL`][PIL], 
      [`OpenCV`][OpenCV] (_i.e._ `cv2`), [`skimage`][skimage], [`matplotlib`][matplotlib] and/or [`scipimage`][scipimage].

This way, a 'simple' definition of processing workflow is possible
                
**Note**

Only the `applier` (block) processor has been override: this functionality is 
available (in the original code) through the `apply` function of the module 
`applier.py`. This function has been rewritten (and improved) in overriding
`rcvs.cvapplier` method. 

The main features of the new `rcvs.cvapplier` method are:
* apply a workflow of CVIP methods over the raster images through 
      appropriate redirection/reformatting of input/output block of data as it 
      deals with all the array format conversions necessary for 'communication' 
      between the different CVIP modules used,
* process blocks in parallel through a map/reduce (namely: `apply_async`/`reduce`) 
      like schema as it supports block and CPU multiprocessing,
* return or write multiple outputs.
    
Note that from version 1.2, parallel implementation has also been incorporated 
(using either `multiprocessing` or `mpi` module) in [`RIOS`][RIOS].

**Examples**
                
As to demonstrate how to reproduce some of the standard/simple CVIP processing
workflows, some examples provided by CVIP platforms have been adapted using 
RCVS, namely:
* RANSAC matching with [`skimage`][skimage] methods, 
* SLIC segmentation with `skimage` methods,
* pyramid decomposition with [`OpenCV`][OpenCV] methods, 
* template matching with `OpenCV` methods, 

as well as some independent implementations:
* image cross-correlation.

**Dependencies**

*require*:      `gdal`, `rios`, `numpy`, `scipy`,       
                `Queue`, `multiprocessing`,                       
                `math`, `re`, `inspect`, `operator`,       
                `itertools`, `collections`           

*optional*:     `cv2`, `skimage`, `PIL`, `matplotlib`, 
                [`vigra`][vigra], [`mahotas`][mahotas]
                pathos

[RIOS]: https://bitbucket.org/chchrsc/rios
[gdal]: https://github.com/geopy/geopy
[matplotlib]: http://matplotlib.org
[OpenCV]: http://opencv.org
[skimage]: http://scikits.appspot.com/scikits-image
[PIL]: http://www.pythonware.com/products/pil
[vigra]: http://ukoethe.github.io/vigra/doc/vigranumpy/index.html
[mahotas]: http://luispedro.org/software/mahotas
[scipimage] : http://docs.scipy.org/doc/scipy/reference/ndimage.html
