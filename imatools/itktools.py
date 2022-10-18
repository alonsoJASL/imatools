#!/usr/bin/env python3

import os, sys, subprocess, pdb, re, struct,errno
# import vtk
# import vtk.util.numpy_support as vtknp
from imatools.ioutils import cout
from imatools.ioutils import fullfile
import SimpleITK as sitk
import numpy as np

def load_image_as_np(path_to_file) :
    """ Reads image into numpy array """
    sitk_t1 = sitk.ReadImage(path_to_file)
    
    t1 = sitk.GetArrayFromImage(sitk_t1)
    origin = sitk_t1.GetOrigin()
    im_size = sitk_t1.GetSize()

    return t1, origin, im_size 


