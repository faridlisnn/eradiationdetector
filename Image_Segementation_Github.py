# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 00:16:11 2021

@author: Farid
"""

import pydicom
import numpy as np
from glob import glob

data_path = r"C:\Users\Data Training"
data = glob(data_path + '/*.dcm')

data_image = [pydicom.dcmread(i) for i in data]
data_array = [i.pixel_array for i in data_image]

# Crop 25% of the data_array image at the center
data_array_centre = []
for i in data_array:
    i = np.array(i)
    y_37 = int(np.around(0.375*len(i)))
    x_37 = int(np.around(0.375*len(i.T)))
    y_62 = int(np.around(0.625*len(i)))
    x_62 = int(np.around(0.625*len(i.T)))    
    data_array_centre.append(i[y_37:y_62,x_37:x_62])

# Divide data_array_centre image into 4 parts
data_array_seg = []
for i in data_array_centre:
    i = np.array(i)
    y_50 = int(np.around(0.5*len(i)))
    x_50 = int(np.around(0.5*len(i.T)))
    y_100 = int(np.around(len(i)))
    x_100 = int(np.around(len(i.T)))
    data_array_seg.append(i[0:y_50,0:x_50])
    data_array_seg.append(i[0:y_50,x_50:x_100])
    data_array_seg.append(i[y_50:y_100,0:x_50])
    data_array_seg.append(i[y_50:y_100,x_50:x_100])

# Divide data_array image into 4 parts
data_array_seg_4 = []
for i in data_array:
    i = np.array(i)
    y_50 = int(0.5*len(i))
    x_50 = int(0.5*len(i.T))
    y_100 = int(len(i))
    x_100 = int(len(i.T))
    data_array_seg_4.append(i[0:y_50,0:x_50])
    data_array_seg_4.append(i[0:y_50,x_50:x_100])
    data_array_seg_4.append(i[y_50:y_100,0:x_50])
    data_array_seg_4.append(i[y_50:y_100,x_50:x_100])