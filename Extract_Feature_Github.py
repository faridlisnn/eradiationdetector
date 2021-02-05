# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 00:13:37 2021

@author: Farid
"""

import pydicom
import numpy as np
from glob import glob
from skimage.feature import greycomatrix, greycoprops
from statistics import mean

data_path = r"D:\backup\data training"
data = glob(data_path + '/*.dcm')

data_image = [pydicom.dcmread(i) for i in data]
data_array = [i.pixel_array for i in data_image]
dataset = []
for i in data_array:
    px = i/(np.amax(i)/255)
    px = np.around(px)
    px = px.astype(int)
    dataset.append(px)
print("Scaling image complete")

def MeanPixel(a):
    # 'a' must be a pixel array
    return a.mean()
    
def SignaltoNoise(a,axis=0, ddof=0):
    # 'a' must be a pixel array
    a = np.asanyarray(a)
    m = a.mean(axis)
    m = m.mean()
    sd = a.std(axis=axis, ddof=ddof)
    sd = sd.mean()
    return np.where(sd == 0, 0, m/sd)
    
# Determine each list variable for dataset
contrast_feature = []
dissimilarity_feature = []
homogeneity_feature = []
energy_feature = []
correlation_feature = []
ASM_feature = []
SNR = []
Mean_Pixel = []

for i in range(len(dataset)):
    # Generate GLCM matrix
    print("Start extracting image - ", i)
    GLCM = greycomatrix(dataset[i], [1],
                        [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256,
                        symmetric = True, normed = False)
    
    SNR.append(SignaltoNoise(data_array[i]))
    Mean_Pixel.append(MeanPixel(data_array[i]))
    
    # Generate properies from GLCM   
    contrast = greycoprops(GLCM, 'contrast')
    dissimilarity = greycoprops(GLCM, 'dissimilarity')
    homogeneity = greycoprops(GLCM, 'homogeneity')
    energy = greycoprops(GLCM, 'energy')
    correlation = greycoprops(GLCM,'correlation')
    ASM = greycoprops(GLCM, 'ASM')
    
    # Average all angles value
    for i in range(len(contrast)):
        contrast[i] = mean(contrast[i])
        dissimilarity[i] = mean(dissimilarity[i])
        homogeneity[i] = mean(homogeneity[i])
        energy[i] = mean(energy[i])
        correlation[i] = mean(correlation[i])
            
    # Place first column to feature variable
    contrast_feature.append(contrast[:,0])
    dissimilarity_feature.append(dissimilarity[:,0])
    homogeneity_feature.append(homogeneity[:,0]) 
    energy_feature.append(energy[:,0]) 
    correlation_feature.append(correlation[:,0]) 
    ASM_feature.append(ASM[:,0]) 
    
# Make a dataset
import pandas as pd
data_ekspos = pd.read_excel('Dataekspos.xlsx', header=None)
data_ekspos = np.array(data_ekspos)

kVp = data_ekspos[:,0]
HVL = data_ekspos[:,1]

Dataset = pd.DataFrame({'SNR' : np.array(SNR),
                        'Pixel Mean' : Mean_Pixel,
                        'kVp' : kVp
                        })

def ExtractColumn(listname,dataframe,string):
    listname = np.array(listname)
    for i in range(len(listname.T)):
        dataframe[string] = listname[:,i]

ExtractColumn(contrast_feature,Dataset,'contrast')
ExtractColumn(dissimilarity_feature,Dataset,'dissimilarity')
ExtractColumn(homogeneity_feature,Dataset,'homogeneity')
ExtractColumn(energy_feature,Dataset,'energy')
ExtractColumn(correlation_feature,Dataset,'correlation')
ExtractColumn(ASM_feature,Dataset,'ASM')

Dataset2 = Dataset.copy()
Dataset["HVL"] = HVL

# Do Feature Scaling
Dataset_scaling = np.array(Dataset2)
for i in range(len(Dataset_scaling.T)):
    Dataset_scaling[:,i] = Dataset_scaling[:,i]/np.amax(Dataset_scaling[:,i])

Dataset_scaled = pd.DataFrame({'SNR' : Dataset_scaling[:,0],
                               'Pixel Mean' : Dataset_scaling[:,1],
                               'kVp' : Dataset_scaling[:,2],
                               'contrast' : Dataset_scaling[:,2],
                               'dissimilarity' : Dataset_scaling[:,3],
                               'homogeneity' : Dataset_scaling[:,4],
                               'energy' : Dataset_scaling[:,5],
                               'correlation' : Dataset_scaling[:,6],
                               'ASM' : Dataset_scaling[:,7],
                               'HVL' : HVL
                               })

# Export dataset
Dataset.to_csv('data_farid.csv', index = False)
Dataset_scaled.to_csv('data_farid_scaled.csv', index = False)

# Find feature correlation 
correlation_dataframe = Dataset.copy()
feature_correlation = correlation_dataframe.corr()
feature_correlation.to_csv('feature_correlation.csv', index = False)