# e-radiationdetector
Electronic radiation detector repository contains my codes that were used in my thesis research. 
The title of the research is "X-rays Beam Quality Prediction in General Radiography using Artificial Neural Network (ANN) Regression Technique". 
I extracted GLCM features and some other statistical features from medical images, and used it as input features to predict the Half Value Layer (HVL) of X-rays beam. 
The final model can be used to replace a radiation detector in measuring HVL, that is why the repository name is Electronic Radiation Detector.

File Description:
Feature_Extraction_Github.py  : This code was used by me to extract some features from medical images
Image_Segmentation_Github.py  : This code will give the segmented or cropped medical images to multiply the amount of data (but in the end, I didn't use the segmented images)
Machine_Learning_Github.py    : This code was used by me to make an ANN model
HVL_actual_trains.xlsx        : This file contains the actual HVL values and the entrance surface dose from the measurement with a radiation detector. You can get the medical images dataset in the repository branch. The first three numbers in the file name shows the tube voltage (kVp) that was used to produce the medical image. You can ensure the exposure parameter and the device which was used in the DICOM metadata using ImageJ, python, MATLAB, etc.
