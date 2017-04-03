import dicom
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from sklearn.utils import shuffle

path = './sample_images/'
dcmFiles = []
for dirName, subDirList, fileList in os.walk(path):
    for fileName in fileList:
        if '.dcm' in fileName.lower():
            dcmFiles.append(os.path.join(dirName,fileName))

id_labels = dict()
with open('stage1_labels.csv', 'r') as labelFile:
    for i,line in enumerate(labelFile):
        if i == 0: continue
        p_id, cancer = line.strip().split(',')
        id_labels[p_id] = int(cancer)

def convert_images(id_labels, dFiles):
    labeled_images = []
    labels = []
    cancer_additions = 0
    for file in dFiles:
        ref = dicom.read_file(file)
        p_id = ref.PatientID
        if p_id in id_labels and id_labels[p_id] == 1:
            labeled_images.append(ref.pixel_array)
            labels.append(id_labels[p_id])
            cancer_additions+=1
            if cancer_additions > 3000: break
    return np.array(labeled_images, dtype=np.float32), np.array(labels)
