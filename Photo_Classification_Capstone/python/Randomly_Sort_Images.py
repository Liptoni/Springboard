# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 18:47:12 2018

@author: Ian
"""

from glob import glob
import numpy as np
import os

#folders that hold the images
folders = ['animals', 'landscape', 'people', 'plants']

#loop through folders and move 20% to validation folder structure
for folder in folders:
    folder_to_search = 'Images\\train\\'+folder + "\\"
    images = glob(folder_to_search + "*.jpg")
    
    #gen the number of images to move
    n_to_sample = int(float(len(images)) * 0.2)
    
    #randomly select the images to move
    sampled = np.random.choice(images, n_to_sample, replace=False)    
    
    #get the filepath of the new image destination then move the files
    new_dest = [img.replace('train', 'val') for img in sampled]    
    for old, new in zip(sampled, new_dest):
        os.rename(old, new)