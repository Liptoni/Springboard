# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 11:07:58 2018

@author: Ian
"""

from glob import glob
import numpy as np
import shutil

np.random.seed(18)

images = glob('Images/all/*/*.jpg')

sampled = np.random.choice(images, 10, replace=False)

new_folder = [img.replace('all', 'testing') for img in sampled]

new_dest = []

for n in new_folder:
    idx = n.find('\\', n.find('\\')+1)+1
    new_dest.append(n[:idx])

for old, new in zip(sampled, new_dest):
    shutil.copy(old, new)
