# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 16:06:45 2018

@author: Ian
"""
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from datetime import datetime

startTime = datetime.now()

#import image paths as a pandas df
images_df = pd.read_csv('Image_Paths.csv')

#split into individual series containing file path for each training/validation category
train_land = images_df[(images_df.Category == 'landscape') & (images_df.Train_Val == 'train')]['File_Path']
train_animal = images_df[(images_df.Category == 'animals') & (images_df.Train_Val == 'train')]['File_Path']
train_people = images_df[(images_df.Category == 'people') & (images_df.Train_Val == 'train')]['File_Path']
train_plants = images_df[(images_df.Category == 'plants') & (images_df.Train_Val == 'train')]['File_Path']

val_land = images_df[(images_df.Category == 'landscape') & (images_df.Train_Val == 'val')]['File_Path']
val_animal = images_df[(images_df.Category == 'animals') & (images_df.Train_Val == 'val')]['File_Path']
val_people = images_df[(images_df.Category == 'people') & (images_df.Train_Val == 'val')]['File_Path']
val_plants = images_df[(images_df.Category == 'plants') & (images_df.Train_Val == 'val')]['File_Path']

#combine training and validation to get series with all imagesfor each category
all_land = train_land.append(val_land)
all_animal = train_animal.append(val_animal)
all_people = train_people.append(val_people)
all_plants = train_plants.append(val_plants)


#list to iterate over, has series, plot title, file name for export
images_list = ((train_land, 'Training Landscapes', 'Color_Plots/train_land_pct.png'), (train_animal, 'Training Animals', 'Color_Plots/train_animal_pct.png'), 
               (train_people, 'Training People', 'Color_Plots/train_people_pct.png'), (train_plants, 'Training Plants', 'Color_Plots/train_plants_pct.png'), 
               (val_land, 'Validation Landscapes', 'Color_Plots/val_land_pct.png'), (val_animal, 'Validation Animals', 'Color_Plots/val_animal_pct.png'), 
               (val_people, 'Validation People', 'Color_Plots/val_people_pct.png'), (val_plants, 'Validation Plants', 'Color_Plots/val_plants_pct.png'), 
               (all_land, 'All Landscapes', 'Color_Plots/all_land_pct.png'), (all_animal, 'All Animals', 'Color_Plots/all_animal_pct.png'), 
               (all_people, 'All People', 'Color_Plots/all_people_pct.png'), (all_plants, 'All Plants', 'Color_Plots/all_plants_pct.png'))

#loop over each series to create plots
for s, plot_title, filename in images_list:
    print(plot_title)
    #instantiate empty dicts with int as default data type
    reds_dict = defaultdict(int)
    greens_dict = defaultdict(int)
    blues_dict = defaultdict(int)
    
    #instantiate figure for plots
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12,4))    
    
    #loop over each image in series
    for idx, image_path in s.iteritems():
        
        #read in image, split out color channels
        image = io.imread(image_path)        
        reds = image[:,:, 0]
        greens = image[:,:, 1]
        blues = image[:,:, 2]
        
        #loop throgh each color channel
        #add pixel value to corresponding key in color dicts
        for i in reds.flat:
            reds_dict[i] += 1
        
        for i in greens.flat:
            greens_dict[i] += 1
        
        for i in blues.flat:
            blues_dict[i] += 1
            
    #use this block if outputting values on percent basis
# =============================================================================
#     #get the total number of pixels counted for each channel
#     red_total = sum(reds_dict.values(), 0.0)
#     green_total = sum(greens_dict.values(), 0.0)
#     blue_total = sum(blues_dict.values(), 0.0)
#     
#     #divide dict values by total pixels to get percent
#     reds_dict = {k: v/red_total for k, v in reds_dict.items()}
#     greens_dict = {k: v/green_total for k, v in greens_dict.items()}
#     blues_dict = {k: v/blue_total for k, v in blues_dict.items()}
#     
# =============================================================================
    
    #use this block if we just want to see raw pixel values (divided by 1000)
    #divide dict values by 1000 so the scale looks better
    reds_dict = {k: v/1000 for k, v in reds_dict.items()}
    greens_dict = {k: v/1000 for k, v in greens_dict.items()}
    blues_dict = {k: v/1000 for k, v in blues_dict.items()}

    
    #plot values on figure
    ax1.plot(*zip(*sorted(reds_dict.items())), color='r', linestyle='-')
    ax1.set_title('Reds')
    ax1.set_ylabel("N / 1000")
    ax1.set_ylim(top=0.04)
    
    
    ax2.plot(*zip(*sorted(greens_dict.items())), color='g', linestyle='-')
    ax2.set_title('Greens')
    ax2.set_ylim(top=0.04)
    
    ax3.plot(*zip(*sorted(blues_dict.items())), color='b', linestyle='-')
    ax3.set_title('Blues')
    ax3.set_ylim(top=0.04)
    
    fig.suptitle(plot_title)    
    plt.tight_layout()    
    fig.subplots_adjust(top=0.88)
    plt.savefig(filename)    
    plt.close()
    
    
print('elapsed:', datetime.now()-startTime)
print('Done!')