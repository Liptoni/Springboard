# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 15:19:11 2018

@author: Ian
"""

from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np

#transform the images so they are all on same scale
#resize smallest edge to 256 pixels, take 224x224 center crop
data_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

#import images to dataset, apply transformations
images_dataset = datasets.ImageFolder(root='Images/all/', transform=data_transform)

classes = list(images_dataset.classes)
print(classes)

#instantiate aggregation arrays of zeros for each class
animal_arr = np.zeros((224, 224, 3), np.float)
land_arr = np.zeros((224, 224, 3), np.float)
people_arr = np.zeros((224, 224, 3), np.float)
plants_arr = np.zeros((224, 224, 3), np.float)

#loop over each image in dataset. Add 1/200 values to the appropriate aggregation array
for i in range(len(images_dataset)):
    img, img_class = images_dataset[i]
    #need to transpose torch tensor, different format than numpy for storing H, W, C
    img = img.numpy().transpose(1,2,0)
    
    #add image to appropriate agg. array based on class
    if img_class == 0:
        animal_arr = animal_arr + img/200
    if img_class == 1:
        land_arr = land_arr + img/200
    if img_class == 2:
        people_arr = people_arr + img/200
    if img_class == 3:
        plants_arr = plants_arr + img/200

#instantiate figure, plot images in 2x2 matrix
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(8,8))

ax1.imshow(animal_arr)
ax1.set_title('Average Animals')
ax1.set_axis_off()

ax2.imshow(land_arr)
ax2.set_title('Average Landscapes')
ax2.set_axis_off()

ax3.imshow(people_arr)
ax3.set_title('Average People')
ax3.set_axis_off()

ax4.imshow(plants_arr)
ax4.set_title('Average Plants')
ax4.set_axis_off()

plt.tight_layout()
plt.savefig('Avg_Images/averages_figure.png')
#plt.show()

#initially used to save images individually. The code has been re-factored since
# =============================================================================
# plt.imsave('Avg_Images/avg_animal.png',animal_arr)
# plt.imsave('Avg_Images/avg_land.png',land_arr)
# plt.imsave('Avg_Images/avg_people.png',people_arr)
# plt.imsave('Avg_Images/avg_plants.png',plants_arr)
# =============================================================================
