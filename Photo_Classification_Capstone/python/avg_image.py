# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 15:19:11 2018

@author: Ian
"""

from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np

data_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

images_dataset = datasets.ImageFolder(root='Images/all/', transform=data_transform)

classes = list(images_dataset.classes)
print(classes)

animal_arr = np.zeros((224, 224, 3), np.float)
land_arr = np.zeros((224, 224, 3), np.float)
people_arr = np.zeros((224, 224, 3), np.float)
plants_arr = np.zeros((224, 224, 3), np.float)


for i in range(len(images_dataset)):
    img, img_class = images_dataset[i]
    img = img.numpy().transpose(1,2,0)
    
    if img_class == 0:
        animal_arr = animal_arr + img/200
    if img_class == 1:
        land_arr = land_arr + img/200
    if img_class == 2:
        people_arr = people_arr + img/200
    if img_class == 3:
        plants_arr = plants_arr + img/200


fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=2, figsize=(8,8))

ax1[0].imshow(animal_arr)
ax1[0].set_title('Average Animals')
ax1[0].set_axis_off()

ax1[1].imshow(land_arr)
ax1[1].set_title('Average Landscapes')
ax1[1].set_axis_off()

ax2[0].imshow(people_arr)
ax2[0].set_title('Average People')
ax2[0].set_axis_off()

ax2[1].imshow(plants_arr)
ax2[1].set_title('Average Plants')
ax2[1].set_axis_off()

plt.tight_layout()
plt.savefig('Avg_Images/averages_figure.png')
#plt.show()


# =============================================================================
# plt.imsave('Avg_Images/avg_animal.png',animal_arr)
# plt.imsave('Avg_Images/avg_land.png',land_arr)
# plt.imsave('Avg_Images/avg_people.png',people_arr)
# plt.imsave('Avg_Images/avg_plants.png',plants_arr)
# =============================================================================
