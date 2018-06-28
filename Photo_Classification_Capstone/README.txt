This repository contains the files associated with my photo classification capstone project.

For this project, I used deep learning techniques in order to classify photos into one of four categories. I build four models using PyTorch (logistic regression, feed-forward NN, simple CNN, ResNet101).
My best results were obtained using transfer learning and re-training a pre-trained ResNet 101 model. I obtained over 98% accuracy with only 640 trianing images and 160 validation images.

Folders:
  Avg_Images - this folder contains the outputs of the average images script (avg_image.py)
  
  Color_Plots - this folder contains the outputs of the scipt used to count the frequency of color values accross images (Image_EDA.py)
  
  Jupyter_Notebooks - this folder contains iPython notebooks used to illustrate a few of my scripts
  
  predictions - this folder contains image outputs of the predictions from the different models I trained
  
  python - this folder contains the different python scripts I used throughout the project
  
  Summaries - this folder contains the capstone summary reports including my proposal, milestone report, and final report.