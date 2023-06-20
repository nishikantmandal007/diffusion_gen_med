### Diffusion models for Medical Image Data Augmentation

## Introduction

There are 4 ways we are going to train the model.

1. No augmentation at all.
2. Augmentation using the keras ImageDataGenerator, applying rotation, shear, zoom, horizontal flip and vertical flip.
3. Augmentation using DCGAN.
4. Augmentation using DDPM.

## Code

1. classifier.py : This file contains the code for the classifier model.
2. gan.py : This file contains the code for the DCGAN model and generating the corresponding images.
3. data_augment.py: This file contains code for generating the augmented images keras ImageDataGenerator.
4. DDPM.py: This file contains the code for the DDPM model
5. gen_DDPM.py: This file contains the code for generating the augmented images using DDPM.

Run the code for generation of augmented images for yes/no case separately in case of GAN and DDPM.
