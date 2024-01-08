# Diffusion Models for Medical Image Data Augmentation

## Introduction

This repository focuses on exploring diffusion models for medical image data augmentation. Data augmentation is crucial in training robust machine learning models, especially in the medical imaging domain. Four different approaches are employed to train the model with varying degrees of augmentation:

1. **No Augmentation:** The model is trained on the original, unaltered dataset.
2. **ImageDataGenerator Augmentation:** The keras ImageDataGenerator is utilized for augmentation, applying operations such as rotation, shear, zoom, horizontal flip, and vertical flip.
3. **DCGAN Augmentation:** The model is trained using a DCGAN (Deep Convolutional Generative Adversarial Network) to generate synthetic images for augmentation.
4. **DDPM Augmentation:** Diffusion models, specifically DDPM (Denoising Diffusion Probabilistic Models), are employed for generating augmented images.

## Code Structure

- **classifier.py:** Contains the code for the classifier model.
- **gan.py:** Contains the code for the DCGAN model and generating corresponding images.
- **data_augment.py:** Contains code for generating augmented images using the keras ImageDataGenerator.
- **DDPM.py:** Contains the code for the DDPM model.
- **gen_DDPM.py:** Contains the code for generating augmented images using DDPM.

## Running the Code

To generate augmented images for the 'yes' and 'no' cases separately:

1. Run the code in `data_augment.py` for ImageDataGenerator augmentation.
2. Run the code in `gan.py` for DCGAN augmentation.
3. Run the code in `DDPM.py` for DDPM augmentation.
4. Execute `gen_DDPM.py` for generating augmented images using DDPM.

Make sure to follow the instructions in each file for proper execution.

## Contribution

Contributions are welcome! Feel free to open issues or create pull requests to enhance the functionality or address any improvements.

