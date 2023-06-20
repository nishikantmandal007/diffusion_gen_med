import os
import numpy as np
from nibabel.testing import data_path
# example_filename = os.path.join(data_path, 'my_brain.nii.gz')
import nibabel as nib

img = nib.load('my_brain (1).nii.gz')
print(img.shape)
# Show the image    
from nilearn import plotting
# save the image to a file
plotting.plot_anat(img, title="My brain")
plotting.show()

for i in range(1,11):
    img_path = 'my_brain (' + str(i) + ').nii.gz'
    img = nib.load(img_path)
    output_path = 'my_brain_' + str(i) + '.png'
   
    # plotting.plot_anat(img, title="My brain", display_mode="z", cut_coords=1, draw_cross=False, annotate=False, output_file=output_path)
    plotting.plot_anat(img, title="My brain",  output_file=output_path)



