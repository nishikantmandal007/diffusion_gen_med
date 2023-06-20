from PIL import Image
import torch
import torch.nn.functional as F
import torchvision
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import cv2
from diffusers import DDPMPipeline
from diffusers import DDPMScheduler
from diffusers import UNet2DModel
filename = 'iter40dim64/unet/diffusion_pytorch_model.bin'
shape = 64


def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet2DModel(
    sample_size=shape,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    # the number of output channels for each UNet block
    # block_out_channels=(128, 128, 256, 256, 512, 512),
    block_out_channels=(128, 128, 256, 256,  512, 512),
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

model.to(device)
model.load_state_dict(torch.load(filename))
sample = torch.randn(1, 3, shape, shape).to(device)
noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
)

for i, t in enumerate(noise_scheduler.timesteps):

    # Get model pred
    with torch.no_grad():
        residual = model(sample, t).sample

    sample = noise_scheduler.step(residual, t, sample).prev_sample
print(sample.shape)
output = show_images(sample).resize((8 * 64, 4*256), resample=Image.NEAREST)
output.save('gen_img.png')
