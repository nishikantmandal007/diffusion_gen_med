from accelerate import Accelerator
from pathlib import Path
from tqdm.auto import tqdm
from huggingface_hub import HfFolder, Repository, whoami
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import UNet2DModel
from diffusers import DDPMScheduler

from diffusers import DDPMPipeline
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import torchvision
import torch.nn.functional as F
import torch
from PIL import Image

# Load the images and labels from X.npy and y.npy
XX = np.load('X.npy')
yy = np.load('y.npy')


def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im


def get_yes():
    x = []
    y = []
    for i in range(len(XX)):
        if yy[i] == 1:
            x.append(XX[i])
            y.append(yy[i])
    x = np.array(x)
    y = np.array(y)
    return x, y


X, y = get_yes()

# Create a data loader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
timesteps = torch.linspace(0, 999, 8).long().to(device)
# X to tensor
X = torch.from_numpy(X).float().to(device)
# X should be (batch_size, channels, width, height)
X = X.permute(0, 3, 1, 2)
# print("X shape", X.shape)
# Pick 8 images from the dataset
X_ = X[:8]
output = show_images(X_).resize((8 * 64, 128), resample=Image.NEAREST)
output.save('output.png')
noise = torch.randn_like(X)
timesteps = torch.linspace(0, 999, X.shape[0]).long().to(device)
noisy_xb = noise_scheduler.add_noise(X, noise, timesteps)
print("Noisy X shape", noisy_xb.shape)
# pick 8 images from the noisy dataset
noisy_xb_ = noisy_xb[10:12]
output = show_images(noisy_xb_).resize((8 * 64, 64), resample=Image.NEAREST)
output.save('output.png')

# Create a model

print("X shape", X.shape[2])
class TrainingConfig:
    image_size = 64
    train_batch_size = 16
    eval_batch_size = 16
    num_epochs = 10
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 1
    mixed_precision = "fp16"
    output_dir = "output"
    seed = 0


config = TrainingConfig()
train_dataloader = torch.utils.data.DataLoader(
    X, batch_size=config.train_batch_size, shuffle=True)

model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    # the number of output channels for each UNet block
    # block_out_channels=(128, 128, 256, 256, 512, 512),
    block_out_channels=(128, 128,  512, 512),
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        # "DownBlock2D",
        # "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        # "UpBlock2D",
        # "UpBlock2D",
        "UpBlock2D",
    ),
)
model.to(device)
noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000
)
sample = X[0].unsqueeze(0)
print("Output shape", model(sample, timestep=0).sample.shape)


optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        logging_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader),
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(
                clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps,
                                   return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[
                0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(
                model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:

                pipeline.save_pretrained(config.output_dir)


from accelerate import notebook_launcher

args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)

# import glob

# sample_images = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))
# Image.open(sample_images[-1])

# optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
# losses = []

# for epoch in range(10):
#     for step, batch in enumerate(train_dataloader):
#         # print(batch)
#         # print("Batch shape", batch.shape)
#         clean_images = batch.to(device)
#         # Sample noise to add to the images
#         noise = torch.randn(clean_images.shape).to(clean_images.device)
#         bs = clean_images.shape[0]

#         # Sample a random timestep for each image
#         timesteps = torch.randint(
#             0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device
#         ).long()

#         # Add noise to the clean images according to the noise magnitude at each timestep
#         noisy_images = noise_scheduler.add_noise(
#             clean_images, noise, timesteps)

#         # Get the model prediction
#         # Show model progress

#         noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

#         # Calculate the loss
#         loss = F.mse_loss(noise_pred, noise)
#         loss.backward(loss)
#         losses.append(loss.item())

#         # Update the model parameters with the optimizer
#         optimizer.step()
#         optimizer.zero_grad()
#     loss_last_epoch = sum(
#         losses[-len(train_dataloader):]) / len(train_dataloader)
#     print(f"Epoch:{epoch+1}, loss: {loss_last_epoch}")

# # save the model
# filename = 'iters' + str(epoch) + 'dim' + str(X.shape[2]) + '.pt'
# torch.save(model.state_dict(), filename)

# # sample = torch.randn(2, 3, X.shape[2], X.shape[2]).to(device)

# # for i, t in enumerate(noise_scheduler.timesteps):

# #     # Get model pred
# #     with torch.no_grad():
# #         residual = model(sample, t).sample

# #     # Update sample with step
# #     sample = noise_scheduler.step(residual, t, sample).prev_sample

# # output = show_images(sample)
# # output.save('output1.png')
