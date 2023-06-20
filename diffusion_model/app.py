import random
import shutil
import uuid
from pathlib import Path

import cv2
import gradio as gr
import mediapy
import mlflow.pytorch
import numpy as np
import torch
from skimage import img_as_ubyte

from models.ddim import DDIMSampler

import nibabel as nib

ffmpeg_path = shutil.which("ffmpeg")
mediapy.set_ffmpeg(ffmpeg_path)

# Loading model
device = torch.device("cpu")
vqvae = mlflow.pytorch.load_model(
    "./trained_models/vae/",
    map_location=device,
)
vqvae.eval()

diffusion = mlflow.pytorch.load_model(
    "./trained_models/ddpm/",
    map_location=device,
)
diffusion.eval()

diffusion = diffusion.to(device)
vqvae = vqvae.to(device)


def sample_fn(
    gender_radio,
    age_slider,
    ventricular_slider,
    brain_slider,
):
    print("Sampling brain!")
    print(f"Gender: {gender_radio}")
    print(f"Age: {age_slider}")
    print(f"Ventricular volume: {ventricular_slider}")
    print(f"Brain volume: {brain_slider}")

    age_slider = (age_slider - 44) / (82 - 44)

    cond = torch.Tensor([[gender_radio, age_slider, ventricular_slider, brain_slider]])
    latent_shape = [1, 3, 20, 28, 20]
    cond_crossatten = cond.unsqueeze(1)
    cond_concat = cond.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    cond_concat = cond_concat.expand(list(cond.shape[0:2]) + list(latent_shape[2:]))
    conditioning = {
        "c_concat": [cond_concat.float().to(device)],
        "c_crossattn": [cond_crossatten.float().to(device)],
    }

    ddim = DDIMSampler(diffusion)
    num_timesteps = 50
    latent_vectors, _ = ddim.sample(
        num_timesteps,
        conditioning=conditioning,
        batch_size=1,
        shape=list(latent_shape[1:]),
        eta=1.0,
    )

    with torch.no_grad():
        x_hat = vqvae.reconstruct_ldm_outputs(latent_vectors).cpu()

    return x_hat.numpy()


def sample_with_text_fn(text_prompt):
    # Not implemented
    pass


def create_videos_and_file(
    gender_radio,
    age_slider,
    ventricular_slider,
    brain_slider,
):
    output_dir = Path(
        f"./outputs/{str(uuid.uuid4())}"
    )
    output_dir.mkdir(exist_ok=True)

    image_data = sample_fn(
        gender_radio,
        age_slider,
        ventricular_slider,
        brain_slider,
    )
    image_data = image_data[0, 0, 5:-5, 5:-5, :-15]
    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())
    image_data = (image_data * 255).astype(np.uint8)

    # Write frames to video
    with mediapy.VideoWriter(
        f"{str(output_dir)}/brain_axial.mp4", shape=(150, 214), fps=12, crf=18
    ) as w:
        for idx in range(image_data.shape[2]):
            img = image_data[:, :, idx]
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            frame = img_as_ubyte(img)
            w.add_image(frame)

    with mediapy.VideoWriter(
        f"{str(output_dir)}/brain_sagittal.mp4", shape=(145, 214), fps=12, crf=18
    ) as w:
        for idx in range(image_data.shape[0]):
            img = np.rot90(image_data[idx, :, :])
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            frame = img_as_ubyte(img)
            w.add_image(frame)

    with mediapy.VideoWriter(
        f"{str(output_dir)}/brain_coronal.mp4", shape=(145, 150), fps=12, crf=18
    ) as w:
        for idx in range(image_data.shape[1]):
            img = np.rot90(np.flip(image_data, axis=1)[:, idx, :])
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            frame = img_as_ubyte(img)
            w.add_image(frame)

    # Create file
    affine = np.array(
        [
            [-1.0, 0.0, 0.0, 96.48149872],
            [0.0, 1.0, 0.0, -141.47715759],
            [0.0, 0.0, 1.0, -156.55375671],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    empty_header = nib.Nifti1Header()
    sample_nii = nib.Nifti1Image(image_data, affine, empty_header)
    nib.save(sample_nii, f"{str(output_dir)}/my_brain.nii.gz")

    # time.sleep(2)

    return (
        f"{str(output_dir)}/brain_axial.mp4",
        f"{str(output_dir)}/brain_sagittal.mp4",
        f"{str(output_dir)}/brain_coronal.mp4",
        f"{str(output_dir)}/my_brain.nii.gz",
    )


def randomise():
    random_age = round(random.uniform(44.0, 82.0), 2)
    return (
        random.choice(["Female", "Male"]),
        random_age,
        round(random.uniform(0, 1.0), 2),
        round(random.uniform(0, 1.0), 2),
    )


def unrest_randomise():
    random_age = round(random.uniform(18.0, 100.0), 2)
    return (
        random.choice([1, 0]),
        random_age,
        round(random.uniform(-1.0, 2.0), 2),
        round(random.uniform(-1.0, 2.0), 2),
    )


# TEXT
title = "Generating Brain Imaging with Diffusion Models"
description = """
<center><a href="https://arxiv.org/abs/2209.07162">[PAPER]</a> <a href="https://academictorrents.com/details/63aeb864bbe2115ded0aa0d7d36334c026f0660b">[DATASET]</a></center>

<details>
<summary><b>Instructions</b></summary>

<p style="margin-top: -3px;">With this app, you can generate synthetic brain images with one click!<br />You have several ways to set how your generated brain will look like:<br /></p>
 <ul style="margin-top: -20px;margin-bottom: -15px;">
  <li style="margin-bottom: -10px;margin-left: 20px;">Use the "<i>Inputs</i>" tab to create well-behaved brains using the same value ranges that our <br />models learned as described in paper linked above</li>
  <li style="margin-left: 20px;">Use the "<i>Unrestricted Inputs</i>" tab to generate the wildest brains!</li>
  <li style="margin-left: 20px;">Use the "<i>Text prompt</i>" tab to generate brains based on text descriptions (Coming soon).</li>
</ul> 
<p>After customisation, just hit "<i>Generate</i>" and wait a few seconds.<br />The generated brain will also be available for download, and you can use your favourite Nifti Viewer to check it.<br />Note: if are having problems with the videos, try our app using chrome. <b>Enjoy!<b><p>
</details>

"""

article = """
Checkout our dataset with [100K synthetic brain](https://academictorrents.com/details/63aeb864bbe2115ded0aa0d7d36334c026f0660b)! 🧠🧠🧠

App made by [Walter Hugo Lopez Pinaya](https://twitter.com/warvito) from [AMIGO](https://amigos.ai/)
<center><img src="https://raw.githubusercontent.com/Warvito/public_images/master/assets/Footer_1.png" alt="Project by amigos.ai" style="width:450px;"></center>
<center><img src="https://raw.githubusercontent.com/Warvito/public_images/master/assets/Footer_2.png" alt="Acknowledgements" style="width:750px;"></center>
"""

demo = gr.Blocks()

with demo:
    gr.Markdown(
        "<h1 style='text-align: center; margin-bottom: 1rem'>" + title + "</h1>"
    )
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            with gr.Box():
                with gr.Tabs():
                    with gr.TabItem("Inputs"):
                        with gr.Row():
                            gender_radio = gr.Radio(
                                choices=["Female", "Male"],
                                value="Female",
                                type="index",
                                label="Gender",
                                interactive=True,
                            )
                            age_slider = gr.Slider(
                                minimum=44,
                                maximum=82,
                                value=63,
                                label="Age [years]",
                                interactive=True,
                            )
                        with gr.Row():
                            ventricular_slider = gr.Slider(
                                minimum=0,
                                maximum=1,
                                value=0.5,
                                label="Volume of ventricular cerebrospinal fluid",
                                interactive=True,
                            )
                            brain_slider = gr.Slider(
                                minimum=0,
                                maximum=1,
                                value=0.5,
                                label="Volume of brain",
                                interactive=True,
                            )
                        with gr.Row():
                            submit_btn = gr.Button("Generate", variant="primary")
                            randomize_btn = gr.Button("I'm Feeling Lucky")

                    with gr.TabItem("Unrestricted Inputs"):
                        with gr.Row():
                            unrest_gender_number = gr.Number(
                                value=1.0,
                                precision=1,
                                label="Gender [Female=0, Male=1]",
                                interactive=True,
                            )
                            unrest_age_number = gr.Number(
                                value=63,
                                precision=1,
                                label="Age [years]",
                                interactive=True,
                            )
                        with gr.Row():
                            unrest_ventricular_number = gr.Number(
                                value=0.5,
                                precision=2,
                                label="Volume of ventricular cerebrospinal fluid",
                                interactive=True,
                            )
                            unrest_brain_number = gr.Number(
                                value=0.5,
                                precision=2,
                                label="Volume of brain",
                                interactive=True,
                            )
                        with gr.Row():
                            unrest_submit_btn = gr.Button("Generate", variant="primary")
                            unrest_randomize_btn = gr.Button("I'm Feeling Lucky")

                        gr.Examples(
                            examples=[
                                [1, 63, 1.3, 0.5],
                                [0, 63, 1.9, 0.5],
                                [1, 63, -0.5, 0.5],
                                [0, 63, 0.5, -0.3],
                            ],
                            inputs=[
                                unrest_gender_number,
                                unrest_age_number,
                                unrest_ventricular_number,
                                unrest_brain_number,
                            ],
                        )
                    with gr.TabItem("Text prompt"):
                        text_prompt = gr.Textbox("Coming soon... Follow me on twitter to get latest updates.", show_label=False, interactive=False)
                        submit_text_btn = gr.Button("Generate", variant="primary", )
                        gr.Examples(
                            examples=[
                                ["32 years old | Normal appearance brain"],
                                ["T2 weighted | Male | 50 years old | There are a few T2 hyperintensities in the deep white matter of the frontal lobes"],
                                ["Minor small vessel change"],
                                ["T1 weighted | There is a mild to moderate arachnoid cyst within the anterior left middle cranial fossa"],
                            ],
                            inputs=[
                                text_prompt
                            ],
                        )


        with gr.Column():
            with gr.Box():
                with gr.Tabs():
                    with gr.TabItem("Axial View"):
                        axial_sample_plot = gr.Video(show_label=False)
                    with gr.TabItem("Sagittal View"):
                        sagittal_sample_plot = gr.Video(show_label=False)
                    with gr.TabItem("Coronal View"):
                        coronal_sample_plot = gr.Video(show_label=False)
                sample_file = gr.File(label="My Brain")

    gr.Markdown(article)

    submit_btn.click(
        create_videos_and_file,
        [
            gender_radio,
            age_slider,
            ventricular_slider,
            brain_slider,
        ],
        [axial_sample_plot, sagittal_sample_plot, coronal_sample_plot, sample_file],
        # [axial_sample_plot, sagittal_sample_plot, coronal_sample_plot],
    )
    unrest_submit_btn.click(
        create_videos_and_file,
        [
            unrest_gender_number,
            unrest_age_number,
            unrest_ventricular_number,
            unrest_brain_number,
        ],
        [axial_sample_plot, sagittal_sample_plot, coronal_sample_plot, sample_file],
        # [axial_sample_plot, sagittal_sample_plot, coronal_sample_plot],
    )

    randomize_btn.click(
        fn=randomise,
        inputs=[],
        queue=False,
        outputs=[gender_radio, age_slider, ventricular_slider, brain_slider],
    )

    unrest_randomize_btn.click(
        fn=unrest_randomise,
        inputs=[],
        queue=False,
        outputs=[
            unrest_gender_number,
            unrest_age_number,
            unrest_ventricular_number,
            unrest_brain_number,
        ],
    )

    # submit_text_btn.click(
    #     fn=sample_with_text_fn,
    #     inputs=[text_prompt],
    #     outputs=[axial_sample_plot, sagittal_sample_plot, coronal_sample_plot],
    # )

# demo.launch(share=True, enable_queue=True)
# demo.launch(enable_queue=True)
demo.queue()
demo.launch()

