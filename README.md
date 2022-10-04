# Gerador-De-Imagens-Inator

## Introduction

This is a Doofenshmirtz inspired Discord bot that takes advantage of the open source python package [Diffusers](https://huggingface.co/blog/stable_diffusion) to make text to image inferences inside a Discord channel.

The users have the ability to generate a grid of sample images as response to their prompt, and the bot will display a grid of buttons, each one respective to their sampel image, with the purpose of starting the upscaling process. The upscaling process will resume the denoising process for the sample image, as well as upscale the final result, using one of the upsampling models available.

Here are some examples of images generated with the text to image pipeline, and upscaled with RealESRGAN.

### /inator a city on an island floating in the sky with clouds in the background, digital art, detailed, 4k --seed 0
![upscaled_realesrgan](https://user-images.githubusercontent.com/75852333/193890780-5c0e6340-e3f7-4fd0-abd0-5e8b2e693393.png)

### /inator dark and terrifying house living room interior overview design, haunting creepy demon is standing in the corner of the room, 8K, ultra wide angle, higly detailed --wallpaper
![dark_and_terrifying_horror_house_living_room_interior_overview_design_demon_with_red_eyes_is_standing_in_the_corner_Moebius_Greg_Rutkowski_Zabrocki_Karlkka_Jayison_Devadas_Phuoc_Quan_trending_on_Artstation_8K_ultra_w](https://user-images.githubusercontent.com/75852333/193911347-d47713c7-9775-40d9-a041-77814f5dac5d.png)

### /inator temple in ruines, forest, stairs, columns, cinematic, detailed, atmospheric, epic, concept art, Matte painting, background, mist, photo-realistic, concept art, volumetric light, cinematic epic, rule of thirds, octane render, 8k, corona render, movie concept art, octane render, cinematic, trending on artstation, movie concept art, cinematic composition, ultra-detailed, realistic, hyper-realistic, volumetric lighting, 8k
![temple_in_ruines_forest_stairs_columns_cinematic_detailed_atmospheric_epic_concept_art_Matte_painting_background_mist_photo-realistic_concept_art_volumetric_light_cinematic_epic_rule_of_thirds_octane_render_8k_corona](https://user-images.githubusercontent.com/75852333/193912762-1d768470-8279-4081-bca6-9114ae3fd0e9.png)

## Setup

First of all, for better performing inferences, a high-memory (>= 10 GB) NVIDIA GPU is advised. Make sure you have installed the most recent drivers and CUDA toolkit. Then, this version of the bot requires > 3.8 Python. Now let's install some dependencies!

```
pip install py-cord
pip install python-dotenv
pip install diffusers==0.3.0 transformers scipy ftfy
pip install realesrgan
```

In case you have no GPU but intend to run inferences on CPU only anyways, install:

```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116https://download.pytorch.org/whl/cpu
```

Otherwise, install:

```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

Then, the final dependencies:

```
pip install Pillow
pip install opencv-contrib-python
```

It's important that you install the packages in this order, because realesgran has a tendency to replace both Torch and OpenCV currently installed versions.

Lastly, create a file in the root directory named ".env". Inside, write the following:

```
TOKEN=YOUR_DISCORD_BOT_TOKEN
PIPELINE_TOKEN=YOUR_PIPELINE_TOKEN
```

YOUR_DISCORD_BOT_TOKEN being the token given to you in the Discord Developer Portal, and YOUR_PIPELINE_TOKEN being your Hugging Faces token which you can get [here](https://huggingface.co/docs/hub/security-tokens).
