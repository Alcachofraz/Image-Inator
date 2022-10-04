# Gerador-De-Imagens-Inator

## Introduction

This is a Doofenshmirtz inspired Discord bot that takes advantage of the open source python package [Diffusers](https://huggingface.co/blog/stable_diffusion) to make text to image inferences inside a Discord channel.

The users have the ability to generate a grid of sample images as response to their prompt, and the bot will display a grid of buttons, each one respective to their sampel image, with the purpose of starting the upscaling process. The upscaling process will resume the denoising process for the sample image, as well as upscale the final result, using one of the upsampling models available.

Here's an example of an image generate with the command "/inator a city on an island floating in the sky with clouds in the background, digital art, detailed, 4k --seed 0", and upscaled with RealESRGAN.

![upscaled_realesrgan](https://user-images.githubusercontent.com/75852333/193890780-5c0e6340-e3f7-4fd0-abd0-5e8b2e693393.png)

## Setup

First of all, for better performing inferences, a high-memory (>= 10 GB) NVIDIA GPU is advised. Ensure you have isntalled the most recent drivers and CUDA toolkit. Then, this version of the bot requires > 3.8 Python. Now let's install some dependencies!

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
PIPELINE_TOKEN=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
```

YOUR_DISCORD_BOT_TOKEN being the token given to you in the Discord Developer Portal, and PIPELINE_TOKEN being your Hugging Faces token which you can get [here](https://huggingface.co/docs/hub/security-tokens).
