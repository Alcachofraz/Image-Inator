# Image-Inator

## Introduction

This is a Doofenshmirtz inspired Discord bot that takes advantage of the open source python package [Diffusers](https://huggingface.co/blog/stable_diffusion) to make text to image inferences inside a Discord channel.

The users have the ability to generate a grid of sample images as response to their prompt, and the bot will display a grid of buttons, each one respective to their sampel image, with the purpose of starting the upscaling process. The upscaling process will resume the denoising process for the sample image, as well as upscale the final result, using one of the upsampling models available.

Here are some examples of images generated with the text to image pipeline, and upscaled with RealESRGAN:

### /inator a city on an island floating in the sky with clouds in the background, digital art, detailed, 4k --seed 0
![upscaled_realesrgan](https://user-images.githubusercontent.com/75852333/193890780-5c0e6340-e3f7-4fd0-abd0-5e8b2e693393.png)

---

### /inator dark and terrifying house living room interior overview design, haunting creepy demon is standing in the corner of the room, 8K, ultra wide angle, higly detailed --wallpaper
![dark_and_terrifying_horror_house_living_room_interior_overview_design_demon_with_red_eyes_is_standing_in_the_corner_Moebius_Greg_Rutkowski_Zabrocki_Karlkka_Jayison_Devadas_Phuoc_Quan_trending_on_Artstation_8K_ultra_w](https://user-images.githubusercontent.com/75852333/194189523-02b811e0-9d7a-4e3f-9452-7d979b6bd8b1.png)

---

### /inator temple in ruines, forest, stairs, columns, cinematic, detailed, atmospheric, epic, concept art, Matte painting, background, mist, photo-realistic, concept art, volumetric light, cinematic epic, rule of thirds, octane render, 8k, corona render, movie concept art, octane render, cinematic, trending on artstation, movie concept art, cinematic composition, ultra-detailed, realistic, hyper-realistic, volumetric lighting, 8k
![temple_in_ruines_forest_stairs_columns_cinematic_detailed_atmospheric_epic_concept_art_Matte_painting_background_mist_photo-realistic_concept_art_volumetric_light_cinematic_epic_rule_of_thirds_octane_render_8k_co (1)](https://user-images.githubusercontent.com/75852333/193936790-f19d2e4d-83ad-492b-b302-3581f1cb043a.png)

---

/inator adorable little raindrop creature designed by Naoto Hattori, symmetrical, extremely cute, rainy, rainstorm, clouds, serene, blue color scheme, extremely intricate, ornate, hyperdetailed, hypermaximalist, hyperrealistic, volumetric lighting, octane render, ultra HD, 8k --portrait
![adorable_little_raindrop_creature_designed_by_Naoto_Hattori_symmetrical_extremely_cute_rainy_rainstorm_clouds_serene_blue_color_scheme_extremely_intricate_ornate_hyperdetailed_hypermaximalist_hyperrealistic_volumetri](https://user-images.githubusercontent.com/75852333/194190449-900c4672-ded6-4dc5-8458-3a3f853a08b3.png)

---

/inator tree house in the forest, atmospheric, hyper realistic, epic composition, cinematic, landscape vista photography by Carr Clifton & Galen Rowell, 16K resolution, Landscape veduta photo by Dustin Lefevre & tdraw, detailed landscape painting by Ivan Shishkin, DeviantArt, Flickr, rendered in Enscape, Miyazaki, Nausicaa Ghibli, Breath of The Wild, 4k detailed post processing, artstation, unreal engine
![tree_house_in_the_forest_atmospheric_hyper_realistic_epic_composition_cinematic_landscape_vista_photography_by_Carr_Clifton__Galen_Rowell_16K_resolution_Landscape_veduta_photo_by_Dustin_Lefevre__tdraw_detailed_landsc](https://user-images.githubusercontent.com/75852333/194189909-a5645ceb-2ec1-4771-90d8-bd8fe2c36b3e.png)


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
