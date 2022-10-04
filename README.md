# Gerador-De-Imagens-Inator

This is a Doofenshmirtz inspired Discord bot that takes advantage of the open source python package [Diffusers](https://huggingface.co/blog/stable_diffusion) to make text to image inferences inside a Discord channel.

The users have the ability to generate a grid of sample images as response to their prompt, and the bot will display a grid of buttons, each one respective to their sampel image, with the purpose of starting the upscaling process. The upscaling process will resume the denoising process for the sample image, as well as upscale the final result, using one of the upsampling models available.
