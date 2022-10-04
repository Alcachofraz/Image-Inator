import os
import torch
import discord
from discord.ui import View, Button
from dotenv import load_dotenv
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from io import BytesIO
import math
from utils import *
from config import Config
import random



"""
*****************
**             **
**  CONSTANTS  **
**             **
*****************
"""

CUDA_AVAILABLE = torch.cuda.is_available()
LOADING_GIFS = [
    # Já feitos:
    'https://media3.giphy.com/media/IeGSrFUxFxxLO/giphy.gif?cid=790b761105241689701d9ba0fc5954fb5c785d57372e7104&rid=giphy.gif&ct=g',
    'https://media.tenor.com/pSBhD_1sL50AAAAC/doof-dr-doof.gif',
    'https://media.tenor.com/Ozkxdmb6O78AAAAC/phineasand-ferb-perrythe-platypus.gif',
    'https://media.tenor.com/XJ0be-IM7hwAAAAC/phineasand-ferb-doofenshmirtz.gif',
    'https://25.media.tumblr.com/eb5d68d46483ea1af1fd33441f3e3498/tumblr_mpys28kMBk1r1tn8jo1_500.gif',
    # Serginho:
    'https://media.tenor.com/twyrTrmsKgUAAAAC/heinz-doofenshmirtz-doofenshmirtz.gif',
    'https://media.tenor.com/cl1rTifVKpAAAAAC/heinz-doofenshmirtz-guitar.gif',
    'https://media.tenor.com/xRp0M3kcSGUAAAAC/heinz-doofenshmirtz-heinz-doofenshmirtz-dancing.gif',
]




"""
***********************
**                   **
**  IINITIALISATION  **
**                   **
***********************
"""

# Load environment vairables:
load_dotenv()
DISCORD_TOKEN = os.getenv('TOKEN')
PIPELINE_TOKEN = os.getenv('PIPELINE_TOKEN')

# Initialise bot with all intents:
intents = discord.Intents.all()
client = discord.Client(intents=intents)

# Initialise config:
config = Config()

# Initialise text to image pipeline:
text_to_image_pipe = StableDiffusionPipeline.from_pretrained(
    'CompVis/stable-diffusion-v1-4',
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=PIPELINE_TOKEN,
) if CUDA_AVAILABLE else StableDiffusionPipeline.from_pretrained(
    'CompVis/stable-diffusion-v1-4',
    use_auth_token=PIPELINE_TOKEN,
)
if CUDA_AVAILABLE:
    text_to_image_pipe.enable_attention_slicing()
    text_to_image_pipe.to("cuda")

# Dummy checker to bypass pipelines' NSFW filter:
def dummy_checker(images, **kwargs):
    return images, False
text_to_image_pipe.safety_checker = dummy_checker

torch.cuda.memory_summary(device=None, abbreviated=False)



"""
****************************
**                        **
**  BOT HELPER FUNCTIONS  **
**                        **
****************************
"""

"""
-----------------------------------------------------------
This is the message the bot sends when "/inator help" is issued.

@param send: Bot's send function
-----------------------------------------------------------
"""
async def help(send):
    return await send("""
    \"/inator iniciar\" para associar o bot a um canal.\n
    \"/inator remover\" para remover o bot de um canal.\n
    \"/inator config" para configurar os parâmetros da geração.\n
    \"/inator <o que te vai na mente>" para gerar uma imagem.
    """)

"""
-----------------------------------------------------------
This is the message the bot sends when it's busy.

@param send: Bot's send function
@param prompt: The prompt of the image that's generating
@param upscaling: Whether it's upscaling or sampling
-----------------------------------------------------------
"""
async def wait(send, prompt, upscaling=False):
    embed = discord.Embed(title="", description="")
    embed.set_image(url=random.choice(LOADING_GIFS))
    return await send('A ' + ('melhorar' if upscaling else 'gerar') + ' **' + prompt + '**... :clock4:\n> ```' + ('Aguarda uns segundinhos' if upscaling else 'Isto pode demorar alguns minutos') + '. Enquanto isso, ignorarei todas as mensagens neste canal.```', embed=embed)

"""
-----------------------------------------------------------
This is the message the bot sends when "/inator config" is issued.

@param send: Bot's send function
-----------------------------------------------------------
"""
async def show_config(send):
    await send(
        'steps  -->**  ' + str(config.steps) + '**' +
        '\nsquare  -->**  ' + str(config.square_width) + ':' + str(config.square_height) + '**' +
        '\nwallpaper  -->**  ' + str(config.wallpaper_width) + ':' +  str(config.wallpaper_height) + '**' +
        '\nportrait_height  -->**  ' + str(config.portrait_width) + ':' + str(config.portrait_height) + '**' +
        '\ngrid_size  -->**  ' + str(config.grid_size) + '**' +
        '\nguidance_scale  -->**  ' + str(config.guidance_scale) + '**' +
        '\nupscaling_model  -->**  ' + str(config.upscaling_model) + '**'
    )

"""
-----------------------------------------------------------
On ready callback. 
Called when the bot is started. Prints login information to console.
-----------------------------------------------------------
"""
@client.event
async def on_ready():
    print("Logged in as a bot {0.user}".format(client))

""" 
-----------------------------------------------------------
Upscale image. Will send a "wait" message and start upscaling the image.
When upscaling is complete, the bot delete the "wait" message and sends the new upscaled image.

@param send: Bot's send function
@param prompt: The prompt of the image to upscale
@param args: Generation arguments for the image
@param seed: Image seed
@param i: Image index
@param image: Image to upscale
-----------------------------------------------------------
"""
async def on_upscale(send, prompt, args, seed, i, image):
    response = await wait(send, prompt, upscaling=True)
    image = upscale_image(image, CUDA_AVAILABLE, config)
    with BytesIO() as binary:
        image.save(binary, 'PNG')
        binary.seek(0)
        await response.delete()
        try:
            await send('**' + prompt + '**\n> ```Semente: ' + str(seed) + ' (' + str(i) + ')```', file=discord.File(fp=binary, filename=prompt.replace(' ', '_') + '.png'))
        except:
            await send('A imagem é demasiado pesada, o Discord não se aguenta.')

"""
-----------------------------------------------------------
Generate button callback.

@param i: Image index
@param send: Bot's send function
@param prompt: Prompt that generated the image
@param seed: Image seed
@param image: Image to upscale with button
@return: Button callback
-----------------------------------------------------------
"""
def get_button_call_back(i, send, prompt, args, seed, image):
    async def button_callback(interaction: discord.Interaction):
        await interaction.response.defer()
        await on_upscale(send, prompt, args, seed, i+1, image)
    return button_callback

""" 
-----------------------------------------------------------
Generate image. Will send a "wait" message and start generating the image.
When upscaling is complete, the bot delete the "wait" message and sends the new upscaled image.

@param send: Bot's send function
@param prompt_with_args: Prompt with arguments
-----------------------------------------------------------
"""
async def on_generate(send, prompt_with_args):
    args, prompt = extract_args(prompt_with_args)
    response = await wait(send, prompt)
    n, images, seed = text_to_image_with_cuda(prompt, args, text_to_image_pipe, config) if CUDA_AVAILABLE else text_to_image_with_cpu(prompt, args, text_to_image_pipe, config)
    image = images_to_grid(images, rows=int(math.sqrt(n)), cols=int(math.sqrt(n)))
    with BytesIO() as binary:
        image.save(binary, 'PNG')
        binary.seek(0)
        view = View(timeout=86400) # Timeout of 1 day
        if (n == 1):
            button = Button(label='Melhorar 🔬', style=discord.ButtonStyle.secondary)
            button.callback = get_button_call_back(0, send, prompt, args, seed, images[0])
            view.add_item(button)
        else:
            button_top_left = Button(label='Melhorar 🔬 ↖', style=discord.ButtonStyle.secondary)
            button_top_left.callback = get_button_call_back(0, send, prompt, args, seed, images[0])
            button_top_right = Button(label='Melhorar 🔬 ↗', style=discord.ButtonStyle.secondary)
            button_top_right.callback = get_button_call_back(1, send, prompt, args, seed, images[1])
            button_bottom_left = Button(label='Melhorar 🔬 ↙', style=discord.ButtonStyle.secondary)
            button_bottom_left.callback = get_button_call_back(2, send, prompt, args, seed, images[2])
            button_bottom_right = Button(label='Melhorar 🔬 ↘', style=discord.ButtonStyle.secondary)
            button_bottom_right.callback = get_button_call_back(3, send, prompt, args, seed, images[3])
            view.add_item(button_top_left)
            view.add_item(button_top_right)
            view.add_item(button_bottom_left)
            view.add_item(button_bottom_right)
        await response.delete()
        try:
            await send('**' + prompt + '**\n> ```Semente: ' + str(seed) + '```\nTens 24 horas para fazer melhorias.', file=discord.File(fp=binary, filename=prompt.replace(' ', '_') + '.png'), view=view)
        except:
            await send('A imagem é demasiado pesada, o Discord não se aguenta.')

"""
-----------------------------------------------------------
On message callback.
Called when the a message is sent on any channel of the server.

@param message: Discord message object
-----------------------------------------------------------
"""
@client.event
async def on_message(message: discord.Message):
    # If the message was sent by the bot:
    if (message.author.bot):
        return

    # Get message content:
    content = str(message.content)

    # Check for command:
    if content.startswith('/inator'):

        # If "/inator start" command, associate the bot to this message's channel:
        if content == '/inator start':
            register_channel(str(message.channel.id), str(message.guild.id))
            await message.channel.send('Feito! Podes começar a criar. Escreve o que te apetecer neste canal, e o inator ilustra.')

        # If "/inator remove" command, remove the bot from this message's channel:
        elif content == '/inator remove':
            unregister_channel(str(message.channel.id), str(message.guild.id))
            await message.channel.send('Removeste o inator deste canal.')

        # Check for "/inator config" command:
        elif content.startswith('/inator config'):
            # If "/inator config" command alone, show config parameters:
            if (content == '/inator config'):
                await show_config(message.channel.send)
            # If "/inator config" contains parameters, update config:
            else:
                result = config.update(content.replace('/inator config ', ''))
                await message.channel.send('Configurações atualizadas.' if result is None else result)
        
        # If "/inator help" command, show possible commands:
        elif content == '/inator help':
            await help(message.channel.send)

        # If "/inator <uknown>":
        else:
            # If channel has started inator:
            if is_channel_registered(str(message.channel.id), str(message.guild.id)):
                content = content.replace('/inator', '').strip() # Trim content

                # If command is empty, show possible commands:
                if (len(content) == 0):
                    await help(message.channel.send) 

                # If 
                else :
                    await on_generate(message.channel.send, content)
            
            # Error, because channel isn't configured to run inator:
            else:
                await message.channel.send('Este canal ainda não foi configurado. Utiliza "/inator start".')

# Run bot!
client.run(DISCORD_TOKEN)
