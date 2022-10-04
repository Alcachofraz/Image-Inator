import os
import json
import numpy
import torch
from PIL import Image
import cv2
from cv2 import dnn_superres
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.archs.rrdbnet_arch import RRDBNet
import re
from config import *

SERVERS_CHANNELS_FILE_PATH = 'servers_channels.txt'


"""
************************
**                    **
**  HELPER FUNCTIONS  **
**                    **
************************
"""

"""
-----------------------------------------------------------
Convert images to an unique image containing a grid of images.

@param images: Original images
@param rows: Grid rows
@param cols: Grid columns
@return: Grid image
-----------------------------------------------------------
"""
def images_to_grid(images, rows, cols):
    assert len(images) == rows*cols
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

"""
-----------------------------------------------------------
Create servers/channels map text file.
-----------------------------------------------------------
"""
def create_map_file():
    with open(SERVERS_CHANNELS_FILE_PATH, 'w') as file:
        file.write(json.dumps({}))

"""
-----------------------------------------------------------
Save map to servers/channels map text file.

@param map: Map to save
-----------------------------------------------------------
"""
def save_map_to_file(map):
    if not os.path.exists(SERVERS_CHANNELS_FILE_PATH):
        create_map_file()
    with open(SERVERS_CHANNELS_FILE_PATH, 'w') as file:
        file.write(json.dumps(map))

"""
-----------------------------------------------------------
Load map from servers/channels map text file.

@return: Map loaded
-----------------------------------------------------------
"""
def load_map_from_file():
    if not os.path.exists(SERVERS_CHANNELS_FILE_PATH):
        create_map_file()
    with open(SERVERS_CHANNELS_FILE_PATH, 'r') as file:
        try:
            return json.loads(file.read())
        except:
            return {}

"""
-----------------------------------------------------------
Register channel to bot. Will add an entry with the channel id to the 
server key in servers/channels map text file.

@param channel: Id of the channel to register
@param server: Id of the server
-----------------------------------------------------------
"""
def register_channel(channel, server):
    map = load_map_from_file()
    if server in map:
        if channel not in map[server]:
            map[server] += [channel]
    else:
        map[server] = [channel]
    save_map_to_file(map)

"""
-----------------------------------------------------------
Unregister channel from bot. Will remove the entry with the channel id from the 
server key in servers/channels map text file.

@param channel: Id of the channel to register
@param server: Id of the server
-----------------------------------------------------------
"""
def unregister_channel(channel, server):
    map = load_map_from_file()
    if server in map:
        map[server] -= [channel]
    save_map_to_file(map)

"""
-----------------------------------------------------------
Checks if a channel is registered to the bot.

@param channel: Id of the channel to check
@param server: Id of the server
@return: Whether the channel is registered or not
-----------------------------------------------------------
"""
def is_channel_registered(channel, server):
    map = load_map_from_file()
    if server in map:
        return channel in map[server]
    return False

"""
-----------------------------------------------------------
Extract arguments from the user prompt. Also remove the arguments from 
the content.

@param content: Prompt content (with arguments)
@return: A map with the arguments and the new content
-----------------------------------------------------------
"""
def extract_args(content):
    args = {
        'wallpaper': False,
        'portrait': False,
        'skip_grid': False,
    }
    if '--wallpaper' in content:
        args['wallpaper'] = True
        content = content.replace('--wallpaper', '')
    elif '--portrait' in content:
        args['portrait'] = True
        content = content.replace('--portrait', '')
    if '--skip_grid' in content:
        args['skip_grid'] = True
        content = content.replace('--skip_grid', '')
    if '--seed' in content:
        aux = content.split()
        seed = str(aux[aux.index('--seed')+1])
        args['seed'] = int(seed)
        content = content.replace('--seed', '')
        content = content.replace(seed, '')
    return args, re.sub(' +', ' ', content).strip()

"""
-----------------------------------------------------------
Converts a PIL image to a OpenCV image.

@param pil_image: PIL image
@return: OpenCV image
-----------------------------------------------------------
"""
def pil_to_cv2(pil_image):
    image = numpy.array(pil_image)  
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

"""
-----------------------------------------------------------
Converts a OpenCV image to a PIL image.

@param cv2_image: OpenCV image
@return: PIL image
-----------------------------------------------------------
"""
def cv2_to_pil(cv2_image):
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_image)

"""
-----------------------------------------------------------
Upscale an image.

@param image: Image to upscale
@param cuda_available: Whether CUDA is available or not
@param config: Config object containing generation parameters
@return: Upscaled image
-----------------------------------------------------------
"""
def upscale_image(image, cuda_available, config):
    print('Upsampling using ' + config.upscaling_model + '...')
    image = pil_to_cv2(image)
    if (config.upscaling_model == 'realesrgan'):
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        sr = RealESRGANer(
            4,
            model_path='upscaling_models/' + config.upscaling_model + '.pth',
            dni_weight=None,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=cuda_available,
            gpu_id=0 if cuda_available else None,
        )
        upsampled, _ = sr.enhance(image, outscale=4)
    else:
        sr = dnn_superres.DnnSuperResImpl_create()
        path = 'upscaling_models/' + config.upscaling_model + '.pb'
        sr.readModel(path)
        if (cuda_available):
            sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        sr.setModel(config.upscaling_model, 4)
        upsampled = sr.upsample(image)
    print('Upsampling done')
    return cv2_to_pil(upsampled)

"""
-----------------------------------------------------------
Convert text prompt to an image.

@param prompt: User prompt (without arguments)
@param args: Prompt arguments
@param pipe: Pipeline
@param config: Config object containing generation parameters
@param generator: Pipeline seed deterministic generator
@return: Number of images generated and the images themselves
-----------------------------------------------------------
"""
def text_to_image(prompt, args, pipe, config, generator):
    grid = not args['skip_grid']
    grid_cols = config.grid_size
    guidance = config.guidance_scale
    steps = config.steps
    width = config.wallpaper_width if args['wallpaper'] else (config.portrait_width if args['portrait'] else config.square_width)
    height = config.wallpaper_height if args['wallpaper'] else (config.portrait_height if args['portrait'] else config.square_height)
    print('Generating image with grid=' + str(grid) + ' grid_size=' + str(grid_cols) + ' guidance=' + str(guidance) + ' steps=' + str(steps) + ' width=' + str(width) +  ' height=' + str(height))
    torch.cuda.empty_cache()
    images = pipe([prompt]*(grid_cols*grid_cols) if grid else prompt, height=height, width=width, guidance_scale=guidance, num_inference_steps=steps, generator=generator).images
    print('Image generated')
    return len(images), images

"""
-----------------------------------------------------------
Convert text prompt to an image, using CPU.

@param prompt: User prompt (without arguments)
@param args: Prompt arguments
@param pipe: Pipeline
@param config: Config object containing generation parameters
@return: Number of images generated, the images themselves and the seed
-----------------------------------------------------------
"""
def text_to_image_with_cpu(prompt, args, pipe, config):
    if ('seed' in args):
        seed = args['seed']
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = torch.Generator()
        seed = generator.seed()
    n, images = text_to_image(prompt, args, pipe, config, generator)
    return n, images, seed

"""
-----------------------------------------------------------
Convert text prompt to an image, using GPU (CUDA).

@param prompt: User prompt (without arguments)
@param args: Prompt arguments
@param pipe: Pipeline
@param config: Config object containing generation parameters
@return: Number of images generated, the images themselves and the seed
-----------------------------------------------------------
"""
def text_to_image_with_cuda(prompt, args, pipe, config):
    with torch.autocast("cuda"):
        if ('seed' in args):
            seed = args['seed']
            generator = torch.Generator('cuda').manual_seed(seed)
        else:
            generator = torch.Generator('cuda')
            seed = generator.seed()
        n, images = text_to_image(prompt, args, pipe, config, generator)
        return n, images, seed
