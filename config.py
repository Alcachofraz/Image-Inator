class Config:
    """
    Default config values
    """
    def __init__(self):
        self.grid_size = 2
        self.guidance_scale = 7.5
        self.square_height = 512
        self.square_width = 512
        self.wallpaper_height = 448
        self.wallpaper_width = 640
        self.portrait_height = 640
        self.portrait_width = 448
        self.steps = 50
        self.upscaling_model = 'realesrgan'

    """
    Replace config paramters with new_congig.

    @param new_config: The config with the new parameters.
    """
    def replace(self, new_config):
        self.grid_size = new_config.grid_size
        self.guidance_scale = new_config.guidance_scale
        self.square_height = new_config.square_height
        self.square_width = new_config.square_width
        self.wallpaper_height = new_config.wallpaper_height
        self.wallpaper_width = new_config.wallpaper_width
        self.portrait_height = new_config.portrait_height
        self.portrait_width = new_config.portrait_width
        self.steps = new_config.steps
        self.upscaling_model = new_config.upscaling_model
  
    """
    Update config parameters.

    @param content: Content of the command issued without "/inator config ".
    @return: A string identifying an error that occurred. In that case, nothing is changed. If no errors occurred, None is returned.
    """
    def update(self, content):
        temp = Config()
        temp.replace(self)
        words = content.split()
        for i, word in enumerate(words):
            if word.startswith('--'):
                word = word.replace('--', '')
                if word == 'steps':
                    steps = words[i+1]
                    if steps <= 0 or steps > 200:
                        return 'O parâmetro "steps" tem de estar entre 0 e 200.'
                    self.steps = int(steps)
                if word == 'square':
                    resolution = words[i+1].split(':')
                    width = int(resolution[0])
                    height = int(resolution[1])
                    if width != height:
                        return 'O parâmetro "height" tem de ser igual a "width".'
                    if width % 64 != 0:
                        return 'O parâmetro "width" tem de ser múltiplo de 64.'
                    if height % 64 != 0:
                        return 'O parâmetro "height" tem de ser múltiplo de 64.'
                    self.square_height = int(width)
                    self.square_width = int(height)
                if word == 'wallpaper':
                    resolution = words[i+1].split(':')
                    width = int(resolution[0])
                    height = int(resolution[1])
                    if width <= height:
                        return 'O parâmetro "width" tem de ser maior que "height".'
                    if width % 64 != 0:
                        return 'O parâmetro "width" tem de ser múltiplo de 64.'
                    if height % 64 != 0:
                        return 'O parâmetro "height" tem de ser múltiplo de 64.'
                    self.wallpaper_height = int(width)
                    self.wallpaper_width = int(height)
                if word == 'portrait':
                    resolution = words[i+1].split(':')
                    width = int(resolution[0])
                    height = int(resolution[1])
                    if width >= height:
                        return 'O parâmetro "height" tem de ser maior que "width".'
                    if width % 64 != 0:
                        return 'O parâmetro "width" tem de ser múltiplo de 64.'
                    if height % 64 != 0:
                        return 'O parâmetro "height" tem de ser múltiplo de 64.'
                    self.portrait_height = int(width)
                    self.portrait_width = int(height)
                if word == 'guidance_scale':
                    guidance = words[i+1]
                    if guidance > 1 or guidance < 0:
                        return 'O parâmetro "guidance_scale" tem de estar entre 0 e 1.'
                    self.guidance_scale = float(guidance)
                if word == 'upscaling_model':
                    model = words[i+1]
                    if model != 'edsr' and model != 'espcn' and model != 'fsrcnn' and model != 'lapsrn' and model != 'realesrgan':
                        return 'O parâmetro upscaling_model tem de ser um dos seguintes: "edsr", "espcn", "fsrcnn", "lapsrn" ou "realesrgan".'
                    self.upscaling_model = str(model)
        self.replace(temp)
