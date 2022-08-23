import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

import discord
from discord.ext import commands

import os
import dotenv

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='>', intents=intents)

dotenv.load_dotenv()
pip = None

@bot.command()
async def dream(ctx, prompt: str):
    await ctx.message.add_reaction('👍')
    prompt = 2 * [prompt]
    with torch.autocast("cuda"):
        image = pipe(prompt)["sample"]

    grid = image_grid(image, rows=1, cols=2)
    filename = os.path.join("outputs/{}.png".format(prompt[0].replace(" ", "_")))
    grid.save(filename)
    
    await ctx.reply(file=discord.File(filename))

@bot.event
async def on_ready():
    global pipe
    # make sure you're logged in with `huggingface-cli login`
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
                                                   revision="fp16",
                                                   torch_dtype=torch.float16,
                                                   use_auth_token=True,
                                                   guidance_scale=7.5,
                                                   num_inference_steps=5000)  
    def dummy(images, **kwargs): return images, False 
    pipe.safety_checker = dummy
    pipe = pipe.to('cuda')
   


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid
    
bot.run(str(os.getenv("TOKEN")))


