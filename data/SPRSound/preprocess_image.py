
import os
from PIL import Image

def preprocess(image)

    im = Image.open(image)
    process = im.resize((256, 256), Image.ANTIALIAS)
    
    return process

