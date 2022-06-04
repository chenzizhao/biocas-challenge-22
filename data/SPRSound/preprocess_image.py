
import os
from PIL import Image

def preprocess(image):

    im = Image.open(image)
    process = im.resize((224, 224), Image.ANTIALIAS)
    
    return process

