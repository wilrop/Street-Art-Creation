import os
import numpy as np
from os import path
from PIL import Image


WIDTH = 512
HEIGHT = 512


if __name__ == "__main__":
    n = 20
    dir = ""
    for idx, filename in enumerate(os.listdir(dir)):
        inp = path.join(dir, filename)
        img = Image.open(inp)
        img = img.resize((WIDTH, HEIGHT))
        outp = path.join(dir, "res" + str(idx + 1) + ".png")
        img.save(outp)
