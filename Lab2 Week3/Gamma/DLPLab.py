from PIL import Image
import numpy as np
from ImageManager import ImageManager

img = Image.open("image/mandril.bmp")
data = np.array(img)

# shape array
width = data.shape[0]
height = data.shape[1]

mode_to_bpp = {"1": 1, "L": 8, "P": 8, "RGB": 24, "RGBA": 32, "CMYK": 32, "YCbCr": 24, "LAB": 24, "HSV": 24, "I": 32, "F": 32}
bitDepth = mode_to_bpp[img.mode]

print("Image %s with %s x %s pixels (%s bits per pixel) has been read!" % (img, width, height, bitDepth))

# Gamma 0.4
im1 = ImageManager()
im1.read("image/mandril.bmp")
im1.adjustGamma(0.4)
im1.write("image/gamma0_4.bmp")

# Gamma 2.2
im2 = ImageManager()
im2.read("image/mandril.bmp")
im2.adjustGamma(2.2)
im2.write("image/gamma2_2.bmp")