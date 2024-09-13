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

# median nonlinear filtering 3*3
im = ImageManager()
im.read("image/mandril.bmp")
im.medianFilter(3)
im.write("image/Median3x3.bmp")

# median nonlinear filtering 7*7
im = ImageManager()
im.read("image/mandril.bmp")
im.medianFilter(7)
im.write("image/Median7x7.bmp")

# median nonlinear filtering 15*15
im = ImageManager()
im.read("image/mandril.bmp")
im.medianFilter(15)
im.write("image/Median15x15.bmp")
