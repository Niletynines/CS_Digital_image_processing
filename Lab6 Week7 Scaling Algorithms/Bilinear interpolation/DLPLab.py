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

# bilinear interpolation Scale 3.5
im = ImageManager()
im.read("image/mandril.bmp")
im.resizeBilinear(3.5,3.5)
im.write("image/bilinearInterpolation3_5.bmp")

# bilinear interpolation Scale 0.35
im1 = ImageManager()
im1.read("image/mandril.bmp")
im1.resizeBilinear(0.35,0.35)
im1.write("image/bilinearInterpolation0_35.bmp")
