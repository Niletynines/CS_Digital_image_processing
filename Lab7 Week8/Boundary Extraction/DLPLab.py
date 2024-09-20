from PIL import Image
import numpy as np
from ImageManager import ImageManager
from ImageManager import StructuringElement

img = Image.open("image\mandrilB.bmp")
data = np.array(img)

# shape array
width = data.shape[0]
height = data.shape[1]

mode_to_bpp = {"1": 1, "L": 8, "P": 8, "RGB": 24, "RGBA": 32, "CMYK": 32, "YCbCr": 24, "LAB": 24, "HSV": 24, "I": 32, "F": 32}
bitDepth = mode_to_bpp[img.mode]

print("Image %s with %s x %s pixels (%s bits per pixel) has been read!" % (img, width, height, bitDepth))

im = ImageManager()
im.read("image\mandrilB.bmp")

se = StructuringElement(3, 3, complex(1, 1))
se.elements[True] = 255

im.boundaryExtraction(se)
im.write("image\BoundaryExtraction.bmp")