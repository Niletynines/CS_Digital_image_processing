from PIL import Image
import numpy as np
from ImageManager import ImageManager

img = Image.open("images/qrcode.bmp")
data = np.array(img)

# shape array
width = data.shape[0]
height = data.shape[1]

mode_to_bpp = {"1": 1, "L": 8, "P": 8, "RGB": 24, "RGBA": 32, "CMYK": 32, "YCbCr": 24, "LAB": 24, "HSV": 24, "I": 32, "F": 32}
bitDepth = mode_to_bpp[img.mode]

print("Image %s with %s x %s pixels (%s bits per pixel) has been read!" % (img, width, height, bitDepth))

im = ImageManager()
im.read("images/qrcode.bmp")

# Define source and destination points
srcPoints = np.array([
    [256, 133], # top-left
    [419, 146], # top-right
    [403, 348], # bottom-right
    [244, 320] # bottom-left
])
dstPoints = np.array([
    [0, 0], # top-left
    [512, 0], # top-right
    [512, 512], # bottom-right
    [0, 512] # bottom-left
])

H = im.calculateHomography(srcPoints, dstPoints)
im.applyHomography(H)
im.write("images/qrcode2.bmp")

#  0 -1  2 
# -3  4 -5
#  6 -7  8

#ADJ
#  0 -3  6
# -1  4 -7
#  2 -5  8