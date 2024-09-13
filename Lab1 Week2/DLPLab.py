from PIL import Image
import numpy as np

img = Image.open("image/mandril.bmp")
data = np.array(img)

# shape array
width = data.shape[0]
height = data.shape[1]

mode_to_bpp = {"1": 1, "L": 8, "P": 8, "RGB": 24, "RGBA": 32, "CMYK": 32, "YCbCr": 24, "LAB": 24, "HSV": 24, "I": 32, "F": 32}
bitDepth = mode_to_bpp[img.mode]

print("Image %s with %s x %s pixels (%s bits per pixel) has been read!" % (img, width, height, bitDepth))

# out = copy
fo = "image/out.bmp"

try:
    img.save(fo)
except:
    print("Write file error")
else:
    print("Image %s has been written!" % (fo))

from ImageManager import ImageManager

# red scale
imred = ImageManager()
imred.read("image/mandril.bmp")

imred.convertToRed()
imred.write("image/red.bmp")

# green scale
imgreen = ImageManager()
imgreen.read("image/mandril.bmp")

imgreen.convertToGreen()
imgreen.write("image/green.bmp")

# blue scale
imblue = ImageManager()
imblue.read("image/mandril.bmp")

imblue.convertToBlue()
imblue.write("image/blue.bmp")

# gray scale
imgray = ImageManager()
imgray.read("image/mandril.bmp")

imgray.convertToGray()
imgray.write("image/gray.bmp")

