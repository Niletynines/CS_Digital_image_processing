from PIL import Image
import numpy as np
from ImageManager import ImageManager

img = Image.open("Final\image\FinalDIP67.bmp")
data = np.array(img)

# shape array
width = data.shape[0]
height = data.shape[1]

mode_to_bpp = {"1": 1, "L": 8, "P": 8, "RGB": 24, "RGBA": 32, "CMYK": 32, "YCbCr": 24, "LAB": 24, "HSV": 24, "I": 32, "F": 32}
bitDepth = mode_to_bpp[img.mode]

print("Image %s with %s x %s pixels (%s bits per pixel) has been read!" % (img, width, height, bitDepth))

im = ImageManager()
im.read("Final\image\FinalDIP67.bmp")

# Define source and destination points
srcPoints = np.array([
    [45, 135], # top-left
    [546, 66], # top-right
    [750, 319], # bottom-right
    [70, 514] # bottom-left
])
dstPoints = np.array([
    [0, 0], # top-left
    [0, 800], # top-right
    [600, 800], # bottom-right
    [600, 0] # bottom-left
])

H = im.calculateHomography(srcPoints, dstPoints)
im.applyHomography(H)
im.write("Final\image\FinalHomo.bmp")

# im.convertToGrayscale()
# im.write("Final\image\Finalgray.bmp")

im.medianFilter(7)
im.write("Final\image\FinalDenoise.bmp")

TempImg = im.otsuThreshold()
im.write("Final\image\FinalBinary.bmp")

im.find_bounding_boxes()

# # Find digit 5

# srcPoints = np.array([
#     [619, 274], # top-left
#     [664, 274], # top-right
#     [664, 358], # bottom-right
#     [619, 358] # bottom-left
# ])
# dstPoints = np.array([
#     [0, 0], # top-left
#     [0, 45], # top-right
#     [84, 45], # bottom-right
#     [84, 0] # bottom-left
# ])

# H = im.calculateHomography(srcPoints,dstPoints)
# im.applyHomography(H)
# im.write("Final\image\Digit5.bmp")