import random
from PIL import Image
import numpy as np

class ImageManager:
    
    #attributes
    width = None
    height = None
    bitDepth = None
    
    img = None
    data = None
    original = None
    
    def read(self, fileName):
        global img
        global data
        global original
        global width
        global height
        global bitDepth
        img = Image.open(fileName)
        data = np.array(img)
        original = np.copy(data)
        width = data.shape[0]
        height = data.shape[1]
        mode_to_bpp = {"1": 1, "L": 8, "P": 8, "RGB": 24, "RGBA": 32, "CMYK": 32,"YCbCr": 24, "LAB": 24, "HSV": 24, "I": 32, "F": 32}
        bitDepth = mode_to_bpp[img.mode]
        print("Image %s width %s x %s pixels (%s bits per pixel) has been read!" % (img, width, height, bitDepth))

    def write(self, fileName):
        global img
        img = Image.fromarray(data)
        try:
            img.save(fileName)
        except:
            print("Write file error")
        else:
            print("Image %s has been written!" % (fileName))
    
    def restoreToOriginal(self):
        global data
        width = original.shape[0]
        height = original.shape[1]
        data = np.zeros([width, height, 3])
        data = np.copy(original)

    # Nearest-neighbour
    def resizeNearestNeighbour(self, scaleX, scaleY):
        global data
        global width
        global height
        newWidth = (int)(round(width * scaleX))
        newHeight = (int)(round(height * scaleY))
        data_temp = np.zeros([width, height, 3])
        data_temp = data.copy()
        data = np.resize(data, [newWidth, newHeight, 3])
        for y in range(newHeight):
            for x in range(newWidth):
                xNearest = (int)(round(x / scaleX))
                yNearest = (int)(round(y / scaleY))
                xNearest = width - 1 if xNearest >= width else xNearest
                xNearest = 0 if xNearest < 0 else xNearest
                yNearest = height - 1 if yNearest >= height else yNearest
                yNearest = 0 if yNearest < 0 else yNearest
                data[x, y, :] = data_temp[xNearest, yNearest, :]