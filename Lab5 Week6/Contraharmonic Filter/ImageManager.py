import math
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
    
    # Contraharmonic Filter
    def contraharmonicFilter(self, size, Q):
        global data
        if (size % 2 == 0):
            print("Size Invalid: must be odd number!")
            return

        data_temp = np.zeros([width, height, 3])
        data_temp = data.copy()
        for y in range(height):
            for x in range(width):
                sumRedAbove = 0
                sumGreenAbove = 0
                sumBlueAbove = 0
                sumRedBelow = 0
                sumGreenBelow = 0
                sumBlueBelow = 0
                subData = data_temp[x - int(size/2):x + int(size/2) + 1, y - int(size/2):y + int(size/2) + 1, :].copy()
                subData = subData ** (Q + 1)
                sumRedAbove = np.sum(subData[:,:,0:1], axis=None)
                sumGreenAbove = np.sum(subData[:,:,1:2], axis=None)
                sumBlueAbove = np.sum(subData[:,:,2:3], axis=None)
                subData = data_temp[x - int(size/2):x + int(size/2) + 1, y - int(size/2):y + int(size/2) + 1, :].copy()
                subData = subData ** Q
                sumRedBelow = np.sum(subData[:,:,0:1], axis=None)
                sumGreenBelow = np.sum(subData[:,:,1:2], axis=None)
                sumBlueBelow = np.sum(subData[:,:,2:3], axis=None)
                if (sumRedBelow != 0): sumRedAbove /= sumRedBelow
                sumRedAbove = 255 if sumRedAbove > 255 else sumRedAbove
                sumRedAbove = 0 if sumRedAbove < 0 else sumRedAbove
                if (math.isnan(sumRedAbove)): sumRedAbove = 0
                if (sumGreenBelow != 0): sumGreenAbove /= sumGreenBelow
                sumGreenAbove = 255 if sumGreenAbove > 255 else sumGreenAbove
                sumGreenAbove = 0 if sumGreenAbove < 0 else sumGreenAbove
                if (sumBlueBelow != 0): sumBlueAbove /= sumBlueBelow
                sumBlueAbove = 255 if sumBlueAbove > 255 else sumBlueAbove
                sumBlueAbove = 0 if sumBlueAbove < 0 else sumBlueAbove
                if (math.isnan(sumBlueAbove)): sumBlueAbove = 0
                data[x, y, 0] = sumRedAbove
                data[x, y, 1] = sumGreenAbove
                data[x, y, 2] = sumBlueAbove    