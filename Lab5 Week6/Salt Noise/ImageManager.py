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
    
    # Salt Noise
    def addSaltNoise(self, percent):
        global data
        
        noOfPX = height * width
        noiseAdded = (int)((percent/100) * noOfPX)
        
        whiteColor = 255
        
        for i in range(noiseAdded):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            data[x, y, 0] = whiteColor
            data[x, y, 1] = whiteColor
            data[x, y, 2] = whiteColor