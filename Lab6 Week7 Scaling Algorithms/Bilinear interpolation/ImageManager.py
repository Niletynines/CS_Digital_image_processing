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

    # bilinear interpolation
    def resizeBilinear(self, scaleX, scaleY):
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
                oldX = x / scaleX
                oldY = y / scaleY
                
                #get 4 coordinates
                x1 = min((int)(np.floor(oldX)), width - 1)
                y1 = min((int)(np.floor(oldY)), height - 1)
                x2 = min((int)(np.ceil(oldX)), width - 1)
                y2 = min((int)(np.ceil(oldY)), height - 1)
                
                #get colours
                color11 = np.array(data_temp[x1, y1, :])
                color12 = np.array(data_temp[x1, y2, :])
                color21 = np.array(data_temp[x2, y1, :])
                color22 = np.array(data_temp[x2, y2, :])
                
                #interpolate x
                P1 = (x2 - oldX) * color11 + (oldX - x1) * color21
                P2 = (x2 - oldX) * color12 + (oldX - x1) * color22
                if x1 == x2:
                    P1 = color11
                    P2 = color22
                
                #interpolate y
                P = (y2 - oldY) * P1 + (oldY - y1) * P2
                if y1 == y2:
                    P = P1
                P = np.round(P)
                data[x, y, :] = P