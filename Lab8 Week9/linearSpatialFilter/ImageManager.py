import sys
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
            
    def convertToGrayscale(self):
        global data
        for y in range(height):
            for x in range(width):
                r = data[x, y, 0]
                g = data[x, y, 1]
                b = data[x, y, 2]
                gray = (int)((0.2126*r) + int(0.7152*g) + int(0.0722*b))
                data[x, y, 0] = gray  
                data[x, y, 1] = gray  
                data[x, y, 2] = gray  
                
    # Thresholding
    def thresholding(self, threshold):
        global data
        self.convertToGrayscale()
        for y in range(height):
            for x in range(width):
                gray = data[x, y, 0]
                gray = 0 if gray < threshold else 255
                data[x, y, 0] = gray
                data[x, y, 1] = gray
                data[x, y, 2] = gray
    
    # linear Spatial Filter otsuThreshold    
    def linearSpatialFilter(self, kernel, size):
        global data
        if (size % 2 ==0):
            print("Size Invalid: must be odd number!")
            return
        data_zeropaded = np.zeros([width + int(size/2) * 2, height + int(size/2) * 2, 3])
        data_zeropaded[int(size/2):width + int(size/2), int(size/2):height + int(size/2), :] = data
        for y in range(int(size/2), int(size/2) + height):
            for x in range(int(size/2), int(size/2) + width):
                subData = data_zeropaded[x - int(size/2):x + int(size/2) + 1, y -int(size/2):y + int(size/2) + 1, :]
                sumRed = np.sum(np.multiply(subData[:,:,0:1].flatten(), kernel))
                sumGreen = np.sum(np.multiply(subData[:,:,1:2].flatten(), kernel))
                sumBlue = np.sum(np.multiply(subData[:,:,2:3].flatten(), kernel))
                sumRed = 255 if sumRed > 255 else sumRed
                sumRed = 0 if sumRed < 0 else sumRed
                sumGreen = 255 if sumGreen > 255 else sumGreen
                sumGreen = 0 if sumGreen < 0 else sumGreen
                sumBlue = 255 if sumBlue > 255 else sumBlue
                sumBlue = 0 if sumBlue < 0 else sumBlue
                data[x - int(size/2), y - int(size/2), 0] = sumRed
                data[x - int(size/2), y - int(size/2), 1] = sumGreen
                data[x - int(size/2), y - int(size/2), 2] = sumBlue