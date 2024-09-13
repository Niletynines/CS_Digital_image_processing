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
    
    # the RED channel
    def convertToRed(self):
        global data
        for y in range(height):
            for x in range(width):
                data[x, y, 1] = 0 # Set green channel to 0
                data[x, y, 2] = 0 # Set blue channel to 0
                
    # the GREEN channel
    def convertToGreen(self):
        global data
        for y in range(height):
            for x in range(width):
                data[x, y, 0] = 0 # Set red channel to 0
                data[x, y, 2] = 0 # Set blue channel to 0
                
    # the BLUE channel
    def convertToBlue(self):
        global data
        for y in range(height):
            for x in range(width):
                data[x, y, 0] = 0 # Set red channel to 0
                data[x, y, 1] = 0 # Set green channel to 0
                
    # the GRAY channel
    def convertToGray(self):
        global data
        for y in range(height):
            for x in range(width):
                r = data[x, y, 0]
                g = data[x, y, 1]
                b = data[x, y, 2]
                
                gray = (int)((0.2126*r) + int(0.7152*g) + int(0.0722*b))
                # gray = int(np.mean(data[x, y, :3]))  # Compute the average of R, G, B channels
                
                data[x, y, 0] = gray  # Set red channel to gray
                data[x, y, 1] = gray  # Set green channel to gray
                data[x, y, 2] = gray  # Set blue channel to gray

    # copy
    def restoreToOriginal(self):
        global data
        data = np.copy(original)
        
    