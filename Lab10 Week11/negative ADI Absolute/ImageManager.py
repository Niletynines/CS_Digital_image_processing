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
                
    # negative ADI Absolute
    def negativeADIAbsolute(self, sequences, threshold, step):
        global data
        data_temp = np.zeros([width, height, 3])
        data_temp = np.copy(data)
        data[data > 0] = 0
        for n in range(len(sequences)):
            #read file
            otherImage = Image.open(sequences[n])
            otherData = np.array(otherImage)

            for y in range(height):
                for x in range(width):
                    dr = int(data_temp[x, y, 0]) - int(otherData[x, y, 0])
                    dg = int(data_temp[x, y, 1]) - int(otherData[x, y, 1])
                    db = int(data_temp[x, y, 2]) - int(otherData[x, y, 2])
                    dGray = int(round((0.2126*dr) + int(0.7152*dg) + int(0.0722*db)))

                    if (dGray < -threshold):
                        newColor = data[x, y, 0] + step
                        newColor = 255 if newColor > 255 else newColor
                        newColor = 0 if newColor < 0 else newColor
                        data[x, y, 0] = newColor
                        data[x, y, 1] = newColor
                        data[x, y, 2] = newColor