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
   
    # Gray Scale
    def convertToGrayscale(self):
        global data
        for y in range(height):
            for x in range(width):
                data[x, y, 0] = int(0.2989 * data[x, y, 0] + 0.5870 * data[x, y, 1] + 0.1140 * data[x, y, 2])
                data[x, y, 1] = int(0.2989 * data[x, y, 0] + 0.5870 * data[x, y, 1] + 0.1140 * data[x, y, 2])
                data[x, y, 2] = int(0.2989 * data[x, y, 0] + 0.5870 * data[x, y, 1] + 0.1140 * data[x, y, 2])

    def restoreToOriginal(self):
        global data
        data = np.copy(original)

    # Histogram
    def getGrayscaleHistogram(self):
        self.convertToGrayscale()
        histogram = np.array([0] * 256)

        for y in range(height):
            for x in range(width):
                histogram[data[x, y, 0]] += 1

        self.restoreToOriginal()
        return histogram

    def writeHistogramToCSV(self, histogram, fileName):
        histogram.tofile(fileName,sep=',',format='%s')