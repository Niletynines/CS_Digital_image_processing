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

    # Gray Scale Histogram
    def getGrayscaleHistogram(self):
        self.convertToGrayscale()
        histogram = np.array([0] * 256)

        for y in range(height):
            for x in range(width):
                histogram[data[x, y, 0]] += 1

        self.restoreToOriginal()
        return histogram
    
    def restoreToOriginal(self):
        global data
        data = np.copy(original)

    # Contrast
    def getContrast(self):
        contrast = 0.0
        histogram = self.getGrayscaleHistogram()
        avgIntensity = 0.0
        pixelNum = width * height

        for i in range(len(histogram)):
            avgIntensity += histogram[i] * i

        avgIntensity /= pixelNum

        for y in range(height):
            for x in range(width):
                contrast += (data[x, y, 0] - avgIntensity) ** 2

        contrast = (contrast / pixelNum) ** 0.5

        return contrast

    def adjustContrast(self, contrast):
        global data
        currentContrast = self.getContrast()

        histogram = self.getGrayscaleHistogram()
        avgIntensity = 0.0
        pixelNum = width * height

        for i in range(len(histogram)):
            avgIntensity += histogram[i] * i

        avgIntensity /= pixelNum
        min = avgIntensity - currentContrast
        max = avgIntensity + currentContrast
        newMin = avgIntensity - currentContrast - contrast / 2
        newMax = avgIntensity + currentContrast + contrast / 2
        newMin = 0 if newMin < 0 else newMin
        newMax = 0 if newMax < 0 else newMax
        newMin = 255 if newMin > 255 else newMin
        newMax = 255 if newMax > 255 else newMax

        if (newMin > newMax):
            temp = newMax
            newMax = newMin
            newMin = temp

        contrastFactor = (newMax - newMin) / (max - min)

        for y in range(height):
            for x in range(width):
                r = data[x, y, 0]
                g = data[x, y, 1]
                b = data[x, y, 2]
                contrast += (data[x, y, 0] - avgIntensity) ** 2
                r = (int)((r - min) * contrastFactor + newMin)
                r = 255 if r > 255 else r
                r = 0 if r < 0 else r

                g = (int)((g - min) * contrastFactor + newMin)
                g = 255 if g > 255 else g
                g = 0 if g < 0 else g
                b = (int)((b - min) * contrastFactor + newMin)
                b = 255 if b > 255 else b
                b = 0 if b < 0 else b

                data[x, y, 0] = r
                data[x, y, 1] = g
                data[x, y, 2] = b