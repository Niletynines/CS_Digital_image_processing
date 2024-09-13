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
    
    # Gray Scale
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
    
    # Harris-Stephens corner detection algorithm
    def detectHarrisFeatures(self, strongest):
        global data
        # Convert to grayscale
        self.convertToGrayscale()
        # Compute gradients Ix and Iy, drop the border
        Ix = np.zeros((height, width), dtype=np.float32)
        Iy = np.zeros((height, width), dtype=np.float32)
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                Ix[y, x] = (float(data[x + 1, y, 0]) - float(data[x - 1, y, 0])) / 2.0
                Iy[y, x] = (float(data[x, y + 1, 0]) - float(data[x, y - 1, 0])) / 2.0
                # Initialize matrices to store products of gradients
        Ix2 = Ix ** 2
        Iy2 = Iy ** 2
        Ixy = Ix * Iy
        
        # Apply 3x3 Gaussian smoothing
        gaussian = np.array([
        [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
        [2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0],
        [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0]
        ])
        
        Sx2 = np.zeros((height, width), dtype=np.float32)
        Sy2 = np.zeros((height, width), dtype=np.float32)
        Sxy = np.zeros((height, width), dtype=np.float32)
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                Sx2[y, x] = np.sum(Ix2[y - 1:y + 2, x - 1:x + 2] * gaussian)
                Sy2[y, x] = np.sum(Iy2[y - 1:y + 2, x - 1:x + 2] * gaussian)
                Sxy[y, x] = np.sum(Ixy[y - 1:y + 2, x - 1:x + 2] * gaussian)
        
        # Compute the corner response function R
        corners = np.zeros((height, width))
        for y in range(height):
            for x in range(width):
                det = Sx2[y, x] * Sy2[y, x] - Sxy[y, x] * Sxy[y, x]
                trace = Sx2[y, x] + Sy2[y, x]
                corners[y, x] = det - 0.04 * (trace ** 2)
                
        cornerPoints = []
        cornerValues = []
        # Maxima Suppression
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if corners[y, x] < 0:
                    continue
                peak = corners[y, x]
                isMaxima = True
                
                # Check 3x3 neighborhood
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        if k == 0 and l == 0:
                            continue
                        if corners[y + k, x + l] > peak:
                            isMaxima = False
                            break
                    if not isMaxima:
                        break
                if isMaxima:
                    insertPos = 0
                    while insertPos < len(cornerValues) and cornerValues[insertPos] > peak:
                        insertPos += 1
                    
                    cornerPoints.insert(insertPos, (x, y))
                    cornerValues.insert(insertPos, peak)
                    
                    if len(cornerPoints) > strongest:
                        cornerPoints.pop()
                        cornerValues.pop()

        # Draw red X on the image at the corner points
        for p in cornerPoints:
            data[p[0], p[1], 0] = 255
            data[p[0], p[1], 1] = 0
            data[p[0], p[1], 2] = 0

            data[p[0] + 1, p[1] + 1, 0] = 255
            data[p[0] + 1, p[1] + 1, 1] = 0
            data[p[0] + 1, p[1] + 1, 2] = 0

            data[p[0] + 1, p[1] - 1, 0] = 255
            data[p[0] + 1, p[1] - 1, 1] = 0
            data[p[0] + 1, p[1] - 1, 2] = 0

            data[p[0] - 1, p[1] + 1, 0] = 255
            data[p[0] - 1, p[1] + 1, 1] = 0
            data[p[0] - 1, p[1] + 1, 2] = 0

            data[p[0] - 1, p[1] - 1, 0] = 255
            data[p[0] - 1, p[1] - 1, 1] = 0
            data[p[0] - 1, p[1] - 1, 2] = 0
        return cornerPoints