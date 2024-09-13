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
    
    # Comvert Color in Picture
    def convertColor(self):
        global data
        for y in range(height):
            for x in range(width):
                
                # Get the RGB values of the current pixel
                r, g, b = data[x, y]
                
                # Set Hair
                if x<=185 and r>=0 and r<=143 and g >=147 and g <= 260 and b>=0 and b<=137:
                    # Convet Hair to Gray
                    r = data[x, y, 0]
                    g = data[x, y, 1]
                    b = data[x, y, 2] 
                    gray = (int)((0.2126*r) + int(0.7152*g) + int(0.0722*b))        
                    data[x, y, 0] = gray  # Red
                    data[x, y, 1] = gray  # Green
                    data[x, y, 2] = gray  # Blue
                    
                # Set Skin
                if r>=0 and r<=180 and g>=73 and g<=253 and b>=18 and b<=229 :
                    #1Sqr
                    if x>=207 and x<=223 and y>=190 and y<=209:
                        continue
                    #2Sqr
                    if x>=207 and x<=226 and y>=350 and y<=368:
                        continue  
                    #3Sqr
                    if x>=222 and x<=242 and y>=173 and y<=193:
                        continue  
                    #4Sqr
                    if x>=221 and x<=242 and y>=334 and y<=353:
                        continue  
                    #5Sqr
                    if x>=238 and x<=256 and y>=159 and y<=177:
                        continue  
                    #6Sqr
                    if x>=238 and x<=257 and y>=319 and y<=337:
                        continue  
                    
                    if r<b and r!=0 and g!=0 :
                        #Convert Skin to Light Brown
                        data[x,y,0] = 215 # Red
                        data[x,y,1] = 181 # Green
                        data[x,y,2] = 133 # Blue
                        
                # Set Beard
                if y>=90 and y<=420 and x>=275 and x<=410 and r>=0 and r<=178 and g>=60 and g<=215 and b>=0 and b<=85 :
                     if r>b or g>b :                       
                        # Convert Beard to Red
                        data[x,y,0] = 255 # Red
                        data[x,y,1] = 0 # Green
                        data[x,y,2] = 0 # Blue
                        
                # Set Sun Glasses
                if x>=89 and x<=280 and r>=0 and r<=64 and g>=0 and g<=130 and b>=0 and b<=116:
                    if x>=189 and x<=272 and y>=96 and y<=416 : 
                        data[x,y,0] = 0 # Red
                        data[x,y,1] = 0 # Green
                        data[x,y,2] = 0 # Blue
                    
                # Set T-Shirt
                if y>=38 and y<=480 and x>=426 and r>=0 and r<=150 and g>=60 and g<=225 and b>=0 and b<=76 :
                    if b>r or g>r:
                        #Convert T-Shirt to Blue
                        data[x,y,0] = 0 # Red
                        data[x,y,1] = 0 # Green
                        data[x,y,2] = 255 # Blue
                    
                # Set BG
                if r >= 86 and r <= 240 and g >= 99 and g <= 252 and b >= 0 and b <= 198:
                    #1Sqr
                    if x>=207 and x<=223 and y>=190 and y<=209:
                        continue
                    #2Sqr
                    if x>=207 and x<=226 and y>=350 and y<=368:
                        continue  
                    #3Sqr
                    if x>=222 and x<=242 and y>=173 and y<=193:
                        continue  
                    #4Sqr
                    if x>=221 and x<=242 and y>=334 and y<=353:
                        continue  
                    #5Sqr
                    if x>=238 and x<=256 and y>=159 and y<=177:
                        continue  
                    #6Sqr
                    if x>=238 and x<=257 and y>=319 and y<=337:
                        continue
                    
                    if g>r and r>b and g!=0 and b!=0 and r!=255:
                        #Convert BG To BlackGray
                        data[x,y,0] = 30 # Red
                        data[x,y,1] = 30 # Green
                        data[x,y,2] = 30 # Blue
                    
    # Remove Green
    def RemoveGreen(self):
        global data       
        for y in range(height):
            for x in range(width):
                r,g,b = data[x,y]
                if r<g or r<=b and b<g  and b==0:
                    if x>=206 and x<=256 and y>=158 and y<=208:
                        continue
                    if x>=206 and x<=257 and y>=320 and y<=368:
                        continue
                    if (y-1<0):
                        continue
                        
                    data[x,y,0] = data[x,y-1,0] # Red
                    data[x,y,1] = data[x,y-1,1] # Green
                    data[x,y,2] = data[x,y-1,2] # Blue
                    
    # Gray Scale
    def convertToGrayscale(self):
        global data
        for y in range(height):
            for x in range(width):
                r = data[x, y, 0]
                g = data[x, y, 1]
                b = data[x, y, 2]
                
                gray = (int)((0.2126*r) + int(0.7152*g) + int(0.0722*b))
                
                data[x, y, 0] = gray  # Set red channel to gray
                data[x, y, 1] = gray  # Set green channel to gray
                data[x, y, 2] = gray  # Set blue channel to gray
                
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
    
    # Write Histogram To CSV
    def writeHistogramToCSV(self, histogram, fileName):
        histogram.tofile(fileName,sep=',',format='%s')

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
    
    # Contrast
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
                
    # Median Filter       
    def medianFilter(self, size):
        global data

        if (size % 2 == 0):
            print("Size Invalid: must be odd number!")
            return

        paddedData = np.zeros([width + size - 1, height + size - 1, 3], dtype=np.uint8)
        paddedData[int((size-1)/2):width + int((size-1)/2), int((size-1)/2):height + int((size-1)/2), :] = data

        output = np.zeros_like(data)

        for y in range(height):
            for x in range(width):
                subData = paddedData[x:x + size, y:y + size, :]

                medRed = np.median(subData[:, :, 0])
                medGreen = np.median(subData[:, :, 1])
                medBlue = np.median(subData[:, :, 2])

                output[x, y, 0] = int(medRed)
                output[x, y, 1] = int(medGreen)
                output[x, y, 2] = int(medBlue)

        data = output