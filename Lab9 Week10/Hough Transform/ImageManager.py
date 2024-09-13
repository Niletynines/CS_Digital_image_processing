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
                
    def houghTransform(self, percent):
        global data
        #The image should be converted to edge map first

        #Work out how the hough space is quantized
        numOfTheta = 720
        thetaStep = math.pi / numOfTheta

        highestR = int(round(max(width, height) * math.sqrt(2)))

        centreX = int(width / 2)
        centreY = int(height / 2)

        print("Hough array w: %s height: %s" % (numOfTheta, (2*highestR)))

        #Create the hough array and initialize to zero
        houghArray = np.zeros([numOfTheta, 2*highestR])

        #Step 1 - find each edge pixel
        #Find edge points and vote in array
        for y in range(3, height - 3):
            for x in range(3, width - 3):
                pointColor = data[x, y, 0]
                if (pointColor != 0):
                #Edge pixel found
                    for i in range(numOfTheta):
                        #Step 2 - Apply the line equation and update hough array
                        #Work out the r values for each theta step
                        r = int((x - centreX) * math.cos(i * thetaStep) + (y - centreY) * math.sin(i * thetaStep))

                        #Move all values into positive range for display purposes
                        r = r + highestR
                        if (r < 0 or r >= 2 * highestR):
                            continue

                        #Increment hough array
                        houghArray[i, r] = houghArray[i, r] + 1

        #Step 3 - Apply threshold to hough array to find line
        #Find the max hough value for the thresholding operation
        maxHough = np.amax(houghArray)

        #Set the threshold limit
        threshold = percent * maxHough
        #Step 4 - Draw lines

        # Search for local peaks above threshold to draw
        for i in range(numOfTheta):
            for j in range(2 * highestR):
                #only consider points above threshold
                if (houghArray[i, j] >= threshold):
                # see if local maxima
                    draw = True
                    peak = houghArray[i, j]

                    for k in range(-1, 2):
                        for l in range(-1, 2):
                        #not seeing itself
                            if (k == 0 and l == 0):
                                continue

                            testTheta = i + k
                            testOffset = j + l

                            if (testOffset < 0 or testOffset >= 2*highestR):
                                continue
                            if (testTheta < 0):
                                testTheta = testTheta + numOfTheta
                            if (testTheta >= numOfTheta):
                                testTheta = testTheta - numOfTheta
                            if (houghArray[testTheta][testOffset] > peak):
                                #found bigger point
                                draw = False
                                break

                    #point found is not local maxima
                    if (not(draw)):
                        continue

                    #if local maxima, draw red back
                    tsin = math.sin(i*thetaStep)
                    tcos = math.cos(i*thetaStep)

                    if (i <= numOfTheta / 4 or i >= (3 * numOfTheta) / 4):
                        for y in range(height):
                            #vertical line
                            x = int((((j - highestR) - ((y - centreY) * tsin)) / tcos) + centreX)

                            if(x < width and x >= 0):
                                data[x, y, 0] = 255
                                data[x, y, 1] = 0
                                data[x, y, 2] = 0
                    else:
                        for x in range(width):
                            #horizontal line
                            y = int((((j - highestR) - ((x - centreX) * tcos)) / tsin) + centreY)

                            if(y < height and y >= 0):
                                data[x, y, 0] = 255
                                data[x, y, 1] = 0
                                data[x, y, 2] = 0
                                
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
    
    # cannyEdgeDetector    
    def cannyEdgeDetector(self, lower, upper):
        global data
        #Step 1 - Apply 5 x 5 Gaussian filter
        gaussian = [2.0 / 159.0, 4.0 / 159.0, 5.0 / 159.0, 4.0 / 159.0, 2.0 / 159.0,
        4.0 / 159.0, 9.0 / 159.0, 12.0 / 159.0, 9.0 / 159.0, 4.0 / 159.0,
        5.0 / 159.0, 12.0 / 159.0, 15.0 / 159.0, 12.0 / 159.0, 5.0 / 159.0,
        4.0 / 159.0, 9.0 / 159.0, 12.0 / 159.0, 9.0 / 159.0, 4.0 / 159.0,
        2.0 / 159.0, 4.0 / 159.0, 5.0 / 159.0, 4.0 / 159.0, 2.0 / 159.0]
        self.linearSpatialFilter(gaussian, 5)
        self.convertToGrayscale()

        #Step 2 - Find intensity gradient
        sobelX = [ 1, 0, -1,
                    2, 0, -2,
                    1, 0, -1]
        sobelY = [ 1, 2, 1,
                0, 0, 0,
                -1, -2, -1]
        magnitude = np.zeros([width, height])
        direction = np.zeros([width, height])
        data_zeropaded = np.zeros([width + 2, height + 2, 3])
        data_zeropaded[1:width + 1, 1:height + 1, :] = data
        for y in range(1, height + 1):
            for x in range(1, width + 1):
                gx = 0
                gy = 0
                subData = data_zeropaded[x - 1:x + 2, y - 1:y + 2, :]
                gx = np.sum(np.multiply(subData[:,:,0:1].flatten(), sobelX))
                gy = np.sum(np.multiply(subData[:,:,0:1].flatten(), sobelY))
                magnitude[x - 1, y - 1] = math.sqrt(gx * gx + gy * gy)
                direction[x - 1, y - 1] = math.atan2(gy, gx) * 180 / math.pi

        #Step 3 - Nonmaxima Suppression
        gn = np.zeros([width, height])
        for y in range(1, height + 1):
            for x in range(1, width + 1):
                targetX = 0
                targetY = 0
                #find closest direction
                if (direction[x - 1, y - 1] <= -157.5):
                    targetX = 1
                    targetY = 0
                elif (direction[x - 1, y - 1] <= -112.5):
                    targetX = 1
                    targetY = -1
                elif (direction[x - 1, y - 1] <= -67.5):
                    targetX = 0
                    targetY = 1
                elif (direction[x - 1, y - 1] <= -22.5):
                    targetX = 1
                    targetY = 1
                elif (direction[x - 1, y - 1] <= 22.5):
                    targetX = 1
                    targetY = 0
                elif (direction[x - 1, y - 1] <= 67.5):
                    targetX = 1
                    targetY = -1
                elif (direction[x - 1, y - 1] <= 112.5):
                    targetX = 0
                    targetY = 1
                elif (direction[x - 1, y - 1] <= 157.5):
                    targetX = 1
                    targetY = 1
                else:
                    targetX = 1
                    targetY = 0
                if (y + targetY >= 0 and y + targetY < height and x + targetX >= 0 and x + targetX < width and magnitude[x - 1, y - 1] < magnitude[x + targetY - 1, y + targetX - 1]):
                    gn[x - 1, y - 1] = 0

                elif (y - targetY >= 0 and y - targetY < height and x - targetX >= 0 and x - targetX < width and magnitude[x - 1, y - 1] < magnitude[x - targetY - 1, y - targetX - 1]):
                    gn[x - 1, y - 1] = 0
                else:
                    gn[x - 1, y - 1] = magnitude[x - 1, y - 1]
                #set back first
                gn[x - 1, y - 1] = 255 if gn[x - 1, y - 1] > 255 else gn[x - 1, y - 1]
                gn[x - 1, y - 1] = 0 if gn[x - 1, y - 1] < 0 else gn[x - 1, y - 1]
                data[x - 1, y - 1, 0] = gn[x - 1, y - 1]
                data[x - 1, y - 1, 1] = gn[x - 1, y - 1]
                data[x - 1, y - 1, 2] = gn[x - 1, y - 1]

        #upper threshold checking with recursive
        for y in range(height):
            for x in range(width):
                if (data[x, y, 0] >= upper):
                    data[x, y, 0] = 255
                    data[x, y, 1] = 255
                    data[x, y, 2] = 255
                    self.hystConnect(x, y, lower)
        #clear unwanted values
        for y in range(height):
            for x in range(width):
                if (data[x, y, 0] != 255):
                    data[x, y, 0] = 0
                    data[x, y, 1] = 0
                    data[x, y, 2] = 0
                    
    def hystConnect(self, x, y, threshold):
        global data
        for i in range(y - 1, y + 2):
            for j in range(x - 1, x + 2):
                if ((j < width) and (i < height) and
                    (j >= 0) and (i >= 0) and
                    (j != x) and (i != y)):
                    value = data[j, i, 0]
                    if (value != 255):
                        if (value >= threshold):
                            data[j, i, 0] = 255
                            data[j, i, 1] = 255
                            data[j, i, 2] = 255
                            self.hystConnect(j, i, threshold)
                        else:
                            data[j, i, 0] = 0
                            data[j, i, 1] = 0
                            data[j, i, 2] = 0