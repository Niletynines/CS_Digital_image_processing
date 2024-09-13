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
    
    # median nonlinear filtering
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