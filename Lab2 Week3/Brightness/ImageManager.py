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
    
    # Brightness
    def adjustBrightness(self, brightness):
        global data
        for y in range(height):
            for x in range(width):
                r = data[x, y, 0]
                g = data[x, y, 1]
                b = data[x, y, 2]
                
                r = r + brightness
                r = 255 if r > 255 else r
                r = 0 if r < 0 else r

                g = g + brightness
                g = 255 if g > 255 else g
                g = 0 if g < 0 else g
                
                b = b + brightness
                b = 255 if b > 255 else b
                b = 0 if b < 0 else b
                
                data[x, y, 0] = r
                data[x, y, 1] = g
                data[x, y, 2] = b
                
    def restoreToOriginal(self):
        global data
        data = np.copy(original)