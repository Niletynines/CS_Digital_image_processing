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

    # Averaging filter
    def averagingFilter(self, size):
        global data
        if (size % 2 == 0):
            print("Size Invalid: must be odd number!")
            return
        
        data_zeropaded = np.zeros([width + int(size/2) * 2, height + int(size/2) * 2, 3], dtype=np.uint8)
        data_zeropaded[int(size/2):width + int(size/2), int(size/2):height + int(size/2), :] = data
        
        for y in range(int(size/2), int(size/2) + height):
            for x in range(int(size/2), int(size/2) + width):
                subData = data_zeropaded[x - int(size/2):x + int(size/2) + 1, y - int(size/2):y + int(size/2) + 1, :]
                avgRed = np.mean(subData[:,:,0:1])
                avgRed = 255 if avgRed > 255 else avgRed
                avgRed = 0 if avgRed < 0 else avgRed
                avgGreen = np.mean(subData[:,:,1:2])
                avgGreen = 255 if avgGreen > 255 else avgGreen
                avgGreen = 0 if avgGreen < 0 else avgGreen
                avgBlue = np.mean(subData[:,:,2:3])               
                avgBlue = 255 if avgBlue > 255 else avgBlue
                avgBlue = 0 if avgBlue < 0 else avgBlue
                data[x - int(size/2), y - int(size/2), 0] = avgRed
                data[x - int(size/2), y - int(size/2), 1] = avgGreen
                data[x - int(size/2), y - int(size/2), 2] = avgBlue
                
    # Unsharp Masking   
    def unsharpMasking(self, size, k):
        global data
        global original
        
        if size not in [3, 7, 15]:
            print("Size Invalid: Only 3x3, 7x7, or 15x15 are allowed!")
            return
        
        OGdata = np.copy(original)   
        
        filteredData = np.copy(data)
        self.averagingFilter(size)          
        filteredData = data      
       
        detail_mask = OGdata - filteredData
        
        sharpened = OGdata + (k * detail_mask)
        
        sharpened = np.clip(sharpened, 0, 255)
        
        data = sharpened.astype(np.uint8)
        

        
    