from PIL import Image
import numpy as np
import cmath
import sys

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
    
    def calculateHomography(self, srcPoints, dstPoints):
        A = np.zeros((8, 8))
        b = np.zeros(8)
        for i in range(4):
            xSrc, ySrc = srcPoints[i]
            xDst, yDst = dstPoints[i]
            A[2 * i] = [xSrc, ySrc, 1, 0, 0, 0, -xSrc * xDst, -ySrc * xDst]
            A[2 * i + 1] = [0, 0, 0, xSrc, ySrc, 1, -xSrc * yDst, -ySrc * yDst]
            
            b[2 * i] = xDst
            b[2 * i + 1] = yDst
        # Solve using Gaussian elimination
        # This function will solve the system A * x = b
        # You can use Gaussian elimination, LU decomposition, or any other method
        return self.gaussianElimination(A, b)
    
    def gaussianElimination(self, A, b):
        n = len(b)
        for i in range(n):
            # Pivoting
            maxIndex = np.argmax(np.abs(A[i:, i])) + i
            A[[i, maxIndex]] = A[[maxIndex, i]]
            b[i], b[maxIndex] = b[maxIndex], b[i]
            # Normalize the row
            for k in range(i + 1, n):
                factor = A[k, i] / A[i, i]
                b[k] -= factor * b[i]
                A[k, i:] -= factor * A[i, i:]
        # Back substitution
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
        # The last element of the homography matrix (h33) is 1
        homography = np.zeros(9)
        homography[:8] = x
        homography[8] = 1
        
        return homography
    
    def invertHomography(self, H):
        # Calculate the determinant of the 3x3 matrix
        det = (H[0] * (H[4] * H[8] - H[5] * H[7])
        - H[1] * (H[3] * H[8] - H[5] * H[6])
        + H[2] * (H[3] * H[7] - H[4] * H[6]))
        if det == 0:
            raise ValueError("Matrix is not invertible")
        invDet = 1.0 / det
        
        # Calculate the inverse using the cofactor matrix
        invH = np.zeros(9)
        invH[0] = invDet * (H[4] * H[8] - H[5] * H[7])
        invH[1] = invDet * (H[2] * H[7] - H[1] * H[8])
        invH[2] = invDet * (H[1] * H[5] - H[2] * H[4])
        invH[3] = invDet * (H[5] * H[6] - H[3] * H[8])
        invH[4] = invDet * (H[0] * H[8] - H[2] * H[6])
        invH[5] = invDet * (H[2] * H[3] - H[0] * H[5])
        invH[6] = invDet * (H[3] * H[7] - H[4] * H[6])
        invH[7] = invDet * (H[1] * H[6] - H[0] * H[7])
        invH[8] = invDet * (H[0] * H[4] - H[1] * H[3])
        
        return invH
    
    def applyHomographyToPoint(self, H, x, y):
        # Homogeneous coordinates calculation after transformation
        xh = H[0] * x + H[1] * y + H[2]
        yh = H[3] * x + H[4] * y + H[5]
        w = H[6] * x + H[7] * y + H[8]
        # Normalize by w to get the Cartesian coordinates in the destination image
        xPrime = xh / w
        yPrime = yh / w
        return np.array([xPrime, yPrime])
    
    def applyHomography(self, H):
        global data
        data_temp = np.zeros([height, width, 3])
        data_temp = np.copy(data)
        invH = self.invertHomography(H)
        for y in range(height):
            for x in range(width):
                # Apply the inverse of the homography to find the corresponding source pixel
                sourcePoint = self.applyHomographyToPoint(invH, x, y)
                srcX = int(round(sourcePoint[0]))
                srcY = int(round(sourcePoint[1]))
                # Check if the calculated source coordinates are within the source image bounds

                if 0 <= srcX < width and 0 <= srcY < height:
                    # Copy the pixel from the source image to the destination image
                    data_temp[y, x] = data[srcY, srcX]
                else:
                    # If out of bounds, set the destination pixel to a default color
                    data_temp[y, x] = [0, 0, 0]

        # Copy the processed image back to the original image
        data = np.copy(data_temp)