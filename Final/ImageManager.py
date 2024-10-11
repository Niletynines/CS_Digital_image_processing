from PIL import Image
import numpy as np
import cmath
import sys
import matplotlib.pyplot as plt

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
    
    def crop_image_array(self, start_row, end_row, start_col, end_col):

        global data

        data_temp = []
        cropped_image_array = []
    
        # วนลูปเพื่อเลือกแถวที่ต้องการ
        for row in range(start_row, end_row):
            row_data = []

            # วนลูปเพื่อเลือกคอลัมน์ที่ต้องการ
            for col in range(start_col, end_col):
                row_data.append(data[row, col])  # เก็บค่าพิกเซลแต่ละจุด

            cropped_image_array.append(row_data)
        
        # แปลงกลับเป็น NumPy array
        data_temp = np.array(cropped_image_array)

        return data_temp
    
    def replace_image_section(self, replacement_array, start_row, start_col):
        
        # แทนที่ส่วนหนึ่งของ image_array ด้วย replacement_array ตามตำแหน่งที่กำหนด
        # image_array: อาร์เรย์ภาพต้นฉบับ
        # replacement_array: อาร์เรย์ภาพที่ตัดมา
        # start_row: แถวเริ่มต้นที่จะวาง replacement_array
        # start_col: คอลัมน์เริ่มต้นที่จะวาง replacement_array
        
        end_row = start_row + replacement_array.shape[0]
        end_col = start_col + replacement_array.shape[1]

        # ตรวจสอบให้แน่ใจว่าขนาดเหมาะสม
        data[start_row:end_row, start_col:end_col] = replacement_array
        # return image_array

            
    def calculateHomography(self, srcPoints, dstPoints):
        # Rotate dstPoints 90 degrees 
        # tempDst = np.copy(dstPoints)
        # dstPoints[0, :] = tempDst[3, :]
        # dstPoints[1, :] = tempDst[0, :]
        # dstPoints[2, :] = tempDst[1, :]
        # dstPoints[3, :] = tempDst[2, :]

        A = np.zeros((8, 8))
        b = np.zeros(8)
        for i in range(4):
            xSrc, ySrc = srcPoints[i]
            xDst, yDst = dstPoints[i]
            A[2 * i ] = [xSrc, ySrc, 1, 0, 0, 0, -xSrc * xDst, -ySrc * xDst]
            A[2 * i + 1] = [0, 0, 0, xSrc, ySrc, 1, -xSrc * yDst, -ySrc * yDst]
            
            b[2 * i] = xDst
            b[2 * i + 1] = yDst
        # Solve using Gaussian elimination
        # This function will solve the system A * x = b
        # You can use Gaussian elimination, LU decomposition, or any other method
        # homography = np.linalg.lstsq(A, b, rcond=None)[0]
        # homography = np.append(homography, 1)  # last element h33 is 1
        # return homography
    
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
        - H[1] * (H[3] * H[8] - H[6] * H[5])
        + H[2] * (H[3] * H[7] - H[6] * H[4]))
        if det == 0:
            raise ValueError("Matrix is not invertible")
        invDet = 1.0 / det
        
        # Calculate Minor
        
        Minor = np.zeros(9)
        Minor[0] = (H[4]*H[8])-(H[7]*H[5])
        Minor[1] = (H[3]*H[8])-(H[6]*H[5])
        Minor[2] = (H[3]*H[7])-(H[6]*H[4])
        Minor[3] = (H[1]*H[8])-(H[7]*H[2])
        Minor[4] = (H[0]*H[8])-(H[6]*H[2])
        Minor[5] = (H[0]*H[8])-(H[6]*H[1])
        Minor[6] = (H[1]*H[5])-(H[4]*H[2])
        Minor[7] = (H[0]*H[5])-(H[6]*H[2])
        Minor[8] = (H[0]*H[4])-(H[3]*H[1])
        
        #Find Cofactor
        
        Cof = np.zeros(9)
        for i in range(9) :
            if i%2==0 :
                Cof[i] = Minor[i]
            else :
                Cof[i] = -1*Minor[i]
                
        #Find Adjoint
        
        Adj = np.zeros(9)
        for i in range(9):
            Adj[i] = Cof[i]
        
        Adj[1] = Cof[3]
        Adj[2] = Cof[6]
        Adj[3] = Cof[1]
        Adj[5] = Cof[7]
        Adj[6] = Cof[2]
        Adj[7] = Cof[5]
        
        # # Calculate the inverse using the cofactor matrix
        invH = np.zeros(9)
        invH[0] = invDet * (H[4] * H[8] - H[5] * H[7]) #11
        invH[1] = invDet * (H[2] * H[7] - H[1] * H[8]) #21
        invH[2] = invDet * (H[1] * H[5] - H[2] * H[4]) #31
        invH[3] = invDet * (H[5] * H[6] - H[3] * H[8]) #21
        invH[4] = invDet * (H[0] * H[8] - H[2] * H[6]) #22
        invH[5] = invDet * (H[2] * H[3] - H[0] * H[5]) #23
        invH[6] = invDet * (H[3] * H[7] - H[4] * H[6]) #31
        invH[7] = invDet * (H[1] * H[6] - H[0] * H[7]) #32
        invH[8] = invDet * (H[0] * H[4] - H[1] * H[3]) #33
        
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

                if 0 <= srcY < width and 0 <= srcX < height:
                    # Copy the pixel from the source image to the destination image
                    data_temp[x, y] = data[srcY, srcX]
                else:
                    # If out of bounds, set the destination pixel to a default color
                    data_temp[x, y] = [0, 0, 0]

        # Copy the processed image back to the original image
        data = np.copy(data_temp)

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
                
    def otsuThreshold(self):
        global data
        # global variance
        self.convertToGrayscale()
        histogram = np.zeros(256)
        for y in range(height):
            for x in range(width):
                histogram[data[x, y, 0]] += 1
        histogramNorm = np.zeros(len(histogram))
        pixelNum = width * height
        for i in range(len(histogramNorm)):
            histogramNorm[i] = histogram[i] / pixelNum
        histogramCS = np.zeros(len(histogram))
        histogramMean = np.zeros(len(histogram))
        for i in range(len(histogramNorm)):
            if (i == 0):
                histogramCS[i] = histogramNorm[i]
                histogramMean[i] = 0
            else:
                histogramCS[i] = histogramCS[i - 1] + histogramNorm[i]
                histogramMean[i] = histogramMean[i - 1] + histogramNorm[i] * i

        globalMean = histogramMean[len(histogramMean) - 1]
        max = sys.float_info.min
        maxVariance = sys.float_info.min
        variance = maxVariance
        countMax = 0
        for i in range(len(histogramCS)):
            if (histogramCS[i] < 1 and histogramCS[i] > 0):
                variance = ((globalMean * histogramCS[i] - histogramMean[i]) ** 2) / (histogramCS[i] * (1 - histogramCS[i]))
            if (variance > maxVariance):
                maxVariance = variance
                max = i
                countMax = 1
            elif (variance == maxVariance):
                countMax = countMax + 1
                max = ((max * (countMax - 1)) + i) / countMax
        self.thresholding(round(max))

    def find_bounding_boxes(self, CropImage, Croped_width):
        # global data
        Croped_width = Croped_width
        bounding_boxes = []
        in_character = False
        start_col = 0

        # วนลูปเพื่อเช็คทีละคอลัมน์ (แนวตั้ง)
        for y in range(Croped_width):
            # เช็คว่ามีตัวอักษรในคอลัมน์นี้หรือไม่ (มีพิกเซลสีขาว)
            if np.any(CropImage[:, y] == 0):
                if not in_character:
                    start_col = y  # เริ่มตัวอักษร
                    in_character = True
            else:
                if in_character:
                    # สิ้นสุดตัวอักษรแล้ว (เจอช่องว่าง)
                    bounding_boxes.append((start_col, y))
                    in_character = False

        # ในกรณีที่ตัวอักษรอยู่ถึงคอลัมน์สุดท้าย
        if in_character:
            bounding_boxes.append((start_col, width))

        # แสดงข้อมูล bounding boxes ที่พบ
        for i, (start, end) in enumerate(bounding_boxes):
            print(f"Bounding Box {i+1}: Start = {start}, End = {end}")

        return bounding_boxes
    
    def find_first_black_pixel(self, image_array):

        rows, cols = np.where(image_array == 0)
        if len(rows) > 0:
            return rows[0], cols[0]  # Return the first black pixel position
        return None  # If no black pixel is found

    def align_images(self, cropped_image_array, micr_array):

        # ค้นหาจุดสีดำแรกในทั้งสองภาพ
        first_black_cropped = self.find_first_black_pixel(cropped_image_array)
        first_black_micr = self.find_first_black_pixel(micr_array)

        if first_black_cropped and first_black_micr:
            # คำนวณการเลื่อนที่จำเป็น
            col_shift = first_black_micr[1] - first_black_cropped[1]

            # ถ้าการเลื่อนต้องการ (col_shift != 0)
            if col_shift != 0:
                # สร้างสำเนาของภาพเพื่อทำการเลื่อนคอลัมน์
                aligned_image = np.copy(cropped_image_array)
                num_cols = cropped_image_array.shape[1]

                # เลื่อนคอลัมน์ทั้งภาพ
                for row in range(cropped_image_array.shape[0]):
                    # เลื่อนคอลัมน์ไปทางซ้าย (col_shift < 0)
                    if col_shift < 0:
                        aligned_image[row, :] = np.roll(cropped_image_array[row, :], col_shift)
                        # เติมคอลัมน์ทางขวาที่เลื่อนออกจากขอบด้วยสีขาว
                        aligned_image[row, col_shift:] = 255
                    # เลื่อนคอลัมน์ไปทางขวา (col_shift > 0)
                    elif col_shift > 0:
                        aligned_image[row, :] = np.roll(cropped_image_array[row, :], col_shift)
                        # เติมคอลัมน์ทางซ้ายที่เลื่อนออกจากขอบด้วยสีขาว
                        aligned_image[row, :col_shift] = 255
                    for col in range (5,7,1):
                        aligned_image[row,col] = cropped_image_array[row,col]
                return aligned_image

        return cropped_image_array

    def compare_characters(self, cropped_image_array, micr_array):
        # ตรวจสอบขนาดของภาพย่อยและภาพ MICR ว่าตรงกันหรือไม่
        if cropped_image_array.shape != micr_array.shape:
            # ปรับขนาดภาพให้ตรงกับ MICR (7x9) ถ้าไม่ตรงกัน
            resized_image = Image.fromarray(cropped_image_array).resize(micr_array.shape[::-1], Image.NEAREST)
            cropped_image_array = np.array(resized_image)

        # ตรวจสอบว่าภาพที่ตัดมี 3 ช่องสี (RGB) หรือไม่
        if len(cropped_image_array.shape) == 3 and cropped_image_array.shape[2] == 3:
            # แปลงภาพ RGB เป็น grayscale
            cropped_image_array = np.mean(cropped_image_array, axis=2)

        # แปลงภาพ grayscale เป็นไบนารี (0, 255)
        cropped_image_array = np.where(cropped_image_array > 128, 255, 0).astype(np.uint8)

        # จัดตำแหน่งภาพให้จุดสีดำแรกอยู่ตรงกัน
        aligned_cropped_image = self.align_images(cropped_image_array, micr_array)

        # แสดงภาพที่นำมาเทียบกัน
        # self.display_comparison(aligned_cropped_image, micr_array)

        # เปรียบเทียบจำนวนพิกเซลที่ตรงกัน (พิกเซลสีดำที่ตรงกัน)
        match_score = np.sum(aligned_cropped_image == micr_array)

        # คำนวณค่าความคล้ายคลึงในแง่ของเปอร์เซ็นต์
        total_pixels = micr_array.size  # จำนวนพิกเซลทั้งหมด
        similarity = match_score / total_pixels  # ค่าความคล้ายคลึงเป็นสัดส่วน

        return similarity

    # def display_comparison(self, cropped_image_array, micr_array):

    #     # สร้าง subplot สำหรับแสดงภาพเคียงข้างกัน
    #     fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    #     # แสดงภาพที่ถูกตัด
    #     axes[0].imshow(cropped_image_array, cmap='gray')
    #     axes[0].set_title("Cropped Image")
    #     axes[0].axis('off')  # ซ่อนแกนของภาพ

    #     # แสดงภาพ MICR
    #     axes[1].imshow(micr_array, cmap='gray')
    #     axes[1].set_title("MICR Character")
    #     axes[1].axis('off')  # ซ่อนแกนของภาพ

    #     # ปรับแต่ง layout และแสดงผล
    #     plt.tight_layout()
    #     plt.show()
    
    # def display_bounding_boxes(self, image_array, bounding_boxes):
        
    #     num_boxes = len(bounding_boxes)
        
    #     # ตั้งค่าขนาดของ figure ให้เหมาะสมกับจำนวน bounding boxes
    #     fig, axes = plt.subplots(1, num_boxes, figsize=(num_boxes * 3, 3))
        
    #     if num_boxes == 1:
    #         axes = [axes]  # ถ้ามีเพียง bounding box เดียว ให้เปลี่ยน axes เป็น list เพื่อการเข้าถึง

    #     for i, (start_col, end_col) in enumerate(bounding_boxes):
    #         # ตัดภาพย่อยจาก bounding box
    #         cropped_image = image_array[:, start_col:end_col]
            
    #         # แสดงภาพ bounding box
    #         axes[i].imshow(cropped_image, cmap='gray')
    #         axes[i].set_title(f"Box {i+1}: ({start_col}, {end_col})")
    #         axes[i].axis('off')  # ซ่อนแกน

    #     plt.tight_layout()
    #     plt.show()

    def match_micr_characters(self, CropImage, bounding_boxes):
        # ตรวจสอบ bounding boxes ก่อน
        bounding_boxes = self.validate_bounding_boxes(bounding_boxes, CropImage.shape)
        
        # แสดง bounding boxes ที่ผ่านการตรวจสอบแล้ว
        # self.display_bounding_boxes(CropImage, bounding_boxes)
        
        matched_characters = []

        for start_col, end_col in bounding_boxes:
            # ตัดภาพย่อยจาก bounding box
            cropped_image = CropImage[:, start_col:end_col]
            
            # แปลงให้เป็นภาพไบนารี (0, 255)
            cropped_image_array = np.where(cropped_image > 128, 255, 0).astype(np.uint8)
            
            # ป้องกันไม่ให้จุดสีดำเล็กกว่า 1 พิกเซลโดยการทำ dilation
            cropped_image_array = self.dilate_black_pixels(cropped_image_array)
            
            # ปรับขนาดภาพย่อยให้ตรงกับขนาด 7x9 ของ MICR
            resized_image = self.resize_image_to_micr(cropped_image_array)
            cropped_image_array_resized = np.array(resized_image)
            
            # แปลงให้เป็นภาพไบนารีอีกครั้ง (0, 255)
            cropped_image_array_resized = np.where(cropped_image_array_resized > 128, 255, 0).astype(np.uint8)
            
            # เปรียบเทียบกับตัวอักษร MICR ทั้งหมดและค้นหาตัวที่ตรงที่สุด
            best_match = None
            best_score = 0
            for char, micr_array in self.micr_characters.items():
                score = self.compare_characters(cropped_image_array_resized, micr_array)
                if score > best_score:
                    best_score = score
                    best_match = char
            
            matched_characters.append(best_match)
            # พิมพ์ตัวอักษรที่ตรงกันสำหรับแต่ละ bounding box
            print(f"Matched Character for bounding box ({start_col}, {end_col}): {best_match}")
        
        # คืนค่ารายการตัวอักษรที่ตรงกัน
        return matched_characters
    
    def resize_image_to_micr(self,cropped_image_array, target_size=(7, 9)):

        # ตรวจสอบขนาดภาพย่อยก่อน
        if cropped_image_array.shape != (target_size[1], target_size[0]):
            # ใช้ Image.fromarray เพื่อแปลง numpy array เป็นรูปภาพ
            resized_image = Image.fromarray(cropped_image_array).resize(target_size, Image.NEAREST)
            # แปลงกลับเป็น numpy array
            cropped_image_array_resized = np.array(resized_image)
        else:
            # หากภาพมีขนาดตรงกันแล้วก็ใช้ภาพเดิม
            cropped_image_array_resized = cropped_image_array

        return cropped_image_array_resized
        
    def validate_bounding_boxes(self, bounding_boxes, image_shape):
    
        validated_boxes = []
        height, width = image_shape[:2]

        for start_col, end_col in bounding_boxes:
            # ตรวจสอบว่า bounding box มีขนาดที่เหมาะสม
            if end_col - start_col > 0:
                # ตรวจสอบว่าอยู่ภายในขอบเขตของภาพ
                if 0 <= start_col < width and 0 < end_col <= width:
                    validated_boxes.append((start_col, end_col))
                else:
                    print(f"Invalid bounding box: start_col={start_col}, end_col={end_col} is out of image bounds.")
            else:
                print(f"Invalid bounding box: start_col={start_col}, end_col={end_col} is too small.")
        
        return validated_boxes

    def dilate_black_pixels(self, image_array):

        # ตรวจสอบว่าภาพเป็น grayscale หรือไม่
        if len(image_array.shape) == 3:
            # แปลงภาพ RGB เป็น grayscale
            image_array = np.mean(image_array, axis=2).astype(np.uint8)
        
        # Kernel ขนาด 3x3 สำหรับการ dilation (ขยายขนาดจุดสีดำ)
        kernel = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]], dtype=np.uint8)
        
        # สร้างสำเนาใหม่ของ image_array สำหรับ dilation
        dilated_image = np.copy(image_array)
        
        # วนลูปผ่านทุกพิกเซลและทำการ dilation
        for i in range(1, image_array.shape[0] - 1):
            for j in range(1, image_array.shape[1] - 1):
                # ตรวจสอบเฉพาะพิกเซลที่เป็นสีดำ (0)
                if image_array[i, j] == 0:
                    # ขยายขนาดพิกเซลสีดำรอบข้างตาม kernel
                    dilated_image[i-1:i+2, j-1:j+2] = np.minimum(dilated_image[i-1:i+2, j-1:j+2], kernel * 255)
        
        return dilated_image
    
    # สร้างตัวเลขและตัวอักษร MICR ในขนาด 9x7
    micr_characters = {
        '0': np.array([
            [255, 0, 0, 0, 0, 0, 255],
            [0, 255, 255, 255, 255, 255, 0],
            [0, 255, 255, 255, 255, 255, 0],
            [0, 255, 255, 255, 255, 255, 0],
            [0, 255, 255, 255, 255, 255, 0],
            [0, 255, 255, 255, 255, 255, 0],
            [0, 255, 255, 255, 255, 255, 0],
            [0, 255, 255, 255, 255, 255, 0],
            [255, 0, 0, 0, 0, 0, 255]
        ], dtype=np.uint8),

        '1': np.array([
            [255, 255, 255, 0, 0, 255, 255],
            [255, 255, 255, 255, 0, 255, 255],
            [255, 255, 255, 255, 0, 255, 255],
            [255, 255, 255, 255, 0, 255, 255],
            [255, 255, 255, 255, 0, 255, 255],
            [255, 255, 255, 0, 0, 0, 0],
            [255, 255, 255, 0, 0, 0, 0],
            [255, 255, 255, 0, 0, 0, 0],
            [255, 255, 255, 0, 0, 0, 0]
        ], dtype=np.uint8),

        '2': np.array([
            [255, 255, 255, 0, 0, 0, 0],
            [255, 255, 255, 255, 255, 255, 0],
            [255, 255, 255, 255, 255, 255, 0],
            [255, 255, 255, 255, 255, 255, 0],
            [255, 255, 255, 0, 0, 0, 0],
            [255, 255, 255, 0, 255, 255, 255],
            [255, 255, 255, 0, 255, 255, 255],
            [255, 255, 255, 0, 255, 255, 255],
            [255, 255, 255, 0, 0, 0, 0]
        ], dtype=np.uint8),

        '3': np.array([
            [255, 255, 0, 0, 0, 0, 255],
            [255, 255, 255, 255, 255, 0, 255],
            [255, 255, 255, 255, 255, 0, 255],
            [255, 255, 255, 255, 255, 0, 255],
            [255, 255, 0, 0, 0, 0, 255],
            [255, 255, 255, 255, 255, 0, 0],
            [255, 255, 255, 255, 255, 0, 0],
            [255, 255, 255, 255, 255, 0, 0],
            [255, 255, 0, 0, 0, 0, 0]
        ], dtype=np.uint8),

        '4': np.array([
            [255, 0, 0, 255, 255, 255, 255],
            [255, 0, 0, 255, 255, 255, 255],
            [255, 0, 0, 255, 255, 255, 255],
            [255, 0, 0, 255, 255, 255, 255],
            [255, 0, 0, 255, 255, 255, 255],
            [255, 0, 0, 255, 255, 0, 0],
            [255, 0, 0, 0, 0, 0, 0],
            [255, 255, 255, 255, 255, 0, 0],
            [255, 255, 255, 255, 255, 0, 0]
        ], dtype=np.uint8),

        '5': np.array([
            [255, 255, 0, 0, 0, 0, 0],
            [255, 255, 0, 255, 255, 255, 255],
            [255, 255, 0, 255, 255, 255, 255],
            [255, 255, 0, 255, 255, 255, 255],
            [255, 255, 0, 0, 0, 0, 0],
            [255, 255, 255, 255, 255, 255, 0],
            [255, 255, 255, 255, 255, 255, 0],
            [255, 255, 255, 255, 255, 255, 0],
            [255, 255, 0, 0, 0, 0, 0]
        ], dtype=np.uint8),

        '6': np.array([
            [255, 0, 0, 0, 0, 255, 255],
            [255, 0, 255, 255, 0, 255, 255],
            [255, 0, 255, 255, 255, 255, 255],
            [255, 0, 255, 255, 255, 255, 255],
            [255, 0, 255, 255, 255, 255, 255],
            [255, 0, 0, 0, 0, 0, 0],
            [255, 0, 255, 255, 255, 255, 0],
            [255, 0, 255, 255, 255, 255, 0],
            [255, 0, 0, 0, 0, 0, 0]
        ], dtype=np.uint8),

        '7': np.array([
            [255, 255, 0, 0, 0, 0, 0],
            [255, 255, 0, 255, 255, 255, 0],
            [255, 255, 0, 255, 255, 255, 0],
            [255, 255, 255, 255, 255, 255, 0],
            [255, 255, 255, 255, 0, 0, 255],
            [255, 255, 255, 255, 0, 255, 255],
            [255, 255, 255, 255, 0, 255, 255],
            [255, 255, 255, 255, 0, 255, 255],
            [255, 255, 255, 255, 0, 255, 255]
        ], dtype=np.uint8),

        '8': np.array([
            [255, 0, 0, 0, 0, 0, 255],
            [255, 0, 255, 255, 255, 0, 255],
            [255, 0, 255, 255, 255, 0, 255],
            [255, 0, 255, 255, 255, 0, 255],
            [255, 0, 0, 0, 0, 0, 255],
            [0, 0, 255, 255, 255, 0, 0],
            [0, 0, 255, 255, 255, 0, 0],
            [0, 0, 255, 255, 255, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.uint8),

        '9': np.array([
            [255, 0, 0, 0, 0, 0, 0],
            [255, 0, 255, 255, 255, 255, 0],
            [255, 0, 255, 255, 255, 255, 0],
            [255, 0, 255, 255, 255, 255, 0],
            [255, 0, 0, 0, 0, 0, 0],
            [255, 255, 255, 255, 255, 0, 0],
            [255, 255, 255, 255, 255, 0, 0],
            [255, 255, 255, 255, 255, 0, 0],
            [255, 255, 255, 255, 255, 0, 0]
        ], dtype=np.uint8)
    }