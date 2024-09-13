from PIL import Image
import numpy as np
from ImageManager import ImageManager

img = Image.open("image/gamemaster_noise_2024.bmp")
data = np.array(img)

# shape array
width = data.shape[0]
height = data.shape[1]

mode_to_bpp = {"1": 1, "L": 8, "P": 8, "RGB": 24, "RGBA": 32, "CMYK": 32, "YCbCr": 24, "LAB": 24, "HSV": 24, "I": 32, "F": 32}
bitDepth = mode_to_bpp[img.mode]

print("Image %s with %s x %s pixels (%s bits per pixel) has been read!" % (img, width, height, bitDepth))

#!! กรุณา Comment เเละ Run ทีละ Step

#! Step One
# ขั้นตอนเเรกให้ทำการลด noise โดยใช้ Median Filter กำหนด size 9x9 ก่อน 
# หลังจากนั้นให้ Contraharmonic Filter ทำการลด Pepper noise ออกโดยกำหนด size 7x7 เเละ Q=1.5
# reduce noise
im = ImageManager()
im.read("image/gamemaster_noise_2024.bmp")
im.medianFilter(9)
im.contraharmonicFilter(7,1.5)
im.write("image/denoiseMed9Con7.bmp")

#! Step Two
# ขั้นตอนที่สองทำการเพิ่ม Contrast เพื่อให้สีในรูปที่ Denoise มาเเล้ว เข้มเเละชัดมากขึ้น
# Contrast
im1 = ImageManager()
im1.read("image/denoiseMed9Con7.bmp")
im1.getContrast()
im1.adjustContrast(150)
im1.write("image/contrastMed9Con7.bmp")

#! Step Three
# ขั้นตอนที่สาม convertColor() ทำการเปลี่ยนสีเเต่ละส่วนจากการกำหนดค่า RGB เเละ XY ให้ตรงกับพื้นที่ที่ต้องการ
# จากนั้น RemoveGreen() จำทำการลบสีเขียวที่ไม่ต้องการ เเละไม่เกี่ยวข้องออกให้
# Color
im2 = ImageManager()
im2.read("image/contrastMed9Con7.bmp")
im2.convertColor()
im2.RemoveGreen()
im2.write("image/final.bmp")

#! Step Four
# ขั้นตอนที่สี่เช็คจำนวนสีจาก Histogram
# Histogram
im3 = ImageManager()
im3.read("image/color.bmp")
histogram = im3.getGrayscaleHistogram()
im3.writeHistogramToCSV(histogram, "image/greenUncle/histogram.csv")
im3.write("image/histogram.bmp")
