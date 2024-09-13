import os
import glob
from PIL import Image
import numpy as np
from ImageManager import ImageManager

# ดึงรายชื่อไฟล์ทั้งหมดจากโฟลเดอร์ images/4 ที่มีนามสกุล .bmp
image_folder = 'images/4'
sequences = glob.glob(os.path.join(image_folder, "*.bmp"))

# ตรวจสอบว่ามีรูปภาพอยู่ในลิสต์หรือไม่
if not sequences:
    print(f"No images found in folder {image_folder}")
else:
    print(f"Found {len(sequences)} images.")

    # เปิดรูปแรกเพื่อดูข้อมูลเบื้องต้น
    img = Image.open(sequences[0])
    data = np.array(img)

    # ดึงขนาดและความลึกของบิต
    width = data.shape[0]
    height = data.shape[1]

    mode_to_bpp = {"1": 1, "L": 8, "P": 8, "RGB": 24, "RGBA": 32, "CMYK": 32, "YCbCr": 24, "LAB": 24, "HSV": 24, "I": 32, "F": 32}
    bitDepth = mode_to_bpp.get(img.mode, 8)  # ใช้ค่า default ที่ 8 ถ้า mode ไม่อยู่ใน dict

    print(f"Image {sequences[0]} with {width} x {height} pixels ({bitDepth} bits per pixel) has been read!")

    # ใช้ฟังก์ชัน ADIAbsolute จาก ImageManager
    im = ImageManager()
    im.read(sequences[0])  # อ่านรูปแรกเข้ามาเพื่อเริ่มกระบวนการ
    im.ADIAbsolute(sequences, 25, 50)  # ประมวลผล ADIAbsolute
    im.write("images/ADIAbsolute.bmp")  # บันทึกภาพใหม่
