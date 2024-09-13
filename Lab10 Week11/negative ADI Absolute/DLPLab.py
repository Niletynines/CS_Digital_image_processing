import os
import glob
from PIL import Image
import numpy as np
from ImageManager import ImageManager

# ดึงรายชื่อไฟล์ทั้งหมดจากโฟลเดอร์ images/4 ที่มีนามสกุล .bmp
image_folder = 'images/4'
sequences = glob.glob(os.path.join(image_folder, "*.bmp"))

if not sequences:
    print(f"No images found in folder {image_folder}")
else:
    print(f"Found {len(sequences)} images.")

    img = Image.open(sequences[0])
    data = np.array(img)

    width = data.shape[0]
    height = data.shape[1]

    mode_to_bpp = {"1": 1, "L": 8, "P": 8, "RGB": 24, "RGBA": 32, "CMYK": 32, "YCbCr": 24, "LAB": 24, "HSV": 24, "I": 32, "F": 32}
    bitDepth = mode_to_bpp.get(img.mode, 8) 

    print(f"Image {sequences[0]} with {width} x {height} pixels ({bitDepth} bits per pixel) has been read!")

    # negative ADI Absolute
    im = ImageManager()
    im.read(sequences[0])  
    im.negativeADIAbsolute(sequences, 25, 50) 
    im.write("images/negativeADIAbsolute.bmp")  
