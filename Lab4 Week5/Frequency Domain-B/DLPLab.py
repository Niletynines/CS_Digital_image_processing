from PIL import Image
import numpy as np
from ImageManager import ImageManager
from FrequencyDomainManager import FrequencyDomainManager

img = Image.open("image/mandril.bmp")
data = np.array(img)

# shape array
width = data.shape[0]
height = data.shape[1]

print(width, height)

mode_to_bpp = {"1": 1, "L": 8, "P": 8, "RGB": 24, "RGBA": 32, "CMYK": 32, "YCbCr": 24, "LAB": 24, "HSV": 24, "I": 32, "F": 32}
bitDepth = mode_to_bpp[img.mode]

print("Image %s with %s x %s pixels (%s bits per pixel) has been read!" % (img, width, height, bitDepth))

im = ImageManager()
im.read("image/mandril.bmp")
    
# Get the frequency domain of the image
fft = im.getFrequencyDomain("image/mandril.bmp")

# # Write the log-scaled spectrum to an image file
fft.writeSpectrumLogScaled("image/mandrilSpectrum.png")

# # Write the phase spectrum to an image file
fft.writePhase("image/mandrilPhase.png")

# Apply a low-pass filter (e.g., radius = 50)
fft.ILPF(3)

# # Write the filtered spectrum and phase
fft.writeSpectrumLogScaled("image/mandrilLowPassSpectrum_R3.png")
fft.writePhase("image/mandrilLowPassPhase_R3.png")

# # Inverse the transformation back to spatial domain
fft.getInverse()
    
# Save the filtered image
im.write("image/mandrilFiltered_R3.png")