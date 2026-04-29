#!/usr/bin/env python3
"""Create small test TIFF files for Render free tier"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds

# Create SMALL test images (256x256 pixels)
height, width = 256, 256

# Band data: NIR, RED, GREEN, BLUE
nir = np.random.randint(1000, 5000, (height, width), dtype=np.uint16)
red = np.random.randint(500, 3000, (height, width), dtype=np.uint16)
green = np.random.randint(500, 3000, (height, width), dtype=np.uint16)
blue = np.random.randint(300, 2000, (height, width), dtype=np.uint16)

# Stack bands
data = np.array([nir, red, green, blue])

# Create transform
transform = from_bounds(0, 0, 100, 100, width, height)

# Write BEFORE image
with rasterio.open(
    'before_small.tif',
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=4,
    dtype=data.dtype,
    transform=transform
) as dst:
    dst.write(data)
    print("✓ Created before_small.tif (256x256 pixels)")

# Create AFTER image with small variation
nir_after = nir + np.random.randint(-200, 200, (height, width))
red_after = red + np.random.randint(-100, 100, (height, width))
green_after = green + np.random.randint(-100, 100, (height, width))
blue_after = blue + np.random.randint(-50, 50, (height, width))

data_after = np.array([nir_after, red_after, green_after, blue_after])
data_after = np.clip(data_after, 0, 65535).astype(np.uint16)

# Write AFTER image
with rasterio.open(
    'after_small.tif',
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=4,
    dtype=data_after.dtype,
    transform=transform
) as dst:
    dst.write(data_after)
    print("✓ Created after_small.tif (256x256 pixels)")

print("\n✅ Test files ready!")
print("File sizes should be < 1MB each")
print("\nUpload these in the app:")
print("  - before_small.tif")
print("  - after_small.tif")
