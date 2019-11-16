import os
from PIL import Image, ImageDraw, ImageColor, ImageOps
from skimage.feature import hog
import numpy as np

def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.size[1], stepSize):
        for x in range(0, image.size[0], stepSize):
            yield (x, y, image.crop([x, y, x + windowSize[1], y + windowSize[0]]))

def draw_red_square(x, y, target_image):
    draw = ImageDraw.Draw(target_image) 
    draw.rectangle((x,y) + (x + 20, y + 20), outline="#ff0000")
    # target_image.save("./dump/img1.png", "PNG")
    return target_image

def draw_grey_square(x, y, target_image, window):
    draw = ImageDraw.Draw(target_image)
    draw.rectangle((x,y) + (x + window.size[1], y + window.size[0]), outline="#333333")
    return target_image

def create_dump_folder_for_images():
    if os.path.exists("./dump"):
        return
    print('Creating dump directory for output images')
    try:
        os.mkdir("./dump")
        print("Successfully created dump folder")
    except OSError:
        print("Could not create a dump folder. Please create one in the same path as this file")

def get_image_as_array(filepath, use_hog, expand_inverted):
    img = Image.open(filepath)
    img = img.convert(mode="L")
    img.resize((20, 20))
    return convert_image_to_array(img, use_hog, expand_inverted)

def convert_image_to_array(img, use_hog, expand_inverted):
    if expand_inverted:
        img = ImageOps.invert(img)
    if use_hog:
        img = hog(img, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(4, 4), block_norm='L2', feature_vector=True)
    list_image = np.array(img, dtype=float).flatten()
    if list_image.max() == 0:
        return []
    return list_image

def get_percentage_of_white(img):
    list_image = np.array(img, dtype=float).flatten()
    numberOfWhite = np.count_nonzero(list_image == 255.)
    return numberOfWhite/400