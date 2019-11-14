import os
from PIL import Image, ImageDraw
from imager import sliding_window
from sklearn.svm import SVC
import sklearn.model_selection as splitter
import numpy as np

def get_image(filepath):
    print(f"filepath: {filepath}")
    img = Image.open(filepath)
    return np.array(img)

def get_data(datapath = "./dataset/chars74k-lite/"):
    image_data = np.array([])
    labels = np.array([])

    for (folder, dirname, files) in os.walk(datapath):
        for filename in files[1:]:
            relative_path = f"{folder}/{filename}"
            image_data = np.append(image_data, get_image(relative_path), axis=1)
            labels = np.append(labels, 1)
    
    return image_data, labels

def split(data, test_size):
    return splitter.train_test_split(data[0], data[1], test_size=test_size)

def fit():
    pass

def main():
    image_data, labels = get_data("./dataset/chars74k-lite/")
    print(f"image_data: {image_data}")
    x_training, x_testing, y_training, y_testing = split([image_data, labels], 20)
    print(f"input 1: {x_training[0]}\nlabel 1: {y_training[0]}")

img = Image.open("./dataset/detection-images/detection-1.jpg")
for (x, y, window) in sliding_window(image=img, stepSize=8, windowSize=(20, 20)):

    # Conditionally draw square if the probability is considered high enough
    if False:
        newImg = img.copy()
        draw = ImageDraw.Draw(newImg) # Creates a copy of the image and draws on it
        draw.rectangle((x,y) + (x + window.size[1], y + window.size[0]), fill=128)
        print(f"X: {x}, Y: {y}, Window: {window.size}")
        newImg.save(f"./dump/img{x}-{y}.png", "PNG")

if __name__ == "__main__":
    main()
