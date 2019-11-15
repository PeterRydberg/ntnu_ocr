import os
from PIL import Image, ImageDraw
from imagetools import sliding_window, draw_red_square
from sklearn.svm import SVC
import sklearn.model_selection as splitter
from sklearn.metrics import classification_report
import numpy as np

alphabetical_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def try_remove_element_from_list(fileList, element):
    try:
        fileList.remove(element)
    except ValueError:
        pass

def remove_unwanted_files(fileList):
    try_remove_element_from_list(fileList, 'LICENSE')
    try_remove_element_from_list(fileList, '.DS_Store') # In case of running the program on a Mac.

def get_image_as_array(filepath):
    img = Image.open(filepath)
    list_image = np.array(img, dtype=float).flatten()
    list_image *= (1.0/list_image.max())
    return list_image

def get_data(datapath = "./dataset/chars74k-lite/"):
    image_data = []
    labels = []

    for i, (folder, dirname, files) in enumerate(os.walk(datapath)):
        remove_unwanted_files(files)
        for filename in files:
            relative_path = f"{folder}/{filename}"
            image_data.append(get_image_as_array(relative_path))
            labels.append(i-1)
    
    return image_data, labels

def split(data, test_size):
    return splitter.train_test_split(data[0], data[1], test_size=test_size)

def fit(inputs, outputs):
    pass

def main():
    image_data, labels = get_data("./dataset/chars74k-lite/")
    x_training, x_testing, y_training, y_testing = split([image_data, labels], 0.2)
    classifier = SVC(gamma="scale", verbose=True, probability=True)
    classifier.fit(x_training, y_training)
    print(f"Classifying: {y_training[0]} and got {classifier.predict([x_training[0]])}")

    predicted_test = classifier.predict(x_testing)
    print(classification_report(y_testing, predicted_test, target_names=alphabetical_labels))

img = Image.open("./dataset/detection-images/detection-1.jpg")
for (x, y, window) in sliding_window(image=img, stepSize=8, windowSize=(20, 20)):

    # Conditionally draw square if the probability is considered high enough
    if True:
        newImg = img.copy()
        draw_red_square(x = x, y = y, target_image = newImg, window = window)
        """
        newImg = img.copy()
        draw = ImageDraw.Draw(newImg) # Creates a copy of the image and draws on it
        draw.rectangle((x,y) + (x + window.size[1], y + window.size[0]), fill=128)
        print(f"X: {x}, Y: {y}, Window: {window.size}")
        newImg.save(f"./dump/img{x}-{y}.png", "PNG")
        """

if __name__ == "__main__":
    main()
