import os
from PIL import Image, ImageDraw
from imagetools import sliding_window, draw_red_square 
from sklearn.svm import SVC
import sklearn.model_selection as splitter
from sklearn.metrics import classification_report
from skimage.feature import hog
import numpy as np
import pickle

alphabetical_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def try_remove_element_from_list(fileList, element):
    try:
        fileList.remove(element)
    except ValueError:
        pass

def remove_unwanted_files(fileList):
    try_remove_element_from_list(fileList, 'LICENSE')
    try_remove_element_from_list(fileList, '.DS_Store') # In case of running the program on a Mac.

def get_image_as_array(filepath, use_hog):
    img = Image.open(filepath)
    return convert_image_to_array(img, use_hog)

def convert_image_to_array(img, use_hog, debug = False):
    if debug: print(f"Img before hog: {img}")
    if use_hog: img = hog(img, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(4, 4), block_norm='L2', feature_vector=False)
    if debug: print(f"Img after hog: {img}")
    list_image = np.array(img, dtype=float).flatten()
    if list_image.max() == 0:
        return []
    if debug: print(f"List image after flatten: {list_image}")
    list_image *= (1.0/list_image.max())
    return list_image

def get_data(datapath = "./dataset/chars74k-lite/", use_hog=False):
    image_data = []
    labels = []

    for i, (folder, dirname, files) in enumerate(os.walk(datapath)):
        remove_unwanted_files(files)
        for filename in files:
            relative_path = f"{folder}/{filename}"
            image_data.append(get_image_as_array(relative_path, use_hog))
            labels.append(i-1)
    
    return image_data, labels

def split(data, test_size):
    return splitter.train_test_split(data[0], data[1], test_size=test_size)

def get_trained_SVC(x_training, y_training):
    svcExists = os.path.isfile("svc.pkl")
    if svcExists:
        print("Getting pre-trained SVC")
        with open("svc.pkl", "rb") as f:
            return pickle.load(f)
    else:
        print("Training...")
        classifier = SVC(gamma="scale", probability=False)
        classifier.fit(x_training, y_training)
        with open("svc.pkl", "wb+") as f:
            pickle.dump(classifier, f)
        return classifier


def main():
    image_data, labels = get_data("./dataset/chars74k-lite/", True)
    x_training, x_testing, y_training, y_testing = split([image_data, labels], 0.2)
    #classifier = SVC(gamma="scale", probability=False)
    #print("Training...")
    #classifier.fit(x_training, y_training)
    classifier = get_trained_SVC(x_training, y_training)
    print(f"Classifying: {y_training[0]} and got {classifier.predict([x_training[0]])}")

    predicted_test = classifier.predict(x_testing)
    print(classification_report(y_testing, predicted_test, target_names=alphabetical_labels))
    check_windows_in_image_with_classifier(classifier = classifier)

def check_windows_in_image_with_classifier(classifier, image_path = "./dataset/detection-images/detection-1.jpg"):
    global alphabetical_labels
    img = Image.open(image_path)
    imgCopy = None
    winHeight = 20
    winWidth = 20
    string = ""
    for (x, y, window) in sliding_window(img, stepSize = 8, windowSize=(winHeight, winWidth)):
        if window.size[0] != winHeight or window.size[1] != winWidth:
            continue
        # Conditionally draw square if the probability is considered high enough
        img_array = convert_image_to_array(window, use_hog = True, debug = False)
        if len(img_array) == 0:
            continue
        predicted = classifier.predict([img_array])
        print(f"predicted of window: {predicted}")
        string += alphabetical_labels[predicted[0]]
        if len(predicted) > 0:
            if not imgCopy: 
                imgCopy = img.copy()
            imgCopy = draw_red_square(x = x, y = y, target_image = imgCopy, window = window)
    print(string)
    imgCopy.save("./dump/concat.png", "PNG")

if __name__ == "__main__":
    main()
