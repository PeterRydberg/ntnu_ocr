import os
from PIL import Image, ImageDraw, ImageOps
from imagetools import sliding_window, draw_red_square 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
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

def get_image_as_array(filepath, use_hog, expand_inverted):
    img = Image.open(filepath)
    return convert_image_to_array(img, use_hog, expand_inverted)

def convert_image_to_array(img, use_hog, expand_inverted):
    if expand_inverted:
        img = ImageOps.invert(img)
    if use_hog:
        img = hog(img, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(4, 4), block_norm='L2', feature_vector=False)
    list_image = np.array(img, dtype=float).flatten()
    if list_image.max() == 0:
        return []
    # list_image *= (1.0/list_image.max())
    return list_image

def get_percentage_of_white(img):
    list_image = np.array(img, dtype=float).flatten()
    numberOfWhite = np.count_nonzero(list_image == 255.)
    return numberOfWhite/400

def get_data(datapath = "./dataset/chars74k-lite/", use_hog=False, expand_inverted = False):
    image_data = []
    labels = []

    for i, (folder, dirname, files) in enumerate(os.walk(datapath)):
        remove_unwanted_files(files)
        for filename in files:
            relative_path = f"{folder}/{filename}"
            image_data.append(get_image_as_array(relative_path, use_hog, False))
            labels.append(i-1)

            if expand_inverted:
                image_data.append(get_image_as_array(relative_path, use_hog, True))
                labels.append(i-1)
    
    return image_data, labels

def split(data, test_size):
    return splitter.train_test_split(data[0], data[1], test_size=test_size)

def get_trained_classifier(path, classifierTraining):
    savedExists = os.path.isfile(path)
    if savedExists:
        print ("Getting pre-trained method")
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        print("Training...")
        classifier = classifierTraining()
        with open(path, "wb+") as f:
            pickle.dump(classifier, f)
        return classifier

def get_letter_prediction(pred):
    return alphabetical_labels[int(pred)]

def evaluate_classifier(inputs, outputs, classifier):
    predicted_test = classifier.predict(inputs)
    print(classification_report(outputs, predicted_test, target_names=alphabetical_labels))

def main():
    image_data, labels = get_data("./dataset/chars74k-lite/", use_hog=True, expand_inverted=True)
    x_training, x_testing, y_training, y_testing = split([image_data, labels], 0.2)

    ### SVC classification ###
    def SVC_training_method():
        classifier_SVC = SVC(gamma="scale", verbose=True, probability=False)
        classifier_SVC.fit(x_training, y_training)
        print(f"\n\nUsing SVC algorithm:\nClassifying: {get_letter_prediction(y_training[0])} and got {get_letter_prediction(classifier_SVC.predict([x_training[0]]))}\n")
        evaluate_classifier(x_testing, y_testing, classifier_SVC)
        return classifier_SVC

    ### K-nearest neighbors classification ###
    def KNN_training_method():
        classifier_KN = KNeighborsClassifier(n_neighbors=6, weights="distance")
        classifier_KN.fit(x_training, y_training)
        print(f"\n\nUsing K-nearest neighbor algorithm:\nClassifying: {get_letter_prediction(y_training[0])} and got {get_letter_prediction(classifier_KN.predict([x_training[0]]))}\n")
        evaluate_classifier(x_testing, y_testing, classifier_KN)
        return classifier_KN

    ### ANN classification ###
    def ANN_training_method():
        classifier_ANN = MLPClassifier(solver="adam", alpha=0.0001, learning_rate_init=0.001, max_iter=10000, activation="logistic", learning_rate="adaptive")
        classifier_ANN.fit(x_training, y_training)
        print(f"\n\nUsing neural network algorithm:\nClassifying: {get_letter_prediction(y_training[0])} and got {get_letter_prediction(classifier_ANN.predict([x_training[0]]))}\n")
        evaluate_classifier(x_testing, y_testing, classifier_ANN)
        return classifier_ANN

    # Testing with different classifiers
    check_windows_in_image_with_classifier(classifier = get_trained_classifier("svc.pkl", SVC_training_method))
    #check_windows_in_image_with_classifier(classifier = get_trained_classifier("knn.pkl", KNN_training_method))
    #check_windows_in_image_with_classifier(classifier = get_trained_classifier("ann.pkl", ANN_training_method))

def scan_image_for_area_with_less_white(x, y, image, white_percentage = 1):
    best_white = white_percentage
    best_image = None
    for x1 in range(x - 15, x + 15):
        for y1 in range(y - 15, y + 15):
            candidate = image.crop([x1, y1, x1 + 20, y1 + 20])
            white_in_candidate = get_percentage_of_white(candidate)
            if white_in_candidate < best_white:
                best_white = white_in_candidate
                best_image = candidate
    return best_image, best_white

def check_windows_in_image_with_classifier(classifier, image_path = "./dataset/detection-images/detection-3.jpg"):
    img = Image.open(image_path)
    imgCopy = None
    winHeight = 20
    winWidth = 20
    string = ""
    for (x, y, window) in sliding_window(img, stepSize = 6, windowSize=(winHeight, winWidth)):
        # Skip windows which surpasses image border
        if window.size[0] != winHeight or window.size[1] != winWidth:
            continue

        white_percentage = get_percentage_of_white(window)

        # If more tn 90 percent of the image is white, it is highly probable that the classifier will be incorrect
        if white_percentage > 0.5:
            continue

        best_candidate, best_white = scan_image_for_area_with_less_white(x, y, img, white_percentage)
        if best_white > 0.5:
            continue
        if best_candidate:
            window = best_candidate
        img_array = convert_image_to_array(window, use_hog = 1, expand_inverted = False)
        # Conditionally draw square if the probability is considered high enough

        # Skip completely white images
        if len(img_array) == 0:
            continue

        predicted = classifier.predict([img_array])
        print(f"predicted of window: {predicted}")
        string += get_letter_prediction(predicted) 
        if len(predicted) > 0:
            if not imgCopy: 
                imgCopy = img.copy()
            imgCopy = draw_red_square(x = x, y = y, target_image = imgCopy, window = window)
    print(string)
    imgCopy.save("./dump/concat.png", "PNG")

if __name__ == "__main__":
    main()
