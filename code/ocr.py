import os
from PIL import Image, ImageDraw, ImageOps
from imagetools import sliding_window, draw_red_square, draw_grey_square 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import sklearn.model_selection as splitter
from sklearn.metrics import classification_report
from skimage.feature import hog
import numpy as np
import pickle
import sys
import pyttsx3


def create_dump_folder_for_images():
    if os.path.exists("./dump"):
        return
    print('Creating dump directory for output images')
    try:
        os.mkdir("./dump")
        print("Successfully created dump folder")
    except OSError:
        print("Could not create a dump folder. Please create one in the same path as this file")


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

def get_data(datapath = "./dataset/chars74k-lite/", use_hog=False, expand_inverted = False):
    image_data = []
    labels = []

    for (folder, dirname, files) in os.walk(datapath):
        remove_unwanted_files(files)
        folder_letter = folder.split("/")[-1]
        for filename in files:
            relative_path = f"{folder}/{filename}"
            image_data.append(get_image_as_array(relative_path, use_hog, False))
            labels.append(folder_letter)

            if expand_inverted:
                image_data.append(get_image_as_array(relative_path, use_hog, True))
                labels.append(folder_letter)
    
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

def evaluate_classifier(inputs, outputs, classifier):
    predicted_test = classifier.predict(inputs)
    print(classification_report(outputs, predicted_test))

def main():
    arguments = sys.argv[1:]
    image_data, labels = get_data("./dataset/chars74k-lite/", use_hog=True, expand_inverted=True)
    x_training, x_testing, y_training, y_testing = split([image_data, labels], 0.2)

    ### SVC classification ###
    def SVC_training_method():
        classifier_SVC = SVC(gamma="scale", verbose=False, probability=False)
        classifier_SVC.fit(x_training, y_training)
        print(f"\n\nUsing SVC algorithm:\nClassifying: {y_training[0]} and got {classifier_SVC.predict([x_training[0]])}\n")
        evaluate_classifier(x_testing, y_testing, classifier_SVC)
        return classifier_SVC

    ### K-nearest neighbors classification ###
    def KNN_training_method():
        classifier_KN = KNeighborsClassifier(n_neighbors=6, weights="distance")
        classifier_KN.fit(x_training, y_training)
        print(f"\n\nUsing K-nearest neighbor algorithm:\nClassifying: {y_training[0]} and got {classifier_KN.predict([x_training[0]])}\n")
        evaluate_classifier(x_testing, y_testing, classifier_KN)
        return classifier_KN

    ### ANN classification ###
    def ANN_training_method():
        classifier_ANN = MLPClassifier(solver="adam", alpha=0.0001, learning_rate_init=0.001, max_iter=20000, activation="logistic", learning_rate="adaptive")
        classifier_ANN.fit(x_training, y_training)
        print(f"\n\nUsing neural network algorithm:\nClassifying: {y_training[0]} and got {classifier_ANN.predict([x_training[0]])}\n")
        evaluate_classifier(x_testing, y_testing, classifier_ANN)
        return classifier_ANN

    # Testing with different classifiers
    if arguments[0] == "svc":
        check_windows_in_image_with_classifier(classifier = get_trained_classifier("svc.pkl", SVC_training_method))
    elif arguments[0] == "knn":
        check_windows_in_image_with_classifier(classifier = get_trained_classifier("knn.pkl", KNN_training_method))
    elif arguments[0] == "ann":
        check_windows_in_image_with_classifier(classifier = get_trained_classifier("ann.pkl", ANN_training_method))
    else:
        print("No method specified from command line, using ANN as default")
        check_windows_in_image_with_classifier(classifier = get_trained_classifier("ann.pkl", ANN_training_method))

def scan_image_for_area_with_less_white(x, y, image, white_percentage = 1):
    best_white = white_percentage
    best_image = None
    image_coordinates = None
    for x1 in range(x - 15, x + 15):
        for y1 in range(y - 15, y + 15):
            candidate = image.crop([x1, y1, x1 + 20, y1 + 20])
            white_in_candidate = get_percentage_of_white(candidate)
            if white_in_candidate < best_white:
                best_white = white_in_candidate
                best_image = candidate
                image_coordinates = (x1, y1)
    return best_image, best_white, image_coordinates

def update_window_cache(window_cache, candidate_coords, prediction):
    updated = False
    # Check for matching coords in the already existing coords, and update if match in range
    for (x1, y1) in window_cache.keys():
        between_x = candidate_coords[0] > (x1 - 15) and candidate_coords[0] < (x1 + 20)
        between_y = candidate_coords[1] > (y1 - 15) and candidate_coords[1] < (y1 + 20)
        if between_x and between_y:
            score_dict = window_cache[(x1, y1)]
            if prediction in score_dict:
                score_dict[prediction] += 1
            else:
                score_dict[prediction] = 1
            window_cache[(x1, y1)] = score_dict
            updated = True
            break

    # If no match was found, create a new entry with the given prediction
    if not updated:
        window_cache[candidate_coords] = {prediction: 1}
    return window_cache 


def check_windows_in_image_with_classifier(classifier, image_path = "./dataset/detection-images/detection-2.jpg"):
    img = Image.open(image_path)
    imgCopy = img.convert(mode = "RGB")
    winHeight = 20
    winWidth = 20
    string = ""
    checked_squares = {}
    for (x, y, window) in sliding_window(img, stepSize = 8, windowSize=(winHeight, winWidth)):
        # Skip windows which surpasses image border
        if window.size[0] != winHeight or window.size[1] != winWidth:
            continue

        white_percentage = get_percentage_of_white(window)

        # If more than 90 percent of the image is white, it is highly probable that the classifier will be incorrect
        if white_percentage > 0.7:
            continue

        best_candidate, best_white, best_cand_coords = scan_image_for_area_with_less_white(x, y, img, white_percentage)
        if best_white > 0.7:
            continue
        if best_candidate:
            window = best_candidate
        
        # If image has passed criterias, prepare it for prediction
        img_array = convert_image_to_array(window, use_hog = 1, expand_inverted = False)

        # Skip completely white images 
        # TODO: Check whether this has any function now that we check white percentage
        if len(img_array) == 0:
            continue

        predicted = classifier.predict(img_array.reshape(1, -1))

        """
        CACHE BEGIN
        This is a cache of squares already checked.
        Whenever a new window is checked, after retrieving the best candidate in a square around the window,
        it is checked for previous existence in the cache.
        If the coordinates for this new candidate is inside another previous candidate, the prediction is added to a dictionary of possibilites for a square.
        This can be utilized such that red squares only are drawn at the end, once for each most probable letter and that the printed text only
        containes one of each character given a square
        """
        # If no better candidate was found, use x,y
        best_cand_coords = best_cand_coords if best_cand_coords else (x, y)
        # Init cache if empty
        if len(checked_squares.keys()) == 0:
            checked_squares[best_cand_coords] = {predicted[0]: 1}
        else:
            checked_squares = update_window_cache(window_cache = checked_squares, candidate_coords = best_cand_coords, prediction = predicted[0])


        # CACHE END

        #print(f"predicted of window: {predicted}")
        if len(predicted) > 0:
            imgCopy = draw_grey_square(x = x, y = y, target_image = imgCopy, window = window)
    
    cache_prediction = ""
    for predictions in checked_squares.values():
        most_probable_prediction = max(predictions.keys(), key=lambda key: predictions[key])
        cache_prediction += most_probable_prediction
    for (x1, y1) in checked_squares.keys():
        imgCopy = draw_red_square(x = x1, y = y1, target_image = imgCopy)
    print(f"Most probable single solution: {cache_prediction}")
    if sys.argv[2] == "--use-tts":
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 10)
        tts_engine.say(cache_prediction)
        tts_engine.runAndWait()
    create_dump_folder_for_images()
    imgCopy.save("./dump/concat.png", "PNG")

if __name__ == "__main__":
    main()
