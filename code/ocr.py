from PIL import Image
from tools.image import sliding_window, draw_red_square, create_dump_folder_for_images, convert_image_to_array, get_percentage_of_white
from tools.classifier import get_trained_classifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import sys
import pyttsx3


def main():
    arguments = sys.argv[1:]

    """
    --- Help option
    """
    if "-h" in arguments or "--help" in arguments:
        print(f"""
This program uses machine learning methods to predict written letters of different fonts, sizes, rotations, etc.
Authors: johannkv, hmrydber, paaledwl

    Usage: python ocr.py [METHOD] [--FLAGS]

Available commands: \n
        svc: Use SVC.
        ann: Use ANN.
        knn: Use KNN.\n
        --help / -h: Display this text. \n
        --no-save: Do not pickle and save trained classifier. \n
        --train: Ignore existing pickles and retrain classifier. \n
        --use-tts: Uses text-to-speech to read the perceived text. \n
        --no-dump: Do not save output image to file. \n
        --image: File name of an image in detection-images which will be used for the prediction.
        """)
        return
    """
    --- End help option
    """

    ### SVC classification ###
    def SVC_training_method(x_training, x_testing, y_training, y_testing):
        classifier_SVC = SVC(gamma="scale", verbose=False, probability=False)
        classifier_SVC.fit(x_training, y_training)
        print(f"\n\nUsing SVC algorithm:\nClassifying: {y_training[0]} and got {classifier_SVC.predict([x_training[0]])}\n")
        return classifier_SVC

    ### K-nearest neighbors classification ###
    def KNN_training_method(x_training, x_testing, y_training, y_testing):
        classifier_KN = KNeighborsClassifier(n_neighbors=6, weights="distance")
        classifier_KN.fit(x_training, y_training)
        print(f"\n\nUsing K-nearest neighbor algorithm:\nClassifying: {y_training[0]} and got {classifier_KN.predict([x_training[0]])}\n")
        return classifier_KN

    ### ANN classification ###
    def ANN_training_method(x_training, x_testing, y_training, y_testing):
        classifier_ANN = MLPClassifier(solver="adam", alpha=0.0001, learning_rate_init=0.001, max_iter=20000, activation="logistic", learning_rate="adaptive")
        classifier_ANN.fit(x_training, y_training)
        print(f"\n\nUsing neural network algorithm:\nClassifying: {y_training[0]} and got {classifier_ANN.predict([x_training[0]])}\n")
        return classifier_ANN


    # Testing with different classifiers
    if not arguments:
        print("No method specified from command line, using ANN as default")
        check_windows_in_image_with_classifier(classifier = get_trained_classifier("ann.pkl", ANN_training_method, True, True))
    elif arguments[0] == "svc":
        print("Using SVC")
        check_windows_in_image_with_classifier(classifier = get_trained_classifier("svc.pkl", SVC_training_method, True, True))
    elif arguments[0] == "knn":
        print("Using KNN")
        check_windows_in_image_with_classifier(classifier = get_trained_classifier("knn.pkl", KNN_training_method, True, True))
    elif arguments[0] == "ann":
        print("Using ANN")
        check_windows_in_image_with_classifier(classifier = get_trained_classifier("ann.pkl", ANN_training_method, True, True))
    else:
        print("Did not recognize method specified from command line, using ANN as default")
        check_windows_in_image_with_classifier(classifier = get_trained_classifier("ann.pkl", ANN_training_method, True, True))

# Scan the target image for areas in close vicinity of the current window
# If a better candidate is found, i.e. less white parts in the window, return it along with its position and white percentage.
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

"""
This is a cache of squares already checked.
Whenever a new window is checked, after retrieving the best candidate in a square around the window,
it is checked for previous existence in the cache.
If the coordinates for this new candidate is inside another previous candidate, the prediction is added to a dictionary of possibilites for a square.
This can be utilized such that red squares only are drawn at the end, once for each most probable letter and that the printed text only
containes one of each character given a square
"""
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
    # Override default image path if specified in arguments 
    if "--image" in sys.argv[1:]:
        index_of_file_path = sys.argv.index("--image")
        image_path = f"./dataset/detection-images/{sys.argv[index_of_file_path + 1]}"
    print(f"Running OCR on {image_path}")

    img = Image.open(image_path)
    imgCopy = img.convert(mode = "RGB")
    winHeight = 20
    winWidth = 20
    string = ""
    checked_squares = {}
    for (x, y, window) in sliding_window(img, stepSize = 8, windowSize=(winHeight, winWidth)):

        white_percentage = get_percentage_of_white(window)
        # If more than 70 percent of the image is white, it is highly probable that the classifier will be incorrect
        # and that a better candidate probably will be found by the sliding window technique.
        if white_percentage > 0.7:
            continue

        # Check for other candidates in the vicinity of the window
        best_candidate, best_white, best_cand_coords = scan_image_for_area_with_less_white(x, y, img, white_percentage)
        
        # If a better candidate is found set this to be the selected window crop. 
        if best_candidate:
            window = best_candidate
        
        # If image has passed criterias, prepare it for prediction
        img_array = convert_image_to_array(window, use_hog = True, expand_inverted = False)

        # Skip empty images that passed previous tests.
        if len(img_array) == 0:
            continue

        # Predict based on the current window
        predicted = classifier.predict(img_array.reshape(1, -1))

        # If no better candidate was found, use x,y
        best_cand_coords = best_cand_coords if best_cand_coords else (x, y)

        # Init cache if empty
        if len(checked_squares.keys()) == 0:
            checked_squares[best_cand_coords] = {predicted[0]: 1}
        else:
            checked_squares = update_window_cache(window_cache = checked_squares, candidate_coords = best_cand_coords, prediction = predicted[0])
    
    """
    Sort through the cache and construct a sentence of the letters found.
    If, for a given area, more letters have been suggested, use the letter predicted most times.
    """
    cache_prediction = ""
    for predictions in checked_squares.values():
        most_probable_prediction = max(predictions.keys(), key=lambda key: predictions[key])
        cache_prediction += most_probable_prediction
    print(f"Most probable single solution: {cache_prediction}")
    
    # For every box registered with a prediction, draw a red square.
    for (x1, y1) in checked_squares.keys():
        imgCopy = draw_red_square(x = x1, y = y1, target_image = imgCopy)

    # Text-to-speech, a funny addition for entertainment purposes.
    if len(sys.argv) > 1 and "--use-tts" in sys.argv[1:]:
        try:
            tts_engine = pyttsx3.init()
            tts_engine.setProperty('rate', 20)
            tts_engine.say(cache_prediction)
            tts_engine.runAndWait()
        except OSError:
            print("Failed to access local text-to-speech method for this device.")
    
    # Save output image to dump folder
    create_dump_folder_for_images()
    if not "--no-dump" in sys.argv[1:]:
        imgCopy.save("./dump/output.png", "PNG")
    imgCopy.show()


if __name__ == "__main__":
    main()
