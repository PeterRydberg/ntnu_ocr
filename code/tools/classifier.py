import os
from tools.data import get_data, split
from sklearn.metrics import classification_report
import sys
import pickle


# Get a pre-trained pickled classifier if it already exists and none of the flag prevents this.
# This allows for mucher faster classification of new images as the model does not need to be trained every time.
# It takes a function which trains a classifier based on training and testing data and pickles it
# if no classifier already exists on file.
def get_trained_classifier(path, classifierTraining, use_hog, expand_inverted):
    saved_exists = os.path.isfile(path)
    retrain = "--train" in sys.argv[1:]
    nosave = "--no-save" in sys.argv[1:]
    
    if saved_exists and not retrain:
        print("Found .pkl-file matching model type")
        print("Getting pre-trained model...")
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        print("Training new model...")
        image_data, labels = get_data("./dataset/chars74k-lite/", use_hog, expand_inverted)
        x_training, x_testing, y_training, y_testing = split([image_data, labels], 0.2)
        
        classifier = classifierTraining(x_training, x_testing, y_training, y_testing)
        evaluate_classifier(x_testing, y_testing, classifier)
        if not nosave:
            if saved_exists:
                os.remove(path)
            with open(path, "wb+") as f:
                pickle.dump(classifier, f)
        return classifier

def evaluate_classifier(inputs, outputs, classifier):
    predicted_test = classifier.predict(inputs)
    print(classification_report(outputs, predicted_test))