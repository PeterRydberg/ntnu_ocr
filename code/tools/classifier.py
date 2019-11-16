import os
from datatools import get_data, split
from sklearn.metrics import classification_report
import pickle


def get_trained_classifier(path, classifierTraining, use_hog, expand_inverted):
    savedExists = os.path.isfile(path)
    if savedExists:
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
        with open(path, "wb+") as f:
            pickle.dump(classifier, f)
        return classifier

def evaluate_classifier(inputs, outputs, classifier):
    predicted_test = classifier.predict(inputs)
    print(classification_report(outputs, predicted_test))