import os
import sklearn.model_selection as splitter
from imagetools import get_image_as_array

def remove_unwanted_files(fileList):
    try_remove_element_from_list(fileList, 'LICENSE')
    try_remove_element_from_list(fileList, '.DS_Store') # In case of running the program on a Mac.


def try_remove_element_from_list(fileList, element):
    try:
        fileList.remove(element)
    except ValueError:
        pass

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
