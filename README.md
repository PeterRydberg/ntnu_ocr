# ntnu_ocr
Assignment 5 for the Machine Learning course at NTNU

### Installation guide
Create a virtual environment using the command `python -m venv env`, then activate the environment. The packages themselves can be installed using the command `pip install -r requirements.txt` inside the `code`-folder. 

#### Usage
When these are installed, run `python ocr.py`. 
The args and flags that can be sent to the program are:
* `svc`: Runs the task by training an SVC model and applying it to the classification.
* `knn`: Runs the task by training a K-Nearest Neighbor model and applying it to the classification.
* `ann`: Runs the task by training an Artificial Neural Network and applying it to the classification.
* `-h / --help`: Displays a help text with all options specified here.
* `--use-tts`: Produces a text-to-speech voice which reads the text on the image.
* `--no-save`: Do not save the trained classifier to file.
* `--train`: Ignores any saved classifiers and retrains.
* `--no-dump`: Do not generate an output image.
* `--image {filename}`: This flag will choose which image the OCR system aims to test on, where {filename} is the path of the file. Defaults to the first detection image provided in the dataset.

When running the program, it will first fetch a classifier, either by training a new one or fetching a cached, pre-trained classifier from Pickle. Then the Sliding Window technique is applied to a given image in order to detect present objects. The classifier is then used to detect which characters each object relates to, and prints the string (and reads it aloud if `--use-tts` is set).
