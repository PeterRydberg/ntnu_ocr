# Assignment 5 - OCR
By Johannes Kvamme (johannkv), Pål Larsen (paaledwl), Peter Rydberg (hmrydber)

## Introduction

- [ ] a) A brief overview of the whole system. This includes a short explanation on how to run it as well as what libraries your system is using

> 1. How to run the program
> 2. What happens when we run the program

### Language and tools
The programming language chosen was Python. Python was selected due to the fact that it is one of, if not the most, utilized programming language used in machine learning. The language contains many good libraries to help support implementations of these algorithms. Scikit-learn^[1] (sklearn) was then chosen as the main library as it is one of the most known Python machine learning libraries for implementing the different classification algorithms (and more).

### Used packages:
> Check if the list contains all the imports

**PIL**: Used for reading, loading, and writing to images.  
**Numpy**: Machine learning requires a lot of matrix calculations which numpy is incredibly good at.  
**Pickle**: Serializes the training models which lets us skip training the model every time the program is ran.  
**Scikit-image**: Another package by Scikit which is dedicated to pre-process images. From Skimage it is for example simple to extract Histogram of Oriented Gradients (HOG) with the method ```hog```.
**pyttsx3**: Text-to-speech synthesizer.

### Installation guide
Create a virtual environment using the command `py -m venv env`, then activate the environment. The packages themselves can be installed using the command `pip install -r requirements.txt` inside the `code`-folder. When these are installed, run `python ocr.py`. The args and flags that can be sent to the program are:
* `svc`: Runs the task by training an SVC model and applying it to the classification.
* `knn`: Runs the task by training a K-Nearest Neighbor model and applying it to the classification.
* `ann`: Runs the task by training an Artificial Neural Network and applying it to the classification.
* `--use-tts`: Produces a text-to-speech voice which reads the text on the image.

### How to use
When running the program, it will first fetch a classifier, either by training a new one or fetching a cached classifier from Pickle. Then the Sliding Window technique is 

## Feature Engineering
- [x] b) Explain how each of the feature engineering techniques you selected work. Why did you select these techniques? Justify your answer.
- [ ] c) Were there any techniques that you wanted to try, but for some reason were not able to? Explain


As mentioned in the assignment, there exists many ways of getting features from an image. We decided that cropping the data by hand would lead to human error, and quickly found by trial and error that feature descriptors worked well.

### Histograms of Oriented Gradients (HOG)
The way that HOGs extract features from an image is similar to other methods, but HOGs creates an overlay of a dense grid of cells and uses local contrast normalization (LCN), which normalizes the contrast of an image non-linearly, to increase accuracy.

In the assignment the article on HOGs^[2], the authors explain how HOGs work better than SIFT and some other methods which is one reason we decided on the method. After trying out the HOG method, easily implemented with ```skimage```, we saw an increase of approximately 10% accuracy on training the data. We quickly decided that HOG was a feature descriptor we would keep on using. 

### Inverted images
When training charachter recongnition softwares, a lot of the training data contains a black background with white text. In the real world, a lot of text have a white background and black text. You can then quickly understand that when using an OCR software in the real world it can scan text wrong more often than in the training data.

By inverting the image colors, making black backgrounds white, and white text black, then adding them to the training samples, the OCR can with higher certainy recongnize charachters in a real world image.

### Other pre processing methods
A couple of other methods were considered, but quickly dropped in favour of HOGs and inverted images. Scale-invariant feature transform (SIFT) was considered at first, but was dropped when reading the article about HOGs^[2] and how HOGs were more effective than SIFTs.
Manually going through the images were considered briefly, but was dropped when the size of the dataset was discovered.

## Character Classification

- [x] d) After having looked at the dataset, did you have any initial idea about what kind of model that might work well? Explain your reasoning.
- [ ] e) Give a description of the two models you elected to use. The description must include a brief explanation on how they work. Why did you select these models?
- [ ] f) A critical evaluation of your two models. How are you measuring their performance? How did they do? Which model gave the best results? Include at least five predictions in the report (both good
and bad).
- [ ] g) Were there any additional models that you would have liked to try, but for some reason were not able to? Explain.


Due to previous experience with assignment 3 in this course and the fact that this assignment is closely related, the group's first thought on which models could be appliccable were the algorithms Artificial Neural Networks (ANNs), Support Vector Machines (SVMs), and K nearest neighbor (K-NN). The group's idea of using these models were then solidifed after reading the article 'Charachter Recognition in Natural Images'^[3] where they implement nearest neighbor and support vector machines to recongnize charachters.

### Our models
Image detection is a fundemental and exciting part of machine learning and as an exercise, fun to implement and try out. In this assignment the group decided to implement three different algorithms. The algorithms chosen were K-NN, SVM, and ANN which sklearn has methods to help implement.

#### K-NN
K-NN being an algorithm that is quite simple to implement, understand, and the fact that it often performs quite well compared to many other algorithms^[4], makes it a quite good candidate for this task. 

K-NN being an instance based learning algorithm, storing the features and labels from the training data. It doesn't generalize the data before it starts classifying. The model's tuning input is the amount of _k_ neigbbors one should search for. When classifying K-NN starts assigning labels to the data based on the frequency of the training samples.

When K-NN is used for pattern recognition the algorithm classifies the objects in an image based on the feature space.

> Did our model do good?

#### SVM
> Describe

> Did our model do good?

#### ANN
Aritficial neural networks are a more complex model than both K-NN and SVMs and often referenced as a 'black box' machine learning algorithm. ANN in short, are inspired by biological neural networks and uses a set of nodes, aritificial neurons, which are based on the biological neurons. Signals, numbers, are sent between these neurons which compute the signals and adjust the weights between the edges of the neurons.

ANNs were originally created to try to copy the human brain and how it works. Over time this mindset was abandoned and entered the area of trying to execute specific tasks. In the early 2010s ANNs became incredibly good at different tasks, like image and pattern recognition^[5]

> Did our model do good?

### Additional models
> Other models?

## Character Detection

- [ ] h) Test your character detector on detection-1.jpg and detection-2.jpg and show the result in
the report. Feel free to find or create additional images to test your detector, if you are so inclined.
- [ ] i) Give an evaluation of your detection system. How does it perform?
- [ ] j) Describe any improvements you made to your detector. Discuss how you can improve your system
further


## Conclusion
- [ ] k) What is the weakest and strongest component of your OCR system (feature engineering, character
classification, and character detection)? Explain your answer.
- [ ] l) What went good/bad with the project? Any lessons learned?

## References
**Id, Reference Link, Title, Authors**
[1]: https://scikit-learn.org/stable/index.html "Scikit Learn" - "Scikit"
[2]: https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf "Histograms of Oriented Gradients for Human Detection" - "Navneet Dalal and Bill Triggs"
[3]: http://personal.ee.surrey.ac.uk/Personal/T.Decampos/papers/decampos_etal_visapp2009.pdf "Charachter Recognition in Natural Images" - "Teofilo E. de Campos, Bodla Rakesh Babu, Manik Varma"
[4]: http://www.rroij.com/open-access/object-recognition-using-knearest-neighborsupported-by-eigen-value-generated-fromthe-features-of-an-image.php?aid=46808 "Object Recognition Using K-Nearest Neighbor Supported By Eigen Value Generated From the Features of an Image" - "Dr. R.Muralidharan"
[5]: https://web.archive.org/web/20180831075249/http://www.kurzweilai.net/how-bio-inspired-deep-learning-keeps-winning-competitions "How bio-inspired deep learning keeps winning competitions", "Amara D. Angelica, Jürgen Schmidhuber"
