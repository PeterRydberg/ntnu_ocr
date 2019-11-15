# Assignment 5 - OCR
By Johannes Kvamme (johannkv), Pål Larsen (paaledwl), Peter Rydberg (hmrydber)

## Introduction
```
a) A brief overview of the whole system. This includes a short explanation on how to run it as well as
what libraries your system is using
```
The programming language chosen was Python. Python was selected due to the fact that it is one of, if not the most, used programming language used in machine learning. The language contains many good libraries to help support implementations of these algorithms. Scikit.learn^[1] (sklearn) was then chosen as the main library as it is one of the most known Python machine learning libraries for implementing the different classification algorithms (and more).

#### Other libraries used in the project are:
**PIL**: Used for reading, loading, and writing to images.  
**Numpy**: Machine learning requires a lot of matrix calculations which numpy is incredibly good at.  
**Pickle**: Serializes the training models which lets us skip training the model every time the program is ran.  
**Skimage**: Another package by Scikit which is dedicated to pre-process images. From Skimage it is simple to extract Histogram of Oriented Gradients (HOG) with the method ```hog```  
**Imagetools**: A library used to help read images, in our case implementing Sliding Window to read the charahters.


## Feature Engineering
```
b) Explain how each of the feature engineering techniques you selected work. Why did you select these
techniques? Justify your answer.
c) Were there any techniques that you wanted to try, but for some reason were not able to? Explain
```
As mentioned in the assignment, there exists many ways of getting features from an image. We decided that cropping the data by hand would lead to human error, and quickly found by trial and error that feature descriptors worked well.

### Histograms of Oriented Gradients (HOG)
The way that HOGs extract features from an image is similar to other methods, but HOGs creates an overlay of a dense grid of cells and uses local contrast normalization (LCN), which normalizes the contrast of an image non-linearly, to increase accuracy.

In the assignment the article on HOGs^[3], the authors explain how HOGs work better than SIFT and some other methods which is one reason we decided on the method. After trying out the HOG method, easily implemented with ```skimage```, we saw an increase of approximately 10% accuracy on training the data. We quickly decided that HOG was a feature descriptor we would keep on using. 

## Character Classification
```
d) After having looked at the dataset, did you have any initial idea about what kind of model that
might work well? Explain your reasoning.
e) Give a description of the two models you elected to use10. The description must include a brief
explanation on how they work. Why did you select these models?
f) A critical evaluation of your two models. How are you measuring their performance? How did they
do? Which model gave the best results? Include at least five predictions in the report (both good
and bad).
g) Were there any additional models that you would have liked to try, but for some reason were not
able to? Explain.
```

### Our models
Image detection is a fundemental and exciting part of machine learning and as an exercise, fun to implement and try out. In this assignment the group decided to implement three different algorithms. The algorithms chosen were K Nearest Neighbor (K-NN), Support Vector Machine (SVM), and Artificial Neural Networks (ANN) which sklearn has methods to help implement.

#### K-NN
K-NN being an algorithms that is quite simple to implement and to understand, and the fact that it often performs quite well compared to many other algorithms^[2] makes it a quite good candidate for this task. 

When K-NN is used for pattern recognition the algorithm classifies the objects in an image based on the training data

With K-NN we got an accuracy of **XX**% on the test set

#### SVM

#### ANN

### Additional models


## Character Detection
```
h) Test your character detector on detection-1.jpg and detection-2.jpg and show the result in
the report. Feel free to find or create additional images to test your detector, if you are so inclined.
i) Give an evaluation of your detection system. How does it perform?
j) Describe any improvements you made to your detector. Discuss how you can improve your system
further
```

## Conclusion
```
k) What is the weakest and strongest component of your OCR system (feature engineering, character
classification, and character detection)? Explain your answer.
l) What went good/bad with the project? Any lessons learned?
```

## References
**Id, Reference Link, Title, Authors**
[1]: https://scikit-learn.org/stable/index.html "Scikit Learn" - "Scikit"
[2]: http://www.rroij.com/open-access/object-recognition-using-knearest-neighborsupported-by-eigen-value-generated-fromthe-features-of-an-image.php?aid=46808 "Object Recognition Using K-Nearest Neighbor Supported By Eigen Value Generated From the Features of an Image" - "Dr. R.Muralidharan"
[3]: https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf "Histograms of Oriented Gradients for Human Detection" - "Navneet Dalal and Bill Triggs"
