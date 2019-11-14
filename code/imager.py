def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.size[1], stepSize):
        for x in range(0, image.shape[0], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
