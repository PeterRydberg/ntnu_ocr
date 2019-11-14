def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.size[1], stepSize):
        for x in range(0, image.size[0], stepSize):
            yield (x, y, image.crop([x, y, x + windowSize[1], y + windowSize[0]]))
