from PIL import ImageDraw, ImageColor

def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.size[1], stepSize):
        for x in range(0, image.size[0], stepSize):
            yield (x, y, image.crop([x, y, x + windowSize[1], y + windowSize[0]]))

def draw_red_square(x, y, target_image):
    draw = ImageDraw.Draw(target_image) 
    draw.rectangle((x,y) + (x + 20, y + 20), outline="#ff0000")
    # target_image.save("./dump/img1.png", "PNG")
    return target_image

def draw_grey_square(x, y, target_image, window):
    draw = ImageDraw.Draw(target_image)
    draw.rectangle((x,y) + (x + window.size[1], y + window.size[0]), outline="#333333")
    return target_image