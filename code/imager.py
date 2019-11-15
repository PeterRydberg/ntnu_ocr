from PIL import ImageDraw

def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.size[1], stepSize):
        for x in range(0, image.size[0], stepSize):
            yield (x, y, image.crop([x, y, x + windowSize[1], y + windowSize[0]]))

def draw_red_square(x, y, target_image, window):
    draw = ImageDraw.Draw(target_image) 
    draw.rectangle((x,y) + (x + window.size[1], y + window.size[0]), outline="red")
    print(f"X: {x}, Y: {y}, Window: {window.size}")
    target_image.save(f"./dump/img{x}-{y}.png", "PNG")
