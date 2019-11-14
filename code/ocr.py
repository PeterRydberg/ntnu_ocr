from PIL import Image, ImageDraw
from imager import sliding_window


img = Image.open("../dataset/detection-images/detection-1.jpg")
for (x, y, window) in sliding_window(image=img, stepSize=8, windowSize=(20, 20)):
    newImg = img.copy()
    draw = ImageDraw.Draw(newImg) # Creates a copy of the image and draws on it
    draw.rectangle((x,y) + (x + window.size[1], y + window.size[0]), fill=128)
    print(f"X: {x}, Y: {y}, Window: {window.size}")
    newImg.save(f"img{x}-{y}.png", "PNG")
