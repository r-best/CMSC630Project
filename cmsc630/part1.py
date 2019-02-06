import os
import cv2
import numpy as np

class Image:
    COLOR_RED = 0
    COLOR_GREEN = 1
    COLOR_BLUE = 2
    COLOR_RGB = 3
    COLOR_GRAYSCALE = 4


def loadImages(path, color=Image.COLOR_RGB):
    """Takes in a filepath and loads all of the images
    present in it, recursing down if the file is a directory

    Arguments:
        path (str): The path of the file/directory
        color (const): The desired color channel(s) of the image,
            should be one of the constants in the Image class
    
    Returns:
        list: A list of numpy ndarrays of shape (height, width, 3),
            representing the images present in the directory
    """
    images = list()
    if os.path.isdir(path):
        for f in os.listdir(path):
            images += loadImages(os.path.join(path, f), color=color)
    else:
        flag = (0 if color == Image.COLOR_GRAYSCALE else 1)
        image = cv2.imread(path, flags=flag)
        if image is not None:
            if color != Image.COLOR_GRAYSCALE and color != Image.COLOR_RGB:
                image = image[:, :, color]
            images.append(image)
    return images
