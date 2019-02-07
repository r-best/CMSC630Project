import os
import cv2
import numpy as np

class Image:
    COLOR_RED = 0
    COLOR_GREEN = 1
    COLOR_BLUE = 2
    COLOR_RGB = 3

    def __init__(self, matrix):
        self.matrix = matrix

    def getHistogram(self, color=COLOR_RGB):
        hist = np.zeros((255,3))
        for row in self.matrix:
            for column in row:
                hist[column[0]][0] += 1
                hist[column[1]][1] += 1
                hist[column[2]][2] += 1
        return hist
    
    def asMatrix(self):
        return self.matrix
    
    def getRed(self):
        return self.matrix[:,:,self.COLOR_RED]
    
    def getGreen(self):
        return self.matrix[:,:,self.COLOR_GREEN]
    
    def getBlue(self):
        return self.matrix[:,:,self.COLOR_BLUE]
    
    def getGrayscale(self, luminosity=False):
        """Returns a grayscale version of the RGB image by
        averaging the three color channels. If the `luminosity`
        flag is set, different weights are used instead to better
        reflect human perception, see 
        https://www.tutorialspoint.com/dip/grayscale_to_rgb_conversion.htm

        Arguments:
            luminosity (boolean): Boolean flag to use luminosity weights
        
        Returns:
            ndarray: A single channel matrix of shape (height, width)
                representing the pixel intensities
        """
        if luminosity:
            w = [0.30, 0.59, 0.11]
        else:
            w = [0.33, 0.33, 0.33]

        return np.sum(np.multiply(self.matrix, w), axis=2)


def loadImages(path):
    """Takes in a filepath and loads all of the images
    present in it, recursing down if the file is a directory

    Arguments:
        path (str): The path of the file/directory
    
    Returns:
        list: A list of Image objects created from the images
            in the directory
    """
    images = list()
    if os.path.isdir(path):
        for f in os.listdir(path):
            images += loadImages(os.path.join(path, f))
    else:
        rgb_matrix = cv2.imread(path)
        if rgb_matrix is not None:
            images.append(Image(rgb_matrix))
        else:
            print(f"Error reading file {path}, skipping")
    return images
