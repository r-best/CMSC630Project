import os
import cv2
import numpy as np

class Image:
    COLOR_RED = 0
    COLOR_GREEN = 1
    COLOR_BLUE = 2
    COLOR_RGB = 3
    COLOR_GRAYSCALE = 4

    def __init__(self, matrix):
        self.matrix = {
            self.COLOR_RED:         matrix[:,:,self.COLOR_RED],
            self.COLOR_GREEN:       matrix[:,:,self.COLOR_GREEN],
            self.COLOR_BLUE:        matrix[:,:,self.COLOR_BLUE],
            self.COLOR_RGB:         matrix,
            self.COLOR_GRAYSCALE:   None
        }
        self.histogram = {
            self.COLOR_RED:         None,
            self.COLOR_GREEN:       None,
            self.COLOR_BLUE:        None,
            self.COLOR_RGB:         None,
            self.COLOR_GRAYSCALE:   None
        }

    def getHistogram(self, color=self.COLOR_RGB):
        if self.histogram[color] is None: # If this color hasn't been calculated yet, do it
            if color == self.COLOR_RGB: # RGB is just stack of R, G, and B (three single-channels)
                self.histogram[color] = np.stack((
                    self.getHistogram(self.COLOR_RED),
                    self.getHistogram(self.COLOR_GREEN),
                    self.getHistogram(self.COLOR_BLUE),
                ), axis=2)
            else: # The single-channel histograms can be calculated as follows
                self.histogram[color] = np.zeros((255,1))
                for row in self.getMatrix(color):
                    for pixel in row:
                        self.histogram[color][pixel] += 1
        
        return self.histogram[color]

    def getMatrix(self, color=None):
        if color == self.COLOR_GRAYSCALE:
            return self.getGrayscale()
        return self.matrix[color]

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
        if self.matrix[self.COLOR_GRAYSCALE] is None:
            if luminosity:
                w = [0.30, 0.59, 0.11]
            else:
                w = [0.33, 0.33, 0.33]
            self.matrix[self.COLOR_GRAYSCALE] = np.sum(np.multiply(self.matrix[self.COLOR_RGB], w), axis=2)

        return self.matrix[self.COLOR_GRAYSCALE]


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
