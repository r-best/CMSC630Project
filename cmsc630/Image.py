import os
import cv2
import numpy as np

class Image:
    """Represents an image and provides methods for interacting
    with and manipulating it
    """
    # CLASS CONSTANTS
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
        """Returns a histogram of the Image's pixel values.

        Arguments:
            color (int): Desired color channel(s), see class color constants
        
        Returns:
            ndarray: A numpy array where each array[x] is the number of pixels
                of value x. If multiple color channels are used, they will be
                indexed as array[x][channel].
        """
        if self.histogram[color] is None: # If this color hasn't been calculated yet, do it
            if color == self.COLOR_RGB: # RGB is just stack of R, G, and B (three single-channels)
                self.histogram[color] = np.stack((
                    self.getHistogram(self.COLOR_RED),
                    self.getHistogram(self.COLOR_GREEN),
                    self.getHistogram(self.COLOR_BLUE),
                ), axis=2)
            else: # The single-channel histograms (R, G, B, Gray) can be calculated as follows
                self.histogram[color] = np.zeros((255,1))
                for row in self.getMatrix(color):
                    for pixel in row:
                        self.histogram[color][pixel] += 1
        
        return self.histogram[color]

    def getMatrix(self, color=None):
        """Returns the Image's pixel matrix with
        the requested color channels

        Arguments:
            color (int): Desired color channel(s), see class color constants
        
        Returns:
            ndarray: A matrix of shape (height, width, num_channels)
                representing the pixels of the Image
        """
            
        if color == self.COLOR_GRAYSCALE:
            return self.getGrayscale()
        return self.matrix[color]

    def getGrayscale(self, luminosity=False, force=False):
        """Returns a grayscale version of the RGB image by
        averaging the three color channels. If the `luminosity`
        flag is set, different weights are used instead to better
        reflect human perception, see 
        https://www.tutorialspoint.com/dip/grayscale_to_rgb_conversion.htm.
        On subsequent calls the matrix calculated in the first call is
        returned again unless the `force` flag is set to save on computation
        time, you should only need to set `force` if you wish to recalculate
        the matrix with a different value of `luminosity`.

        Arguments:
            luminosity (boolean): Boolean flag to use luminosity weights
            force (boolean): Overrides lazy loading to recalculate grayscale
                matrix regardless of if one is already present in memory
        
        Returns:
            ndarray: A single channel matrix of shape (height, width)
                representing the pixel intensities
        """
        if self.matrix[self.COLOR_GRAYSCALE] is None or force:
            if luminosity:
                w = [0.30, 0.59, 0.11]
            else:
                w = [0.33, 0.33, 0.33]
            self.matrix[self.COLOR_GRAYSCALE] = np.sum(np.multiply(self.matrix[self.COLOR_RGB], w), axis=2)

        return self.matrix[self.COLOR_GRAYSCALE]

    @staticmethod
    def fromFile(path):
        """Takes in a filepath and loads it as an image,
        recursing down if the path is a directory

        Arguments:
            path (str): The path of the file/directory
        
        Returns:
            list: A list of new Image objects
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
