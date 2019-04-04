import os
import cv2
import copy
import logging
import numpy as np
from time import time


logging.basicConfig(format="%(levelname)s: %(message)s")

class Image():
    """Represents an image and provides methods for interacting
    with and manipulating it

    The Image is constructed from a numpy ndarray of shape (height, width, 3), the 3
    representing the red, green, and blue channels (the same format used in OpenCV).
    This matrix is split up and stored as the individual colors to more easily
    perform operations on specific channels.
    
    Histograms for each color channel and the RGB and grayscale versions of the image
    are all calculated lazily upon request and then stored to avoid unnecessary
    recalculation. If you manually modify the R, G, or B channels outside of the
    provided class methods you must call the `Image.invalidateLazies()` method to
    reset them or you will continue to receive the old stored versions.

    Class constructor also takes an optional `timer` argument. If set to True, all
    non-getter methods (quantization, equalization, etc..) will return their execution
    time in addition to their usual arguments
    """
    # CLASS COLOR CONSTANTS
    COLOR_RED = 0
    COLOR_GREEN = 1
    COLOR_BLUE = 2
    COLOR_RGB = 3
    COLOR_GRAYSCALE = 4

    # QUANTIZATION TYPES
    QUANT_UNIFORM = "uniform"
    QUANT_MEAN = "mean"
    QUANT_MEDIAN = "median"
    
    # FILTER STRATEGIES
    FILTER_STRAT_LINEAR = "linear"
    FILTER_STRAT_MEAN = "mean"
    FILTER_STRAT_MEDIAN = "median"

    # FILTER BORDER TYPES
    FILTER_BORDER_IGNORE = 'ignore'
    FILTER_BORDER_CROP = 'crop'
    FILTER_BORDER_PAD = 'pad'
    FILTER_BORDER_EXTEND = 'extend'

    def __init__(self, name, matrix, timer=False):
        self.name = name
        self.matrix = [
            np.uint8(matrix[:,:,self.COLOR_RED]),
            np.uint8(matrix[:,:,self.COLOR_GREEN]),
            np.uint8(matrix[:,:,self.COLOR_BLUE]),
            matrix,
            None
        ]
        self.histogram = [None, None, None, None, None]
        self.timer = timer
    
    from .utils import getHistogram, getMatrix, getGrayscale, equalize, _equalize, quantize, _quantize
    from .filters import filter, _filter, sobel, prewitt, _edgeFilter
    from .segment import threshold, _threshold, kmeans
    from .noise import makeGaussianNoise, makeSaltnPepperNoise

    def copy(self):
        """Returns a deep copy of this Image object, including the
        precomupted lazy values

        Returns
            Image: A deep copy of this Image object
        """
        ret = Image(self.name, self.getMatrix(), timer=self.timer)
        ret.matrix = copy.deepcopy(self.matrix)
        ret.histogram = copy.deepcopy(self.histogram)
        return ret

    def invalidateLazies(self):
        """Should be called whenever the Image's R, G, or B matrices
        are altered, deletes the precomputed values that are dependent
        on them so they can be recalculated next time they are requested.
        """
        self.matrix[self.COLOR_RGB] = None
        self.matrix[self.COLOR_GRAYSCALE] = None
        self.histogram = [None, None, None, None, None]

    def _normalize(self, color=3):
        """Checks to make sure the values of the color channel are in the range [0-255]
        and adjusts them if they're not. Called by the other functions of this class
        when they edit matrices in a way that could produce values outside of the range.

        Arguments:
            color (int): Desired color channel(s), see class color constants
        """
        if color == self.COLOR_RGB:
            color = self.COLOR_RED
            self._normalize(color=self.COLOR_GREEN)
            self._normalize(color=self.COLOR_BLUE)

        # Normalize pixel values, make sure they're all within the 0-255 range
        if np.min(self.matrix[color]) < 0:
            self.matrix[color] = np.add(self.matrix[color], -1*np.min(self.matrix[color]))
        if np.max(self.matrix[color]) > 255:
            scale = 255/np.max(self.matrix[color])
            self.matrix[color] = np.multiply(self.matrix[color], scale)

    def saveToFile(self, dir, name=None, color=3):
        """Takes in a directory path and saves the Image to that directory

        Arguments:
            dir (str): Directory to save the file to
            name (str): The name of the file to save to, defaults to the
                name of the file this Image was loaded from
        
        Returns:
            (boolean): Whether or not the file was saved correctly
        """
        path = os.path.join(dir, self.name if name is None else name)
        if not os.path.exists(dir):
            logging.warn(f"Directory '{dir}' does not exist, creating it and continuing")
            os.mkdir(dir)
        if not os.path.isdir(dir):
            logging.error(f"Cannot save image: '{dir}' is a file")
            return False
        try:
            output = np.uint8(self.getMatrix(color))

            if color != Image.COLOR_GRAYSCALE:
                zero = np.zeros((self.getMatrix().shape[:2]), dtype="uint8")
                if color == Image.COLOR_RED:        output = np.stack((output, zero, zero), axis=2)
                elif color == Image.COLOR_GREEN:    output = np.stack((zero, output, zero), axis=2)
                elif color == Image.COLOR_BLUE:     output = np.stack((zero, zero, output), axis=2)
                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            cv2.imwrite(path, output)
            return True
        except Exception as e:
            logging.fatal(f"Error saving file: {e}")
            return False

    @staticmethod
    def fromFile(path, timer=False):
        """Takes in a path to an image file and creates a new
        Image object from it. If the path is a directory, calls
        `Image.fromDir()` instead and returns its results.

        Arguments:
            path (str): The path to the desired image
            timer (boolean): Whether or not to have image processing
                methods of this object return their run time
        
        Returns:
            An Image if the path is a file, a list of Images if
            the path is a directory, or None if the path is invalid
        """
        if os.path.isdir(path):
            return Image.fromDir(path, timer=timer)
        
        rgb_matrix = cv2.imread(path)
        if rgb_matrix is not None:
            rgb_matrix = cv2.cvtColor(rgb_matrix, cv2.COLOR_BGR2RGB)
            return Image(os.path.basename(path), rgb_matrix, timer=timer)
        else:
            logging.error(f"Error reading file {path}")
            return None


    @staticmethod
    def fromDir(path, timer=False):
        """Takes in a filepath and loads it as an image,
        recursing down if the path is a directory

        Arguments:
            path (str): The path of the file/directory
            timer (boolean): Whether or not to have image processing
                methods of this object return their run time
        
        Returns:
            list: A list of new Image objects
        """
        images = list()
        if os.path.isdir(path):
            for f in os.listdir(path):
                images += Image.fromDir(os.path.join(path, f), timer=timer)
        else:
            rgb_matrix = cv2.imread(path)
            if rgb_matrix is not None:
                rgb_matrix = cv2.cvtColor(rgb_matrix, cv2.COLOR_BGR2RGB)
                images.append(Image(os.path.basename(path), rgb_matrix, timer=timer))
            else:
                logging.error(f"Error reading file {path}, skipping")
        return images
