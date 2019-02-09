import os
import cv2
import copy
import numpy as np

class Image:
    """Represents an image and provides methods for interacting
    with and manipulating it
    """
    # CLASS COLOR CONSTANTS
    COLOR_RED = 0
    COLOR_GREEN = 1
    COLOR_BLUE = 2
    COLOR_RGB = 3
    COLOR_GRAYSCALE = 4

    # QUANTIZATION TYPES
    QUANT_UNIFORM = "uniform"
    QUANT_MEDIAN = "median"
    QUANT_MEAN = "mean"
    
    # CLASS FILTER CONSTANTS
    FILTER_LINEAR = "linear"
    FILTER_MEDIAN = "median"

    def __init__(self, matrix):
        self.matrix = [
            matrix[:,:,self.COLOR_RED],
            matrix[:,:,self.COLOR_GREEN],
            matrix[:,:,self.COLOR_BLUE],
            matrix,
            None
        ]
        self.histogram = [None, None, None, None, None]
    
    def copy(self):
        ret = Image(self.getMatrix())
        ret.matrix = copy.deepcopy(self.matrix)
        ret.histogram = copy.deepcopy(self.histogram)
        return ret

    def getHistogram(self, color=3):
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
                self.histogram[color] = np.zeros((255,1), dtype="int")
                for row in self.getMatrix(color):
                    for pixel in row:
                        self.histogram[color][pixel] += 1
        
        return self.histogram[color]

    def getMatrix(self, color=3):
        """Returns the Image's pixel matrix with
        the requested color channels

        Arguments:
            color (int): Desired color channel(s), see class color constants
        
        Returns:
            ndarray: A matrix of shape (height, width, num_channels)
                representing the pixels of the Image
        """
        if self.matrix[color] is None:
            if color == self.COLOR_GRAYSCALE:
                return self.getGrayscale()
            if color == self.COLOR_RGB:
                self.matrix[color] = np.stack((
                    self.getMatrix(self.COLOR_RED),
                    self.getMatrix(self.COLOR_GREEN),
                    self.getMatrix(self.COLOR_BLUE),
                ), axis=2)
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
            self.matrix[self.COLOR_GRAYSCALE] = np.sum(np.multiply(self.matrix[self.COLOR_RGB], w), axis=2, dtype="int")

        return self.matrix[self.COLOR_GRAYSCALE]
    
    def makeSaltnPepperNoise(self):
        """
        """
        # Add salt & pepper noise to R, G, and B
        return
    
    def makeGaussianNoise(self):
        """
        """
        # Add gaussian noise to R, G, and B
        return
    
    def equalize(self, color=3):
        """

        Returns:
            Image: A new Image object that has been equalized
        """
        return
    
    def quantize(self, delta=16, technique='uniform', color=3):
        """Returns a new copy of this Image object that has been quantized
        on the desired channel

        Arguments:
            delta (int): The quantization level, i.e. max number of color values
                that will be present in the resulting image
            technique: Desired quantization technique, see class quantization constants
            color (int): Desired color channel(s), see class color constants
        
        Returns:
            Image: A new copy of the original image, quantized according to parameters
        """
        if(self.matrix[color] is None):
            self.getMatrix(color)

        ret = self.copy()

        if color == self.COLOR_RGB:
            ret.matrix[self.COLOR_RED] = self._quantize(ret, delta, technique, color=self.COLOR_RED)
            ret.matrix[self.COLOR_GREEN] = self._quantize(ret, delta, technique, color=self.COLOR_GREEN)
            ret.matrix[self.COLOR_BLUE] = self._quantize(ret, delta, technique, color=self.COLOR_BLUE)
            ret.matrix[self.COLOR_RGB] = np.stack((
                    ret.getMatrix(self.COLOR_RED),
                    ret.getMatrix(self.COLOR_GREEN),
                    ret.getMatrix(self.COLOR_BLUE),
                ), axis=2)
        else:
            ret.matrix[color] = self._quantize(ret, delta, technique, color=color)

        return ret
    def _quantize(self, B, delta, technique, color):
        """Helper function for `Image.quantize()`, performs the math to quantize a
        single color channel

        Arguments:
            B (Image): The new copy Image produced in `Image.quantize()`
            delta (int): The quantization level, i.e. max number of color values
                that will be present in the resulting image
            technique: Desired quantization technique, see class quantization constants
            color (int): Desired color channel(s), see class color constants
        
        Returns:
            ndarray: The color channel of B that has been quantized
        """
        # max_peak = np.max(ret.getHistogram(color))
        # levels = list(range(0, max_peak+1, int(max_peak/delta)))
        # if levels[-1] != max_peak: levels[-1] = max_peak
        # print(len(levels), levels, max_peak)

        buckets = list(range(0, 256, int(255/delta)))
        if buckets[-1] != 255: buckets[-1] = 255
        for i, _ in enumerate(buckets):
            if i == 0: continue

            bucket = list(range(buckets[i-1], buckets[i]))
            B.matrix[color][np.where(np.isin(self.matrix[color][:], bucket))] = delta*i
        return B.matrix[color]

    
    def applyFilter(self, filter=None):
        """
        """
        if filter is None:
            return "Please specify a filter to use"
        return

    @staticmethod
    def fromDir(path):
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
