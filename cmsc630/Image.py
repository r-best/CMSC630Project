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
    QUANT_MEAN = "mean"
    QUANT_MEDIAN = "median"
    
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
        """Returns a new copy of this Image object that has been equalized
        on the desired channel

        Arguments:
            color (int): Desired color channel(s), see class color constants

        Returns:
            Image: A new copy of the original image, equalized according to parameters
        """
        B = self.copy()

        # If we want to equalize RGB, equalize R, G, and B separately and remash them into RGB
        # TODO Parallelize this?
        if color == self.COLOR_RGB:
            B.matrix[self.COLOR_RED] = self._equalize(B, color=self.COLOR_RED)
            B.matrix[self.COLOR_GREEN] = self._equalize(B, color=self.COLOR_GREEN)
            B.matrix[self.COLOR_BLUE] = self._equalize(B, color=self.COLOR_BLUE)
            B.matrix[self.COLOR_RGB] = np.stack((
                    B.getMatrix(self.COLOR_RED),
                    B.getMatrix(self.COLOR_GREEN),
                    B.getMatrix(self.COLOR_BLUE),
                ), axis=2)
        # Else we only want a single channel, so just do it & return it
        else:
            B.matrix[color] = self._equalize(B, color)

        return B
    def _equalize(self, B, color):
        """Helper function for `Image.equalize()`, performs the math to equalize a
        single color channel using the algorithm described at
        https://www.tutorialspoint.com/dip/histogram_equalization.htm

        Arguments:
            B (Image): The new copy Image produced in `Image.equalize()`
            color (int): Desired color channel(s), see class color constants
        
        Returns:
            ndarray: The color channel of B that has been equalized
        """
        cdf = np.zeros(255, dtype="int")
        num_pixels = np.sum(self.getHistogram(color))
        total = 0
        for i, level in enumerate(self.getHistogram(color)):
            total += level/num_pixels
            cdf[i] = int(total * 255)
        
        return cdf[B.getMatrix(color)]
    
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

        # Make a new copy of the Image to work with
        ret = self.copy()

        # If we want to quantize RGB, quantize R, G, and B separately and remash them into RGB
        # TODO Parallelize this?
        if color == self.COLOR_RGB:
            ret.matrix[self.COLOR_RED] = self._quantize(ret, delta, technique, color=self.COLOR_RED)
            ret.matrix[self.COLOR_GREEN] = self._quantize(ret, delta, technique, color=self.COLOR_GREEN)
            ret.matrix[self.COLOR_BLUE] = self._quantize(ret, delta, technique, color=self.COLOR_BLUE)
            ret.matrix[self.COLOR_RGB] = np.stack((
                    ret.getMatrix(self.COLOR_RED),
                    ret.getMatrix(self.COLOR_GREEN),
                    ret.getMatrix(self.COLOR_BLUE),
                ), axis=2)
        # Else we only want a single channel, so just do it & return it
        else:
            ret.matrix[color] = self._quantize(ret, delta, technique, color=color)

        return ret
    def _quantize(self, B, delta, technique, color):
        """Helper function for `Image.quantize()`, performs the math to quantize a
        single color channel

        Arguments:
            B (Image): The new copy Image produced in `Image.quantize()`
            delta (int): The step size of the quantizer, i.e. the number of pixel values in each 'bucket'
            technique: Desired quantization technique, see class quantization constants
            color (int): Desired color channel(s), see class color constants
        
        Returns:
            ndarray: The color channel of B that has been quantized
        """
        print(f"Quantizing with '{technique}' technique to {int(256/delta)} color levels")
        # Get the list of starting indices of each bucket
        buckets = list(range(0, 256, delta))
        if buckets[-1] != 255: buckets.append(255)
        
        # Go through each bucket and reassign the pixels in them to a new value
        # depending on the technique
        for i, _ in enumerate(buckets):
            if i == 0: continue

            bucket = list(range(buckets[i-1], buckets[i])) # List of all pixel values in current bucket
            
            if technique == self.QUANT_UNIFORM:
                B.matrix[color][np.where(np.isin(self.matrix[color][:], bucket))] = delta*i + delta/2
            elif technique == self.QUANT_MEAN:
                values = np.array([], dtype="int")
                for index in bucket:
                    value = self.getHistogram(color)[index][0]
                    values = np.concatenate((values, np.full((value), value, dtype='int')))
                mean = np.floor(np.mean(values))
                B.matrix[color][np.where(np.isin(self.matrix[color][:], bucket))] = mean
            elif technique == self.QUANT_MEDIAN:
                values = np.array([], dtype="int")
                for index in bucket:
                    value = self.getHistogram(color)[index][0]
                    values = np.concatenate((values, np.full((value), value, dtype='int')))
                median = np.floor(np.median(bucket))
                B.matrix[color][np.where(np.isin(self.matrix[color][:], bucket))] = median

        return B.matrix[color]

    
    def applyFilter(self, filter=None, color=3):
        """Takes in a filter and applies it to each pixel of the Image, producing a new Image
        as output. The filter must be a square 2D array-like of odd degree (i.e. has a clear
        center pixel). However, non-square filters can be acheived by simply setting the entries
        you don't need to zero.

        Filter Examples:
            >[[ 0 0 1 0 0 ]
            > [ 0 0 1 0 0 ]     5x5 filter that emphasizes the
            > [ 0 0 1 0 0 ]     vertical and ignores everything else
            > [ 0 0 1 0 0 ]
            > [ 0 0 1 0 0 ]]

            >[[ 2 0 1 ]
            > [ 0 1 0 ]         3x3 filter that emphasizes diagonals
            > [ 2 0 1 ]]        with a higher preference to those on the left

            >[[ -1 -1 -1 ]
            > [ -1  5 -1 ]      3x3 filter that penalizes everything but the
            > [ -1 -1 -1 ]]     center pixel, which it emphasizes heavily

        Arguments:
            filter: A 2D square array-like of weights to pass over the image pixel by pixel
            color (int): Desired color channel(s), see class color constants
        
        Returns:
            A new copy of this Image object with the filter applied to the desired color channel(s)
        """
        if filter is None:
            return "Please specify a filter to use"

        filter = np.array(filter)

        if filter.shape[0] % 2 == 0 or filter.shape[1] % 2 == 0:
            return "Filter must have a clear center, if you want a non-square filter try padding it with zeroes"

        B = self.copy()

        # If we want to apply filter to RGB, process R, G, and B separately and remash them into RGB
        # TODO Parallelize this?
        if color == self.COLOR_RGB:
            B.matrix[self.COLOR_RED] = self._applyFilter(B, filter, color=self.COLOR_RED)
            B.matrix[self.COLOR_GREEN] = self._applyFilter(B, filter, color=self.COLOR_GREEN)
            B.matrix[self.COLOR_BLUE] = self._applyFilter(B, filter, color=self.COLOR_BLUE)
            B.matrix[self.COLOR_RGB] = np.stack((
                    B.getMatrix(self.COLOR_RED),
                    B.getMatrix(self.COLOR_GREEN),
                    B.getMatrix(self.COLOR_BLUE),
                ), axis=2)
        # Else we only want a single channel, so just do it & return it
        else:
            B.matrix[color] = self._applyFilter(B, filter, color=color)

        return B
    def _applyFilter(self, B, filter, color):
        """Helper function for `Image.applyFilter()`, performs the math to apply the filter to a
        single color channel

        Arguments:
            B (Image): The new copy Image produced in `Image.quantize()`
            filter (ndarray): A 2D square ndarray of weights to pass over the image pixel by pixel
            color (int): Desired color channel(s), see class color constants
        
        Returns:
            ndarray: The color channel of B that has been altered
        """
        width_left = int(filter.shape[0]/2)
        width_right = filter.shape[1] - width_left - 1
        height_top = int(filter.shape[1]/2)
        height_bottom = filter.shape[0] - height_top - 1
        
        mat = self.getMatrix(color)
        for i in range(height_top, mat.shape[0]-height_bottom):
            for j in range(width_left, mat.shape[1]-width_right):
                B.matrix[color][i,j] = np.sum(np.multiply(
                    self.matrix[color][i-width_left:i+width_right+1, j-height_top:j+height_top+1],
                    filter
                ))
                if B.matrix[color][i,j] < 0:
                    np.add(B.matrix[color], -1*np.min(B.matrix[color]))
                if B.matrix[color][i,j] > 255:
                    scale = 255/np.max(B.matrix[color])
                    np.multiply(B.matrix[color], scale)

        return B.matrix[color]

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
            rgb_matrix = cv2.cvtColor(rgb_matrix, cv2.COLOR_BGR2RGB)
            if rgb_matrix is not None:
                images.append(Image(rgb_matrix))
            else:
                print(f"Error reading file {path}, skipping")
        return images
