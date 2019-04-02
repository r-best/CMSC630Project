import logging
import numpy as np
from time import time


def getHistogram(self, color=3):
    """Returns a histogram of the Image's pixel values.

    Arguments:
        color (int): Desired color channel(s), see class color constants
    
    Returns:
        ndarray: A numpy array where each array[x] is the number of pixels
            of value x. If multiple color channels are used, they will be
            indexed as array[x][channel].
    """
    t0 = time()

    if self.histogram[color] is None: # If this color hasn't been calculated yet, do it
        if color == self.COLOR_RGB: # RGB is just stack of R, G, and B (three single-channels)
            self.histogram[color] = np.stack((
                self.getHistogram(self.COLOR_RED),
                self.getHistogram(self.COLOR_GREEN),
                self.getHistogram(self.COLOR_BLUE),
            ), axis=1)
        else: # The single-channel histograms (R, G, B, Gray) can be calculated as follows
            self.histogram[color] = np.zeros(256, dtype="uint8")
            for row in self.getMatrix(color):
                for pixel in row:
                    self.histogram[color][pixel] += 1
    
    return self.histogram[color] if not self.timer else (self.histogram[color], time()-t0)

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
        self.matrix[self.COLOR_GRAYSCALE] = np.sum(
            np.multiply(self.getMatrix(color=self.COLOR_RGB), w),
            axis=2, dtype="int")

    return self.matrix[self.COLOR_GRAYSCALE]

def equalize(self, color=3):
    """Returns a new copy of this Image object that has been equalized
    on the desired channel

    Arguments:
        color (int): Desired color channel(s), see class color constants

    Returns:
        Image: A new copy of the original image, equalized according to parameters
    """
    t0 = time()
    logging.info("Equalizing image...")

    B = self.copy()

    # If we want to equalize RGB, equalize R, G, and B separately and remash them into RGB
    if color == self.COLOR_RGB:
        B.matrix[self.COLOR_RED] = self._equalize(B, color=self.COLOR_RED)
        B.matrix[self.COLOR_GREEN] = self._equalize(B, color=self.COLOR_GREEN)
        B.matrix[self.COLOR_BLUE] = self._equalize(B, color=self.COLOR_BLUE)
    # Else we only want a single channel, so just do it & return it
    else:
        B.matrix[color] = self._equalize(B, color)
    
    # Invalidate cached matrices if R, G, or B was edited
    if color in [0,1,2,3]: B.invalidateLazies()

    t1 = time()-t0
    logging.info(f"Done equalizing in {t1}s")
    return B if not self.timer else (B, t1)
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
    cdf = np.zeros(256, dtype="int")
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
    t0 = time()
    logging.info("Quantizing image...")

    if(self.matrix[color] is None):
        self.getMatrix(color)

    # Make a new copy of the Image to work with
    B = self.copy()

    # If we want to quantize RGB, quantize R, G, and B separately and remash them into RGB
    # TODO Parallelize this?
    if color == self.COLOR_RGB:
        B.matrix[self.COLOR_RED] = self._quantize(B, delta, technique, color=self.COLOR_RED)
        B.matrix[self.COLOR_GREEN] = self._quantize(B, delta, technique, color=self.COLOR_GREEN)
        B.matrix[self.COLOR_BLUE] = self._quantize(B, delta, technique, color=self.COLOR_BLUE)
    # Else we only want a single channel, so just do it & return it
    else:
        B.matrix[color] = self._quantize(B, delta, technique, color=color)
    
    # Invalidate cached matrices if R, G, or B was edited
    if color in [0,1,2,3]: B.invalidateLazies()

    t1 = time()-t0
    logging.info(f"Done quantizing in {t1}s...")

    # Calculate mean squared quantization error
    t0_msqe = time()
    pdf = self.getHistogram(color) / 256
    MSQE = 0
    for i in range(len(self.getHistogram(color))):
        MSQE += np.square(self.getHistogram(color)[i] - B.getHistogram(color)[i]) * pdf[i]

    t1_msqe = time()-t0_msqe
    logging.info(f"MSQE of {MSQE} (computed in {t1_msqe}s)")
    return B if not self.timer else (B, t1, t1_msqe)
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
    # Get the list of starting indices of each bucket
    buckets = list(range(0, 256, delta))
    if buckets[-1] != 256: buckets[-1] = 256

    logging.info(f"Quantizing with '{technique}' technique to {len(buckets)-1} color levels")

    # Go through each bucket and reassign the pixels in them to a new value
    # depending on the technique
    for i in range(1, len(buckets)):
        bucket = list(range(buckets[i-1], buckets[i])) # List of all pixel values in current bucket

        if technique == self.QUANT_UNIFORM:
            B.matrix[color][np.where(np.isin(self.matrix[color][:], bucket))] = delta*i + delta/2
        elif technique == self.QUANT_MEAN:
            values = np.array([], dtype="int")
            for index in bucket:
                value = self.getHistogram(color)[index]
                values = np.concatenate((values, np.full((value), value, dtype='int')))
            mean = np.floor(np.mean(values))
            B.matrix[color][np.where(np.isin(self.matrix[color][:], bucket))] = mean
        elif technique == self.QUANT_MEDIAN:
            values = np.array([], dtype="int")
            for index in bucket:
                value = self.getHistogram(color)[index]
                values = np.concatenate((values, np.full((value), value, dtype='int')))
            median = np.floor(np.median(bucket))
            B.matrix[color][np.where(np.isin(self.matrix[color][:], bucket))] = median
    
    B._normalize(color)
    return B.matrix[color]
