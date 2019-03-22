import os
import cv2
import copy
import logging
import numpy as np
from time import time
import pathos.pools as pp

logging.basicConfig(format="%(levelname)s: %(message)s")

class Image:
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
            matrix[:,:,self.COLOR_RED],
            matrix[:,:,self.COLOR_GREEN],
            matrix[:,:,self.COLOR_BLUE],
            matrix,
            None
        ]
        self.histogram = [None, None, None, None, None]
        self.timer = timer
    
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

    
    def filter(self, filter=None, strategy='linear', border='ignore', color=3):
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
            strategy: The strategy to use when applying the filter, either 'linear' or 'median',
                see class filter constants
        
        Returns:
            A new copy of this Image object with the filter applied to the desired color channel(s)
        """
        t0 = time()
        logging.info("Filtering image...")

        if filter is None:
            return "Please specify a filter to use"

        filter = np.array(filter)

        if filter.shape[0] % 2 == 0 or filter.shape[1] % 2 == 0:
            return "Filter must have a clear center, if you want a non-square filter try padding it with zeroes"

        B = self.copy()

        # If we want to apply filter to RGB, process R, G, and B separately and remash them into RGB
        # TODO Parallelize this?
        if color == self.COLOR_RGB:
            B.matrix[self.COLOR_RED] = self._filter(B, filter, strategy, border, color=self.COLOR_RED)
            B.matrix[self.COLOR_GREEN] = self._filter(B, filter, strategy, border, color=self.COLOR_GREEN)
            B.matrix[self.COLOR_BLUE] = self._filter(B, filter, strategy, border, color=self.COLOR_BLUE)
        # Else we only want a single channel, so just do it & return it
        else:
            B.matrix[color] = self._filter(B, filter, strategy, border, color=color)
        
        # Invalidate cached matrices if R, G, or B was edited
        if color in [0,1,2,3]: B.invalidateLazies()

        t1 = time()-t0
        logging.info(f"Done filtering in {t1}s")
        return B if not self.timer else (B, t1)
    def _filter(self, B, filter, strategy, border, color):
        """Helper function for `Image.filter()`, performs the math to apply the filter to a
        single color channel

        Arguments:
            B (Image): The new copy Image produced in `Image.filter()`
            filter (ndarray): A 2D square ndarray of weights to pass over the image pixel by pixel
            color (int): Desired color channel(s), see class color constants
            strategy: The strategy to use when applying the filter, either 'linear' or 'median',
                see class filter constants
        
        Returns:
            ndarray: The color channel of B that has been altered
        """
        width_left = int(filter.shape[0]/2)
        width_right = filter.shape[1] - width_left - 1
        height_top = int(filter.shape[1]/2)
        height_bottom = filter.shape[0] - height_top - 1

        mat = self.getMatrix(color)
        Bmat = B.getMatrix(color)

        def process_row(i):
            row = Bmat[i]
            for j in range(width_left, mat.shape[1]-width_right):
                weighted = np.multiply(
                    mat[i-height_top:i+height_bottom+1, j-width_left:j+width_right+1],
                    filter
                )

                if strategy == self.FILTER_STRAT_LINEAR:
                    row[j] = np.sum(weighted)
                elif strategy == self.FILTER_STRAT_MEAN:
                    row[j] = np.mean(weighted)
                elif strategy == self.FILTER_STRAT_MEDIAN:
                    row[j] = np.median(weighted)
            return row

        p = pp.ProcessPool()
        filtered = np.array(
            p.map(process_row, range(height_top, mat.shape[0]-height_bottom))
        )

        # If border type is 'ignore', just return the original borders of the image
        if border == self.FILTER_BORDER_IGNORE:
            Bmat = np.vstack((
                Bmat[0:height_top],
                filtered,
                Bmat[mat.shape[0]-height_bottom:]
            ))
        # If border type is 'crop', remove the borders and return a smaller image
        elif border == self.FILTER_BORDER_CROP:
            Bmat = filtered[:,width_right:-width_right]
        # If border type is 'pad', just replace the border with zeros
        elif border == self.FILTER_BORDER_PAD:
            filtered[:, :width_left] = 0
            filtered[:, -width_right:] = 0
            Bmat = np.vstack((
                np.zeros((height_top, filtered.shape[1]), dtype='int'),
                filtered,
                np.zeros((height_bottom, filtered.shape[1]), dtype='int')
            ))
        # If border type is 'extend', take the outermost pixels in the filtered area and
        # extend them out into the space the border should occupy
        elif border == self.FILTER_BORDER_EXTEND:
            filtered[:,:width_left] = np.tile(filtered[:,width_left], (width_left, 1)).T
            filtered[:,-width_right:] = np.tile(filtered[:,-width_right-1], (width_right, 1)).T
            Bmat = np.vstack((
                np.tile(filtered[0], (height_top, 1)),
                filtered,
                np.tile(filtered[-1], (height_bottom, 1))
            ))

        B._normalize(color)

        return Bmat
    
    def makeSaltnPepperNoise(self, rate=0.30, color=3):
        """Adds noise to the image by giving each pixel a {rate}% chance
        to have its value changed to either 0 or 255, resulting in black
        and white spots all over the image

        Arguments:
            rate (float): A number in the range [0-1] representing each pixel's
                chance of being corrupted
            color (int): Desired color channel(s), see class color constants
        
        Returns:
            Image: A new copy of the original image, corrupted with salt & pepper noise
        """
        t0 = time()
        logging.info("Adding salt&pepper noise to image...")

        B = self.copy()

        for i in range(B.matrix[color].shape[0]):
            for j in range(B.matrix[color].shape[1]):
                if np.random.uniform(0, 1) <= rate:
                    new_value = np.random.choice([0, 255])
                    B.matrix[color][i][j] = new_value
                    if color == self.COLOR_RGB:
                        B.matrix[self.COLOR_RED][i][j] = new_value
                        B.matrix[self.COLOR_GREEN][i][j] = new_value
                        B.matrix[self.COLOR_BLUE][i][j] = new_value
        
        # Invalidate cached matrices if R, G, or B was edited
        if color in [0,1,2,3]: B.invalidateLazies()

        t1 = time()-t0
        logging.info(f"Done making noisy in {t1}s")
        return B if not self.timer else (B, t1)
    
    def makeGaussianNoise(self, rate=0.30, mean=None, stddev=None, color=3):
        """Adds noise to the image by giving each pixel a {rate}% chance
        to have its value changed to a random value generated from a normal
        distribution with the given mean and standard deviation. If mean and
        standard deviation are not provided they will be calculated from the image

        Arguments:
            rate (float): A number in the range [0-1] representing each pixel's
                chance of being corrupted
            mean (float): The mean of the gaussian distribution
            stddev (float): The standard deviation of the gaussian distribution
            color (int): Desired color channel(s), see class color constants
        
        Returns:
            Image: A new copy of the original image, corrupted with salt & pepper noise
        """
        t0 = time()
        logging.info("Adding gaussian noise to image...")

        B = self.copy()

        if mean is None: mean = np.mean(B.getMatrix(color))
        if stddev is None: stddev = np.std(B.getMatrix(color))

        for i in range(B.matrix[color].shape[0]):
            for j in range(B.matrix[color].shape[1]):
                if np.random.uniform(0, 1) <= rate:
                    new_value = np.random.normal(loc=mean, scale=stddev)
                    B.matrix[color][i][j] = new_value
                    if color == self.COLOR_RGB:
                        B.matrix[self.COLOR_RED][i][j] = new_value
                        B.matrix[self.COLOR_GREEN][i][j] = new_value
                        B.matrix[self.COLOR_BLUE][i][j] = new_value
        
        B._normalize(color)
        
        # Invalidate cached matrices if R, G, or B was edited
        if color in [0,1,2,3]: B.invalidateLazies()

        t1 = time()-t0
        logging.info(f"Done making noisy in {t1}s")
        return B if not self.timer else (B, t1)

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
