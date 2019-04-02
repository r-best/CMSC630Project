import logging
import numpy as np
from time import time
import pathos.pools as pp


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
