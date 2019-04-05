import logging
import numpy as np
from time import time
import pathos.pools as pp


def filter(self, filter=None, strategy='linear', border='ignore', normalize=True, color=3):
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
        B.matrix[self.COLOR_RED] = self._filter(B, filter, strategy, border, normalize, color=self.COLOR_RED)
        B.matrix[self.COLOR_GREEN] = self._filter(B, filter, strategy, border, normalize, color=self.COLOR_GREEN)
        B.matrix[self.COLOR_BLUE] = self._filter(B, filter, strategy, border, normalize, color=self.COLOR_BLUE)
    # Else we only want a single channel, so just do it & return it
    else:
        B.matrix[color] = self._filter(B, filter, strategy, border, normalize, color=color)
    
    # Invalidate cached matrices if R, G, or B was edited
    if color in [0,1,2,3]: B.invalidateLazies()

    t1 = time()-t0
    logging.info(f"Done filtering in {t1}s")
    return B if not self.timer else (B, t1)
def _filter(self, B, filter, strategy, border, normalize, color):
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
    Bmat = np.int64(B.getMatrix(color))

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

    if normalize:
        B._normalize(color)
        Bmat = np.uint8(Bmat)

    return Bmat

def laplace(self, color=3):
    """
    """
    s = np.array([
        [0,  1,  0],
        [1, -4,  1],
        [0,  1,  0]
    ])

    B = self.filter(s, color=color)
    
    if color == self.COLOR_RGB:
        B.matrix[0] = self.getMatrix(0) - B.getMatrix(0)
        B.matrix[1] = self.getMatrix(1) - B.getMatrix(1)
        B.matrix[2] = self.getMatrix(2) - B.getMatrix(2)
    else:
        B.matrix[color] = self.getMatrix(color) - B.getMatrix(color)
    
    return B

def sobel(self, dx, dy, color=3):
    """
    """
    filt_x = np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]
    ])
    filt_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])
    return self._gradient(filt_x, filt_y, dx, dy, color=color)

def prewitt(self, dx, dy, color=3):
    """
    """
    filt_x = np.array([
        [-1,  0,  1],
        [-1,  0,  1],
        [-1,  0,  1]
    ])
    filt_y = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ])
    return self._gradient(filt_x, filt_y, dx, dy, color=color)

def _gradient(self, filt_x, filt_y, dx, dy, color=3):
    """https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
    """
    def process_mat(x, y):
        # Gradient and edge direction calculation
        grad = np.sqrt(np.square(x) + np.square(y))

        directions = np.arctan2(y, x) * 180 / np.pi
        directions[directions<0] += 180
        
        directions[np.logical_or(
            np.logical_and(directions>=0, directions<22.5),
            np.logical_and(directions>=157.5, directions<=180)
        )] = 0 # E-W edges (0)
        directions[np.logical_and(directions>=22.5, directions<67.5)] = 1 # SW-NE edges (45)
        directions[np.logical_and(directions>=67.5, directions<112.5)] = 2 # N-S edges (90)
        directions[np.logical_and(directions>=112.5, directions<157.5)] = 3 # NW-SE edges (135)

        return np.uint8(grad), directions

    if dx != 0: x = self.filter(filt_x*dx, normalize=False, color=color)
    if dy != 0: y = self.filter(filt_y*dy, normalize=False, color=color)

    if dx != 0 and dy != 0:
        if color == self.COLOR_RGB:
            directions = [-1, -1, -1]
            for i in range(3):
                x.matrix[i], directions[i] = process_mat(x.matrix[i], y.matrix[i])
        else:
            x.matrix[color], directions = process_mat(x.matrix[color], y.matrix[color])
        return x, directions
    elif dx != 0:
        return x, None
    else:
        return y, None

def canny(self, minEdge=100, maxEdge=200, color=3):
    """
    """
    # Gradient and edge direction calculation
    B, directions = self.sobel(1, 1, color=color)

    C = B.copy()

    # Non-maximum suppression
    mat = B.getMatrix(color)
    for i in range(1, mat.shape[0]-1):
        for j in range(1, mat.shape[1]-1):
            if directions[i,j] == 0: # E-W edges
                argmax = np.argmax([mat[i,j], mat[i,j-1], mat[i,j+1]])
            elif directions[i,j] == 1: # SW-NE edges
                argmax = np.argmax([mat[i,j], mat[i-1,j+1], mat[i+1,j-1]])
            elif directions[i,j] == 2: # N-S edges
                argmax = np.argmax([mat[i,j], mat[i-1,j], mat[i+1,j]])
            elif directions[i,j] == 3: # NW-SE edges
                argmax = np.argmax([mat[i,j], mat[i-1,j-1], mat[i+1,j+1]])

            if argmax != 0:
                C.matrix[color][i,j] = 0
    
    D = C.copy()
    
    # Hysteresis thresholding
    C.matrix[color][C.matrix[color]<=minEdge] = 0
    C.matrix[color][C.matrix[color]>=maxEdge] = 255
    D.matrix[color][(0,-1),:] = 0
    D.matrix[color][:,(0,-1)] = 0
    for i in range(1, mat.shape[0]-1):
        for j in range(1, mat.shape[1]-1):
            if np.max(C.matrix[color][i-1:i+2,j-1:j+2]) == 255:
                D.matrix[color][i,j] = 255
            else:
                D.matrix[color][i,j] = 0
    print(np.unique(D.matrix[color]))

    return D
