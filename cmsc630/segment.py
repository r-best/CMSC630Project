import logging
import numpy as np
from time import time


def kmeans(self, k, epochs=15, color=3):
    """Segments the image into k pixel values using a k-means algorithm

    Arguments:
        k (int): Number of means, i.e. how many groups to segment the image into
        epochs (int): Number of iterations, higher means more accurate (up to a point)
        color (int): Desired color channel(s), see class color constants
    
    Returns:
        A copy of the image segmented into k groups
    """
    B = self.copy()

    if color == self.COLOR_RGB:
        B.matrix[self.COLOR_RED] = _kmeans(B.getMatrix(self.COLOR_RED), k, epochs)
        B.matrix[self.COLOR_GREEN] = _kmeans(B.getMatrix(self.COLOR_GREEN), k, epochs)
        B.matrix[self.COLOR_BLUE] = _kmeans(B.getMatrix(self.COLOR_BLUE), k, epochs)
    else:
        B.matrix[color] = _kmeans(B.getMatrix(color), k, epochs)
    return B
def _kmeans(mat, k, epochs):
    """Helper function to kmeans

    Arguments:
        mat (ndarray): The image matrix to segment
        k (int): Number of means, i.e. how many groups to segment the image into
        epochs (int): Number of iterations, higher means more accurate (up to a point)
    
    Returns:
        The input matrix thresholded into k groups
    """
    centers = np.random.choice(256, size=k) # List of k mean values, randomly assigned
    for _ in range(epochs):
        assignments = list(np.full(k, -1)) # k lists representing the pixels assigned to each mean
        # For each pixel, assign it to its closest mean
        for row in mat:
            for pixel in row:
                closest_center = np.argmin(np.abs(centers-pixel))
                if assignments[closest_center] == -1: assignments[closest_center] = list()
                assignments[closest_center].append(pixel)
        # Find the means that actually had values assigned to them, and update the
        # centers to be the means of those new sets
        changed_indices = [i for i in range(len(assignments)) if assignments[i] != -1]
        for center in changed_indices:
            centers[center] = np.mean(assignments[center])

    # Setting each mean to be the middle point between itself and the next mean transforms
    # our set of means into a set of cutoff points that can be passed to the threshold function
    centers = np.sort(centers)
    for i in range(centers.shape[0]-1):
        centers[i] = (centers[i] + centers[i+1]) / 2
    centers = centers[:-1]

    return _threshold(mat, centers)


def otsu(self, color=3):
    """Segments the image into 2 groups using the Otsu method (see
    https://en.wikipedia.org/wiki/Otsu%27s_method). Calculates the within
    group variance for each possible threshold value and uses the minimum one

    Arguments:
        k (int): Number of means, i.e. how many groups to segment the image into
        epochs (int): Number of iterations, higher means more accurate (up to a point)
        color (int): Desired color channel(s), see class color constants
    
    Returns:
        A copy of the image segmented into k groups
    """
    B = self.copy()

    if color == self.COLOR_RGB:
        B.matrix[self.COLOR_RED] = _otsu(B.getMatrix(self.COLOR_RED), self.getHistogram(self.COLOR_RED))
        B.matrix[self.COLOR_GREEN] = _otsu(B.getMatrix(self.COLOR_GREEN), self.getHistogram(self.COLOR_GREEN))
        B.matrix[self.COLOR_BLUE] = _otsu(B.getMatrix(self.COLOR_BLUE), self.getHistogram(self.COLOR_BLUE))
    else:
        B.matrix[color] = _otsu(B.getMatrix(color), self.getHistogram(color))

    # Invalidate cached matrices if R, G, or B was edited
    if color in [0,1,2,3]: B.invalidateLazies()

    return B
def _otsu(mat, hist):
    """Helper function to otsu

    Arguments:
        mat (ndarray): The image matrix to threshold
        hist (ndarray): The histogram computed from the image matrix
    
    Returns:
        The input matrix thresholded into 2 groups using the otsu method
    """
    wgv = np.zeros(256)
    for i in range(256):
        Po = hist[:i] / 256
        Pb = hist[i:] / 256

        po = np.sum(Po[:i])
        pb = np.sum(Pb[i:])

        if po == 0 or pb == 0:
            wgv[i] = -1
            continue
        
        uo = np.sum(np.multiply(range(Po.shape[0]), Po) / po)
        ub = np.sum(np.multiply(range(Pb.shape[0]), Pb) / pb)

        varo = np.sum(np.multiply(np.square(np.subtract(range(Po.shape[0]), uo)), Po) / po)
        varb = np.sum(np.multiply(np.square(np.subtract(range(Pb.shape[0]), ub)), Pb) / pb)

        wgv[i] = varo*po + varb*pb

    return _threshold(mat, np.argmax(wgv[wgv > -1]))


def threshold(self, levels, color=3):
    """
    """
    B = self.copy()

    if color == self.COLOR_RGB:
        B.matrix[self.COLOR_RED] = _threshold(B.getMatrix(self.COLOR_RED), levels)
        B.matrix[self.COLOR_GREEN] = _threshold(B.getMatrix(self.COLOR_GREEN), levels)
        B.matrix[self.COLOR_BLUE] = _threshold(B.getMatrix(self.COLOR_BLUE), levels)
    else:
        B.matrix[color] = _threshold(B.getMatrix(color), levels)

    # Invalidate cached matrices if R, G, or B was edited
    if color in [0,1,2,3]: B.invalidateLazies()

    return B
def _threshold(b, levels):
    """
    """
    if type(levels) is not list and type(levels) is not np.ndarray:
        levels = [levels]
    levels = np.sort(levels)

    # Anything below first level goes to 0
    b[b < levels[0]] = 0
    # For each level, set everything between it and the next to incrementing i
    for i in range(len(levels)-1):
        b[np.logical_and(
            b >= levels[i],
            b < levels[i+1]
        )] = i+1
    # Anything above last level goes to highest unused i
    b[b >= levels[-1]] = len(levels)

    # Scale the small i values up to fill the whole 0-255 range
    return np.multiply(b, int(255/len(levels)), dtype=np.uint8)
