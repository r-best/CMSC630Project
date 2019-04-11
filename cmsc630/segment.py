import logging
import numpy as np
from time import time


def kmeans(self, k, epochs=15, color=3):
    """
    """
    B = self.copy()

    if color == self.COLOR_RGB:
        B.matrix[self.COLOR_RED] = _kmeans(B.getMatrix(self.COLOR_RED), k, epochs, self.COLOR_RED)
        B.matrix[self.COLOR_GREEN] = _kmeans(B.getMatrix(self.COLOR_GREEN), k, epochs, self.COLOR_GREEN)
        B.matrix[self.COLOR_BLUE] = _kmeans(B.getMatrix(self.COLOR_BLUE), k, epochs, self.COLOR_BLUE)
    else:
        B.matrix[color] = _kmeans(B.getMatrix(color), k, epochs, color)
    return B
def _kmeans(b, k, epochs, color):
    """
    """
    centers = np.random.choice(256, size=k)
    for _ in range(epochs):
        assignments = list(np.full(k, -1))
        for row in b:
            for pixel in row:
                closest_center = np.argmin(np.abs(centers-pixel))
                if assignments[closest_center] == -1: assignments[closest_center] = list()
                assignments[closest_center].append(pixel)
        changed_indices = [i for i in range(len(assignments)) if assignments[i] != -1]
        for center in changed_indices:
            centers[center] = np.mean(assignments[center])
    return _threshold(b, np.sort(centers))


def otsu(self, color=3):
    """
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
    """
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
