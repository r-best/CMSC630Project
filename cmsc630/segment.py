import logging
import numpy as np
from time import time


def kmeans(self, k, epochs=30, color=3):
    """
    """
    centers = np.random.choice(256, size=k)
    for _ in range(epochs):
        assignments = list(np.full(k, -1))
        for row in self.getMatrix(color):
            for pixel in row:
                closest_center = np.argmin(np.abs(centers-pixel))
                if assignments[closest_center] == -1: assignments[closest_center] = list()
                assignments[closest_center].append(pixel)
        changed_indices = [i for i in range(len(assignments)) if assignments[i] != -1]
        for center in changed_indices:
            centers[center] = np.mean(assignments[center])
        print(centers)
    return self.threshold(centers, color=color)


def threshold(self, levels, color=3):
    """
    """
    if type(levels) is not list and type(levels) is not np.ndarray:
        levels = [levels]
    levels = np.sort(levels)

    B = self.copy()

    if color == self.COLOR_RGB:
        B._threshold(levels, self.COLOR_RED)
        B._threshold(levels, self.COLOR_GREEN)
        B._threshold(levels, self.COLOR_BLUE)
    else:
        B._threshold(levels, color)

    # Invalidate cached matrices if R, G, or B was edited
    if color in [0,1,2,3]: B.invalidateLazies()

    return B
def _threshold(self, levels, color):
    """
    """
    # Anything below first level goes to 0
    self.matrix[color][self.matrix[color] < levels[0]] = 0
    # For each level, set everything between it and the next to incrementing i
    for i in range(len(levels)-1):
        self.matrix[color][np.logical_and(
            self.matrix[color] >= levels[i],
            self.matrix[color] < levels[i+1]
        )] = i+1
    # Anything above last level goes to highest unused i
    self.matrix[color][self.matrix[color] >= levels[-1]] = len(levels)

    # Scale the small i values up to fill the whole 0-255 range
    self.matrix[color] = np.multiply(self.matrix[color], int(255/len(levels)), dtype=np.uint8)
