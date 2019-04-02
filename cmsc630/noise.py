import logging
import numpy as np
from time import time


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
