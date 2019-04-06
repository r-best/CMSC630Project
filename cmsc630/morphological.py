import logging
import numpy as np
from time import time


def dilate(self, structure, color=3):
    """
    """
    structure[structure>0] = 255
    B = self.copy()
    if color == self.COLOR_RGB:
        B.matrix[self.COLOR_RED] = _dilate(B.getMatrix(self.COLOR_RED), structure)
        B.matrix[self.COLOR_GREEN] = _dilate(B.getMatrix(self.COLOR_GREEN), structure)
        B.matrix[self.COLOR_BLUE] = _dilate(B.getMatrix(self.COLOR_BLUE), structure)
    else:
        B.matrix[color] = _dilate(B.getMatrix(color), structure)
    
    if color in [0,1,2,3]: B.invalidateLazies()

    return B
def _dilate(mat, structure):
    """
    """
    offset_w = int(structure.shape[0]/2)
    offset_h = int(structure.shape[1]/2)

    dilated = np.zeros_like(mat)
    for i in range(offset_w, mat.shape[0]-offset_w-1):
        for j in range(offset_h, mat.shape[1]-offset_h-1):
            if mat[i,j] == 255:
                dilated[i-offset_w:i+offset_w+1,j-offset_h:j+offset_h+1] = np.maximum(
                    mat[i-offset_w:i+offset_w+1,j-offset_h:j+offset_h+1],
                    structure
                )
    return dilated

def erode(self, structure, color=3):
    """
    """
    B = self.copy()
    if color == self.COLOR_RGB:
        B.matrix[self.COLOR_RED] = _erode(B.getMatrix(self.COLOR_RED), structure)
        B.matrix[self.COLOR_GREEN] = _erode(B.getMatrix(self.COLOR_GREEN), structure)
        B.matrix[self.COLOR_BLUE] = _erode(B.getMatrix(self.COLOR_BLUE), structure)
    else:
        B.matrix[color] = _erode(B.getMatrix(color), structure)
    
    if color in [0,1,2,3]: B.invalidateLazies()

    return B
def _erode(mat, structure):
    """
    """
    # Compute logical NOT of structuring element, anything above 0 becoming 0
    # and any 0s becoming 255
    structure[structure>0] = 1
    structure[structure==0] = 255
    structure[structure==1] = 0

    offset_w = int(structure.shape[0]/2)
    offset_h = int(structure.shape[1]/2)

    eroded = np.zeros_like(mat)
    for i in range(offset_w, mat.shape[0]-offset_w-1):
        for j in range(offset_h, mat.shape[1]-offset_h-1):
            if np.all(np.logical_or(mat[i-offset_w:i+offset_w+1,j-offset_h:j+offset_h+1], structure)):
                eroded[i,j] = 255
    return eroded
