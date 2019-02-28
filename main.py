import os
from cmsc630 import Image
import numpy as np
from time import time
import matplotlib.pyplot as plt


def display(img):
    """Display all color channels of an image
    """
    fig = plt.figure(figsize=(8,8))

    fig.add_subplot(3, 2, 1)
    plt.imshow(img.getMatrix())
    fig.add_subplot(3, 2, 2)
    plt.imshow(img.getMatrix(Image.COLOR_RED), cmap='Reds')
    fig.add_subplot(3, 2, 3)
    plt.imshow(img.getMatrix(Image.COLOR_GREEN), cmap='Greens')
    fig.add_subplot(3, 2, 4)
    plt.imshow(img.getMatrix(Image.COLOR_BLUE), cmap='Blues')
    fig.add_subplot(3, 2, 5)
    plt.imshow(img.getMatrix(Image.COLOR_GRAYSCALE), cmap='gray')
    fig.add_subplot(3, 2, 6)
    plt.imshow(img.getGrayscale(luminosity=True, force=True), cmap='gray')

    plt.show()



CLASSES = [ "cyl", "inter", "let", "mod", "para", "super", "svar" ]

data = dict()
avg_hist = dict()

for filepath in os.listdir("./train"):
    for prefix in CLASSES:
        if filepath.startswith(prefix):
            if prefix not in data:
                data[prefix] = list()
                avg_hist[prefix] = np.zeros(256)
            img = Image.fromFile(f"./train/{filepath}", timer=True)
            if img is not None: data[prefix].append(img)
            break


fig = plt.figure()

for i, prefix in enumerate(CLASSES):
    for img in data[prefix]:
        avg_hist[prefix] += img.getHistogram(color=Image.COLOR_GRAYSCALE)[0]
    avg_hist[prefix] = avg_hist[prefix] / len(data[prefix])

    a1 = plt.subplot(3, 3, i+1)
    a1.set_title(prefix)
    a1.bar(range(256), avg_hist[prefix])

plt.show()
