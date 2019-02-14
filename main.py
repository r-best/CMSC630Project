from cmsc630 import Image
import numpy as np
import matplotlib.pyplot as plt


def display(img):
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


x = Image.fromDir("./test/1.png")[0]
# y = x.equalize()
# y = x.quantize(delta=128, technique=Image.QUANT_MEDIAN)
# y = x.applyFilter([[ 0, 0, 0 ],
#                    [ 0, 1, 1 ],
#                    [ 0, 0, 0 ]])
# y = x.applyFilter(np.ones((35, 35)), strategy=Image.FILTER_STRAT_MEAN, border=Image.FILTER_BORDER_EXTEND)
y = x.makeGaussianNoise(rate=0.25)

# filter = np.zeros((101, 101))
# filter[50, 50] = -1
# y = x.applyFilter(filter, border=Image.FILTER_BORDER_PAD)

display(y)


# print(np.zeros((1, 3), dtype='int'))
