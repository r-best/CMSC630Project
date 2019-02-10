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

filter = np.zeros((401, 401))
filter[400, 400] = 1
y = x.applyFilter(filter)

display(y)

# x = np.array([5, 2, 6, 3])

# print(x[np.where(x == 6)])
# x[x == 6] = 5
# print(x)

# a = [np.array([
#     [11, 12, 20, 14, 15],
#     [16, 17, 18, 19, 20]
# ])]
# i = np.where(np.isin(a[0][:], [11, 20]))
# print(i)
# a[0][i] = 5
# print(a[0])
