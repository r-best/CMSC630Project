from cmsc630 import Image
import numpy as np
import matplotlib.pyplot as plt

x = Image.fromDir("./train/cyl01.BMP")[0]
y = x.quantize(color=Image.COLOR_GRAYSCALE)

print(x.getMatrix(color=Image.COLOR_GRAYSCALE))
print(y.matrix[Image.COLOR_GRAYSCALE])

# plt.imshow(x.getGrayscale(), cmap='gray')
# plt.imshow(y.getGrayscale(), cmap='gray')
# plt.show()

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
