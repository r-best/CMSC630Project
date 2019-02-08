from cmsc630 import Image
import matplotlib.pyplot as plt

x = Image.fromDir("./train")
print(x[0].shape)
