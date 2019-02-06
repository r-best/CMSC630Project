from cmsc630.part1 import Image, loadImages
import matplotlib.pyplot as plt

x = loadImages("./train", color=Image.COLOR_RGB)
print(x[0].shape)
