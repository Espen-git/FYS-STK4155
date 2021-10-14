import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# Load the terrain
terrain = imread('SRTM_data_Norway_1.tif')
show_terrain = False
if show_terrain:
    # Show the terrain
    plt.figure()
    plt.title('Terrain over Norway 1')
    plt.imshow(terrain, cmap='gray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

print(terrain.shape)
# z = terrain


def p1():
    return _

def p2():
    return _

def p3():
    return _

def p4():
    return _

def p5():
    return _