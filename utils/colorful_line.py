from matplotlib.collections import LineCollection
import numpy as np
import math
import matplotlib.pyplot as plt


def draw_colorful_line(draw_data: np.ndarray):
    # Define a color list corresponding to each point; dimension matches the number of points
    colors = []
    length = len(draw_data)
    x = np.arange(length)
    # Define which points belong to which color
    red = [1, 2, 3]
    blue = [4, 5, 6]
    cyan = [7, 8, 9]
    # Fill the color list
    for i in range(length):
        if i in red:
            colors.append('red')
        elif i in blue:
            colors.append('blue')
        elif i in cyan:
            colors.append('cyan')
        else:
            colors.append('purple')
    y = draw_data

    points = np.array([x, y]).T.reshape(-1, 1, 2)  # shape (152, 1, 2)
    # Slice because colors are drawn on line segments, not on points
    segments = np.concatenate([points[:-1], points[1:]], axis=1)  # shape (151, 2, 2)
    lc = LineCollection(segments, color=colors)
    ax = plt.axes()
    ax.set_xlim(0, length)
    ax.set_ylim(min(y), max(y))
    ax.add_collection(lc)
    plt.show()
    plt.close()


if __name__ == '__main__':
    pi = 3.1415

    x = np.linspace(0, 4 * pi, 100)
    y = [math.cos(xx) for xx in x]
    lwidths = abs(x)
    color = []
    for i in range(len(y)):
        if i < 50:
            color.append('#FF0000')
        else:
            color.append('#000000')
    print(color)
    # print(x)
    # print(y)
    print('--------------------------------------')
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    print(np.array([x, y]).shape)
    print(points.shape)
    print('--------------------------------------')
    print(points[:-1].shape)
    print(points[1:].shape)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    print(segments.shape)
    lc = LineCollection(segments, linewidths=lwidths, color=color)

    ax = plt.axes()
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.add_collection(lc)
    plt.show()
    plt.close()

    '''
    fig, a = plt.subplots()
    a.add_collection(lc)
    a.set_xlim(0, 4*pi)
    a.set_ylim(-1.1, 1.1)
    fig.show()
    '''
