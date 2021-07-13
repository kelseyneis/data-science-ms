from matplotlib import pyplot as plt
import matplotlib
# recommended color for different digits
color_mapping = ['red', 'green', 'blue', 'yellow', 'magenta', 'orangered',
                 'cyan', 'purple', 'gold', 'pink']


def plot2d(data, label, split='train'):
    # 2d scatter plot of the hidden features
    plt.title("2D plot for " + split)
    plt.scatter(
        data[:, 0], data[:, 1], c=label, cmap=matplotlib.colors.ListedColormap(color_mapping))
    plt.show()


def plot3d(data, label, split='train'):
    # 3d scatter plot of the hidden features
    # Creating dataset
    ax = plt.axes(projection="3d")
    # Creating plot
    ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], c=label,
                 cmap=matplotlib.colors.ListedColormap(color_mapping))
    plt.title("3D plot for " + split)
    # show plot
    plt.show()
