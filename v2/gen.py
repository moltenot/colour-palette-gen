from PIL import Image
from sklearn.cluster import KMeans
from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def get_closest_colour(colour, colour_list):
    """return the colour in the colour_list closest to colour"""
    distances = [np.linalg.norm(c - colour) for c in colour_list]
    return colour_list[np.argmin(distances)]


class ColourPalette:
    """represent an image colour palette

    Attributes:
        im (np.ndarray): the image
        labels (np.ndarray): the cluster labels
        centroids (np.ndarray): the cluster centroids
        colours (np.ndarray): the colours in the palette
        frequencies (np.ndarray): the frequencies of the colours in the palette
    """

    def __init__(self, image_path):
        print(f'getting colours from {image_path}')
        self.im = np.asarray(Image.open(image_path))[:, :, :3]
        flat = decimate_image(self.im, 5000)

        s_time = time()
        k_means = KMeans(n_clusters=6, n_init="auto")
        k_means.fit(flat)
        print(f"finished in {np.round(time() - s_time)} seconds")

        self.labels = k_means.labels_
        self.centroids = k_means.cluster_centers_
        self.frequencies = get_frequencies(self.labels, self.centroids)

        self.colours = get_int_colours(self.centroids)

        # make a thumbnail
        self.thumbnail = Image.open(image_path).resize((200, 200))

        # label each pixel in the thumbnail by which colour is closest to it
        self.thumbnail = np.asarray(self.thumbnail)[:, :, :3]
        self.thumbnail_mask = np.zeros(self.thumbnail.shape, dtype=int)
        for i in range(self.thumbnail.shape[0]):
            for j in range(self.thumbnail.shape[1]):
                self.thumbnail_mask[i, j, :] = get_closest_colour(self.thumbnail[i, j], self.colours)

    def plot_image(self):
        plt.imshow(self.im)
        plt.show()

    def plot_palette(self):
        plot_colours_frequencies(self.colours, self.frequencies)
        plt.show()

    def plot_thumbnail(self):
        plt.imshow(self.thumbnail_mask)
        plt.show()


def decimate_image(image: np.ndarray, N: int) -> np.ndarray:
    """reduce the size of the image until it is composed of roughly N pixels

    :param image: the image to be decimated
    :type image: np.ndarray
    :param N: the number of pixels to keep
    :type N: int
    :returns: the decimated image
    """
    height, width, channels = image.shape
    flat = image.reshape((height * width, channels))

    target_number_colours = N
    # want to remove the colours in the flattened image until there are about `target_number_colours`
    factor = flat.shape[0] / target_number_colours
    number_of_halves = int(np.floor(np.log2(factor)))

    for _ in range(number_of_halves):
        # half the length of the array until it is just a little over the target number of colours long
        if (flat.shape[0] % 2) != 0:
            flat = flat[:-1, ::]
        flat = np.reshape(flat, (-1, 2, channels))
        flat = flat[:, 0, :]

    return flat


def get_int_colours(float_colours):
    ret_colours = []
    for col in float_colours:
        int_colour = [round(i) for i in col]
        ret_colours.append(int_colour)
    return ret_colours


def get_frequencies(labels, centroids):
    labels = list(labels)
    percents = []
    for i in range(len(centroids)):
        j = labels.count(i)
        j = j / (len(labels))
        percents.append(j)
    return percents


def get_rgba_colours(colours: list[list[float]]) -> list[list[float]]:
    ret = []
    for colour in colours:
        rgba_colour = [i / 255 for i in colour]
        rgba_colour.append(1)  # for the alpha
        ret.append(rgba_colour)
    return ret


def get_starts(frequencies):
    starts = [0]
    for i, percent in enumerate(frequencies):
        starts.append(starts[i] + percent)
    return starts


def plot_colours_frequencies(colours, frequencies):
    fig, ax = plt.subplots(frameon=False)
    fig.set_figheight(1)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    rgba_colours = get_rgba_colours(colours)
    starts = get_starts(frequencies)

    for colour, start, width in zip(rgba_colours, starts, frequencies):
        plt.barh(0, [width], left=[start], color=[colour])

    return fig, ax


def plot_colours_long(colours: list[int]):
    square = ([0, 0, 1, 1], [0, 1, 1, 0])
    x = square[0]
    y = square[1]

    num_colums = 5
    list_length = len(colours)
    num_rows = np.ceil(list_length / num_colums)
    print(f"list length {list_length}")
    print(f"number rows: {num_rows}")
    print(f"number colums:{num_colums}")

    fig, axes = plt.subplots(1, 5)
    fig.set_size_inches(10, 2)

    for i, ax in enumerate(axes):
        if i >= len(colours): break
        ax.fill(x, y, colors.to_hex([i / 255 for i in colours[i]]))
        ax.axis('off')


def plot_colours(colours: list[int]):
    square = ([0, 0, 1, 1], [0, 1, 1, 0])
    x = square[0]
    y = square[1]
    fig, axes = plt.subplots(1, 5)
    fig.set_size_inches(10, 2)

    for i, ax in enumerate(axes):
        ax.fill(x, y, colors.to_hex([i / 255 for i in colours[i]]))
        ax.axis('off')


if __name__ == '__main__':
    image_path = "../images/1_buller.jpg"
    palette = ColourPalette(image_path)

    # palette.plot_image()
    # palette.plot_palette()
    palette.plot_thumbnail()
