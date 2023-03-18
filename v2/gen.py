import os

import svgwrite
from PIL import Image
from skimage import measure
from sklearn.cluster import KMeans
from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def get_closest_colour(colour, colour_list):
    """return the colour in the colour_list closest to colour"""
    distances = [np.linalg.norm(np.array(c) - np.array(colour)) for c in colour_list]
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
        self.image_path = image_path
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

    def plot_image(self):
        plt.imshow(self.im)
        plt.show()

    def plot_palette(self):
        plot_colours_frequencies(self.colours, self.frequencies)
        plt.show()

    def export_thumbnail(self, output_path):
        print(f'exporting thumbnail to {output_path}')
        export_thumbnail(self.image_path, self.colours, self.frequencies, output_path)


def export_thumbnail(image_path, colours, frequencies, out_path):
    """create a thumbnail from an image based on a colour palette and how common those colours are

    """
    VERBOSE=False

    # load the image with PIL
    image = Image.open(image_path)

    # save the height and width of the image, so we have the aspect ratio for later
    height = image.height
    width = image.width
    if VERBOSE:
        print("raw image has (height, width) = ({}, {})".format(height, width))

    max_dimension = 30
    # set small_width and small_height to keep the aspect ratio but be no longer larger than max_dimension
    aspect_ratio = height / width
    if height > width:
        small_height = max_dimension
        small_width = int(small_height / aspect_ratio)
    else:
        small_width = max_dimension
        small_height = int(small_width * aspect_ratio)
    max_distance = 2
    svg = svgwrite.Drawing(out_path, size=(small_width, small_height))
    thumb = np.asarray(image.resize((small_width, small_height)))

    plt.imshow(thumb)
    plt.show()

    # sort the colours by frequency, most common first
    colour_frequencies = list(zip(colours, frequencies))
    colour_frequencies.sort(key=lambda x: x[1], reverse=True)

    for colour, frequency in colour_frequencies:
        if VERBOSE:
            print(f"{colour}: {frequency}")

        # make a mask of the thumbnail by pixels that are the closest to the current colour
        mask = np.zeros(thumb.shape[:2], dtype=bool)
        for i in range(thumb.shape[0]):
            for j in range(thumb.shape[1]):
                if get_closest_colour(thumb[i, j], colours) == colour:
                    mask[i, j] = True
                else:
                    mask[i, j] = False

        # convert the binary mask to an image to show it
        tmp = mask.astype(float)
        plt.imshow(tmp)
        if VERBOSE:
            plt.title("mask for the colour {}".format(colour))
            plt.show()

        # expand the area of the mask
        new_mask = mask.copy()
        new_mask[:, 1:] = mask[:, 1:] + mask[:, :-1]
        new_mask[1:, :] = new_mask[1:, :] + new_mask[:-1, :]

        mask = new_mask

        # now that we have the mask, iterate over the distinct areas in the mast
        img_labeled, island_count = measure.label(mask.astype(np.uint8), return_num=True, connectivity=1)
        if VERBOSE:
            print("found {} islands in the mask".format(island_count))
        for i in range(island_count + 1):
            # show the island with the index i
            current_island = img_labeled == i

            # intersect with the mask
            current_island = np.logical_and(current_island, mask)

            # skip if the number of pixels on the island is less than a percentage of the total number of pixels
            if np.sum(current_island) < int(small_width * small_height * 0.05):
                continue

            if VERBOSE:
                print(f"island index {i} has {np.sum(current_island)} pixels, finding polyline")

            # pad with 2px on all side before finding contours, so we can contour the edge of the image
            padded = np.pad(current_island, 2)

            if VERBOSE:
                plt.imshow(padded)
                plt.show()

            contours = measure.find_contours(padded.astype(np.uint8), 0)
            if VERBOSE:
                print(f"found {len(contours)} contours on this island")

            for contour in contours:
                write_compressed_contour_to_svg(colour, contour, svg, thumb, max_distance)

    # save the SVG
    print("saving SVG to {}".format(out_path))
    svg.save()


def write_compressed_contour_to_svg(colour, contour, svg, thumb, max_distance):
    # subtract (2,2) from each point in the contour
    contour -= np.array([2, 2])
    print(f"contour has {len(contour)} points")
    if len(contour) < 4:
        print("contour is too small to compress")
        approx_polygon = contour
    else:
        # show_contours_on_image([contour], thumb)
        # reduce the number of points in the contours with the approximate_polygon method
        approx_polygon = measure.approximate_polygon(contour, max_distance)
        # show_contours_on_image([approx_polygon], thumb)
    # add the contour to the SVG
    x, y = approx_polygon.T
    svg.add(svgwrite.shapes.Polygon(np.stack([y, x], axis=1), fill=f"rgb({colour[0]}, {colour[1]}, {colour[2]})"))


def show_contours_on_image(contours, thumb):
    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(thumb, cmap=plt.cm.gray)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
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
    # image_path = "../images/1_buller.jpg"
    # palette = ColourPalette(image_path)
    # palette.export_thumbnail("buller.svg")

    images_dir = "../images"
    thumbnail_dir = "../thumbnails"

    if not os.path.exists(thumbnail_dir):
        os.makedirs(thumbnail_dir)

    # export thumbnails for each image in the images directory
    for image_name in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_name)
        palette = ColourPalette(image_path)
        palette.export_thumbnail(os.path.join(thumbnail_dir, os.path.basename(image_name) + ".svg"))
