# Colour palette generator

The idea of this repository is to do as many fancy tricks on images as possible, and integrate these back into my website at https://theo.molteno.nz.

So far this has involved using K-means clustering to generate the 5 most significant colours in each image.

In version 2 this includes an algorithm to generate a small SVG thumbnail from the image coloured using the same colours extracted from the colour palette generation step.

## Package

This is packaged as Python Package using the [flit build system](https://flit.pypa.io/en/stable/). 

Any colour palette generator should extend `utils.ImageAnalyzer` in order to have predictable inferences on each image.
