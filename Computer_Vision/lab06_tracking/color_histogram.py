import numpy as np

def color_histogram(x1, y1, x2, y2, frame, hist_bin):
    """
    Compute color histogram of a given region in a given frame.
    :param x1: x coordinate of the top left corner of the region
    :param y1: y coordinate of the top left corner of the region
    :param x2: x coordinate of the bottom right corner of the region
    :param y2: y coordinate of the bottom right corner of the region
    :param frame: frame to compute the histogram from
    :param hist_bin: number of bins in the histogram
    :return: color histogram of the given region in the given frame
    """
    height = frame.shape[0]
    width = frame.shape[1]
    bbox_height = y2 - y1 
    bbox_width = x2 - x1
    center_x = np.clip((x1 + x2) / 2, bbox_width / 2, width - bbox_width / 2)
    center_y = np.clip((y1 + y2) / 2, bbox_height / 2, height - bbox_height / 2)
    x1 = int(center_x - bbox_width / 2)
    y1 = int(center_y - bbox_height / 2)
    x2 = int(center_x + bbox_width / 2)
    y2 = int(center_y + bbox_height / 2)
    bin_size = 256 / hist_bin
    hist = np.zeros((hist_bin, hist_bin, hist_bin))
    for i in range(x1, x2):
        for j in range(y1, y2):
            r = int(frame[j, i, 0] / bin_size)
            g = int(frame[j, i, 1] / bin_size)
            b = int(frame[j, i, 2] / bin_size)
            hist[r, g, b] += 1
    hist = hist / np.sum(hist)
    return hist
    