import itertools
import skimage
import numpy as np
import skimage.filters


def gaussian(image, sigma):
    return skimage.filters.gaussian(image, sigma)


def intensity_scale(img, old_range, new_range):
    shift = -old_range[0] + new_range[0] * \
        (old_range[1] - old_range[0]) / (new_range[1] - new_range[0])
    scale = (new_range[1] - new_range[0]) / (old_range[1] - old_range[0])
    return (img + shift) * scale


def normalize(img, out_range=(-1, 1)):
    min_value = np.min(img)
    max_value = np.max(img)
    old_range = (min_value, max_value)
    return intensity_scale(img, old_range, out_range)


def bounding_box(image):
    dim = image.ndim
    start = []
    end = []
    for ax in itertools.combinations(reversed(range(dim)), dim - 1):
        # 2,1 2,0 1,0
        # 获取非零元素对应索引的数组
        nonzero = np.any(image, axis=ax)
        nonzero_where = np.where(nonzero)[0]
        if len(nonzero_where) > 0:
            curr_start, curr_end = nonzero_where[[0, -1]]
        else:
            curr_start, curr_end = np.nan, np.nan
        start.append(curr_start)
        end.append(curr_end)
    return np.array(start), np.array(end)


def connected_component(image, dtype=np.uint8, connectivity=1, calculate_bounding_box=True):
    # 标记互为邻居的像/体素点，获取有多少这样的组成块
    labels = np.zeros(image.shape, dtype=dtype)
    if calculate_bounding_box:
        # find the bounding_box, which makes the function typically faster
        start, end = bounding_box(image)
        if np.any(np.isnan(start)) or np.any(np.isnan(end)):
            # if bounding box is not found, return a zeroes image with 0 number of labels
            return labels, 0
        # +1 because bounding_box is inclusive
        slices = tuple([slice(s, e + 1) for s, e in zip(start, end)])
    else:
        slices = tuple([slice(0, s) for s in image.shape])
    image_cropped = image[slices]
    labels_cropped, num = skimage.measure.label(
        image_cropped, connectivity=connectivity, return_num=True, background=0)
    labels = np.zeros(image.shape, dtype=dtype)
    labels[slices] = labels_cropped
    return labels, num


def largest_connected_component(image):
    labels, num = connected_component(image)
    if num == 0:
        return np.zeros_like(image)
    counts = np.bincount(labels.flatten())
    largest_label = np.argmax(counts[1:]) + 1
    lcc = (labels == largest_label)
    return lcc


def bb(image, transformation, image_spacing):
    image_thresholded = (np.squeeze(image / np.max(image))
                         > 0.5).astype(np.uint8)
    #image_thresholded = (np.squeeze(image) > 8).astype(np.uint8)
    image_thresholded = largest_connected_component(image_thresholded)
    start, end = bounding_box(image_thresholded)
    #start = np.flip(start.astype(np.float64) * np.array(image_spacing, np.float64))
    #end = np.flip(end.astype(np.float64) * np.array(image_spacing, np.float64))
    #start_transformed = transformation.TransformPoint(start)
    #end_transformed = transformation.TransformPoint(end)
    # return start_transformed, end_transformed
    return start, end
