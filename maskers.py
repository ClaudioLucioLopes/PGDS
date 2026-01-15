import heapq
import warnings
from typing import Iterable
from typing import Union
import numpy as np
from numpy import ndarray
from skimage.transform import resize

def generate_interpolated_float_masks_for_image(image_shape: Iterable[int],
                                                p_keep: float,
                                                number_of_masks: int,
                                                number_of_features: int):
    """Generates a set of random masks of float values to mask image data.

    Args:
        image_shape (int): Size of a single sample of input data, for images without the channel axis.
        p_keep: ?
        number_of_masks: Number of masks
        number_of_features: Number of features (or blobs) in both dimensions

    Returns:
        The generated masks (np.ndarray)
    """
    grid = np.random.choice(a=(True, False),
                            size=(number_of_masks, number_of_features,
                                  number_of_features),
                            p=(p_keep, 1 - p_keep)).astype('float32')
    mask_shape = image_shape[:2]
    cell_size = np.ceil(np.array(mask_shape) / number_of_features)
    up_size = (number_of_features + 1) * cell_size
    masks = np.empty((number_of_masks, *mask_shape), dtype=np.float32)
    for i in range(masks.shape[0]):
        y_offset = np.random.randint(0, cell_size[0])
        x_offset = np.random.randint(0, cell_size[1])
        # Linear upsampling and cropping
        upscaled = _upscale(grid[i], up_size)
        masks[i, :, :] = upscaled[y_offset:y_offset + image_shape[0],
                                  x_offset:x_offset + image_shape[1]]
    masks = masks.reshape(-1, *mask_shape, 1)
    return masks


def _upscale(grid_i, up_size):
    """Up samples and crops the grid to result in an array with size up_size."""
    return resize(grid_i,
                  up_size,
                  order=1,
                  mode='reflect',
                  anti_aliasing=False)
