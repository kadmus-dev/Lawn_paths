from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io



def load_images(images_names):
    images = []
    for name in images_names:
        images.append(io.imread(name))
    return images


def blend(image, mask, alpha=0.2):
    """
    Blend image and it's mask on the same picture.

    Parameters
    ------------------
    image : np.ndarray, dtype=np.uint8
    mask: np.ndarray, dtype=np.int32
    alpha : float
        transparency parameter, 0 <= alpha <= 1

    Returns
    ------------------
    blended : np.ndarray, dtype=np.uint8
        result of applying PIL.Image.blend to given parameters
    """
    mask = mask.astype(np.int32)
    zeros = np.zeros_like(mask)
    mask = np.dstack((mask * 255, zeros, zeros)).astype(np.uint8)
    return Image.blend(
        Image.fromarray(image),
        Image.fromarray(mask), alpha=alpha
    )


def get_blocks(images, block_size, verbose=True):
    """
    Get blocks of square images and labels with given block size.

    Parameters
    ------------------
    images : list of np.ndarray
        list of images
    labels : list of np.ndarray
        list of labels corresponding to images
    block_size : int
        the size of the blocks into which images and labels will be divided
    threshold : float
        function will return blocks such that np.sum(block_label) > threshold

    Returns
    ------------------
    images_blocks : list of np.ndarray, dtype=np.uint8
        list of all selected blocks
    labels_blocks : list of np.ndarray, dtype=np.int32
        list of all selected labels corresponding to images
    """
    images_blocks = []
    pad_width = (block_size - (images[0].shape[0] % block_size)) // 2
    for k in tqdm(range(len(images)), disable=not verbose):
        if len(images[k].shape) == 3:
            image_padded = np.pad(images[k], ((pad_width, pad_width), (pad_width, pad_width), (0, 0)))
        else:
            image_padded = np.pad(images[k], ((pad_width, pad_width), (pad_width, pad_width)))
        n_blocks_height, n_blocks_width = image_padded.shape[0] // block_size, image_padded.shape[1] // block_size
        for j in range(n_blocks_height):
            for i in range(n_blocks_width):
                up, down = j * block_size, (j + 1) * block_size
                left, right = i * block_size, (i + 1) * block_size
                image_block = image_padded[up:down,left:right]
                images_blocks.append(image_block)
    return np.array(images_blocks, dtype=np.uint8)


def mask_pipeline(image, model, block_size=256, verbose=False):
    image_blocks = get_blocks([image], block_size, verbose)
    result_blocks = np.round(model.predict(image_blocks.astype(np.float32))).astype(np.uint8)

    img_size = image.shape[0]
    pad_width = (block_size - (img_size % block_size)) // 2
    mask_shape = (img_size + 2 * pad_width, img_size + 2 * pad_width)
    mask = np.zeros(mask_shape, dtype=np.uint8)
    height, width = mask_shape
    num = 0
    for j in range(height // block_size):
        for i in range(width // block_size):
            up, down = j * block_size, (j+1) * block_size
            left, right = i * block_size, (i+1) * block_size
            mask[up:down,left:right] = result_blocks[num][:,:,0]
            num += 1
    return mask[pad_width:img_size + pad_width, pad_width:img_size + pad_width]
