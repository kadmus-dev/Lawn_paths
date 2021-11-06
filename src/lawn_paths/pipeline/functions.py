import os, json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io


def get_blocks(image, block_size):
    image_blocks = []
    pad_width = (block_size - (image.shape[0] % block_size)) // 2

    if len(image.shape) == 3:
        image_padded = np.pad(image, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)))
    else:
        image_padded = np.pad(image, ((pad_width, pad_width), (pad_width, pad_width)))
    n_blocks_height, n_blocks_width = image_padded.shape[0] // block_size, image_padded.shape[1] // block_size
    for j in range(n_blocks_height):
        for i in range(n_blocks_width):
            up, down = j * block_size, (j + 1) * block_size
            left, right = i * block_size, (i + 1) * block_size
            image_block = image_padded[up:down, left:right]
            image_blocks.append(image_block)
    return np.array(image_blocks, dtype=np.uint8)


def train_val_blocks(src, dest, ext, block_size, split, put_dict_sum=False, json_name=None, verbose=True):
    train, val = [], []

    filenames = sorted(filter(lambda x: x.endswith(ext), list(os.walk(src))[0][2]))
    if ext == '.tif':
        for k in range(len(filenames)):
            filename = filenames[k]
            image = io.imread(os.path.join(src, filename), plugin='tifffile')
            blocks = get_blocks(image, block_size)

            blocks_prefix = os.path.join(dest, filename).split('.')[0] + '_block'
            names = [blocks_prefix + str(i) + ext for i in range(blocks.shape[0])]
            for i in range(blocks.shape[0]):
                io.imsave(names[i], blocks[i], plugin='tifffile', check_contrast=False)

            if k in split[0]:  # train
                train += names
            elif k in split[1]:  # val
                val += names

    elif ext == '.npy':
        block_sums_train = []
        block_sums_val = []
        for k in range(len(filenames)):
            filename = filenames[k]
            mask = np.load(os.path.join(src, filename), fix_imports=False)
            blocks = get_blocks(mask, block_size)

            blocks_prefix = os.path.join(dest, filename).split('.')[0] + '_block'
            names = [blocks_prefix + str(i) + ext for i in range(blocks.shape[0])]
            block_sums = dict()
            for i in range(blocks.shape[0]):
                block_sums[names[i]] = int(np.sum(blocks[i]))
                np.save(names[i], blocks[i])

            if k in split[0]:  # train
                train += names
                block_sums_train += list(block_sums.items())
            elif k in split[1]:  # val
                val += names
                block_sums_val += list(block_sums.items())

        if put_dict_sum and json_name is not None:
            with open(os.path.join(dest, json_name + '_train.json'), 'w') as f:
                json.dump(dict(block_sums_train), f)
            with open(os.path.join(dest, json_name + '_val.json'), 'w') as f:
                json.dump(dict(block_sums_val), f)

    return train, val


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
    return Image.blend(Image.fromarray(image), Image.fromarray(mask), alpha=alpha)


def show_images(images, titles=None):
    """
    Show a list of images with given titles.
    
    Parameters
    ------------------
    images : list of np.ndarray, dtype=np.uint8
    titles : list of str
        titles of pictures, should have same len as images
    
    Returns
    ------------------
    None
    """
    if titles is None:
        titles = []
    n = len(images)
    figure = plt.figure(figsize=(15, 15))
    for i in range(n):
        figure.add_subplot(1, n, i + 1)
        plt.imshow(images[i])
        if titles:
            plt.xlabel(titles[i])
    plt.show(block=True)


def get_augs(image, mask, transforms, num_augs):
    transformed_images, transformed_labels = [], []
    for j in range(num_augs):
        transformed = transforms(image=image, mask=mask)
        is_duplicate = np.any(
            [np.all(transformed['image'] == transformed_image) for transformed_image in transformed_images])
        if not is_duplicate:
            transformed_images.append(transformed['image'])
            transformed_labels.append(transformed['mask'])

    return np.array(transformed_images, dtype=np.float32), np.array(transformed_labels, dtype=np.float32)


def put_augs(pairs, dest_dir, transforms, num_augs, json_path, verbose=True):
    augs_data = dict()
    for img_name, mask_name in pairs:
        image = io.imread(img_name, plugin='tifffile')
        mask = np.load(mask_name, fix_imports=False)
        images, masks = get_augs(image, mask, transforms, num_augs)

        img_prefix = os.path.join(os.path.join(dest_dir, 'images'), img_name.split('/')[-1].split('.')[0] + '_aug')
        label_prefix = os.path.join(os.path.join(dest_dir, 'labels'), mask_name.split('/')[-1].split('.')[0] + '_aug')

        for i in range(images.shape[0]):
            img_dest = img_prefix + str(i) + '.tif'
            mask_dest = label_prefix + str(i) + '.npy'
            io.imsave(img_dest, np.clip(images[i].astype(np.int32), 0, 255).astype(np.uint8), check_contrast=False)
            np.save(mask_dest, np.clip(masks[i].astype(np.int32), 0, 255).astype(np.uint8))
            augs_data[img_dest] = mask_dest

    with open(json_path, 'w') as f:
        json.dump(augs_data, f)
