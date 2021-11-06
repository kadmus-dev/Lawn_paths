import argparse
from tensorflow import keras
from functions import *


# creating command line parameters parser
parser = argparse.ArgumentParser()
parser.add_argument('command', help='get_mask, blend or get_npy command')
parser.add_argument('arguments', nargs='+', help='arguments of commands: \
                                            list of images for get_mask,  \
                                            image and mask for blend')
args = parser.parse_args()


if args.command == 'get_mask':
    images = load_images(args.arguments)
    model = keras.models.load_model('pipeline_model_v2', compile=False)
    for i in tqdm(range(len(images))):
        mask = mask_pipeline(images[i], model).astype(np.int32)
        mask = np.dstack((np.ones(mask.shape) * 255, (1 - mask) * 255, (1 - mask) * 255)).astype(np.uint8)
        io.imsave(args.arguments[i].split('.')[0] + '_mask.tif', mask)

elif args.command == 'blend':
    if len(args.arguments) != 2:
        raise ValueError("Wrong number of input images for blend:" + str(len(args.arguments)))
    # image, mask = load_images(args.arguments)[:]
    image = io.imread(args.arguments[0])
    mask = np.load(args.arguments[1])
    # mask = np.clip((1 - mask[:,:,1] / 255).astype(np.int32), 0, 255).astype(np.uint8)
    blended = np.array(blend(image, mask), dtype=np.uint8)
    io.imsave(args.arguments[0].split('.')[0] + '_blended.tif', blended)
elif args.command == 'get_npy':
    mask = io.imread(args.arguments[0])
    mask_npy = np.zeros(mask.shape[:2], dtype=np.int32)
    height, width = mask_npy.shape
    mask_npy[(mask == [255,0,0])[:,:,1]] = 1
    print(np.sum(mask_npy))
    fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
    axes[0].imshow(mask)
    axes[1].imshow(mask_npy, cmap='gray')
    plt.show()

    np.save(args.arguments[0].split('.')[0] + '.npy', mask_npy)
else:
    raise ValueError("Wrong command:" + args.command)
