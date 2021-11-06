from lawn_paths.pipeline.walking_paths import *
import yaml


def run(src_dir, dest_dir):
    # loading config file
    config_path = "../src/lawn_paths/pipeline/main_config.yaml"
    with open(config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    # loading model from checkpoint
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    checkpoint = "../src/lawn_paths/pipeline/pipeline_checkpoint.ckpt"
    pipeline = WalkingPathsDetector.load_from_checkpoint(checkpoint_path=checkpoint, map_location=torch.device('cpu'),
                                                         hparams=hparams).to(device)
    pipeline.eval()

    filenames = sorted(filter(lambda x: x.endswith('tif'), list(os.walk(src_dir))[0][2]))
    for filename in filenames:
        img_src = os.path.join(src_dir, filename)
        mask_dest = os.path.join(dest_dir, filename.split('.')[0] + '.npy')

        image = io.imread(img_src, plugin='tifffile')
        predict = pipeline.predict(image, block_size=hparams['block_size'])
        np.save(mask_dest, predict.astype(np.uint8))
