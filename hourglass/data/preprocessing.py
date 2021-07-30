import h5py
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from pathlib import Path
from hourglass.data.heatmap import HeatmapGenerator
from hourglass.data.transforms import apply_transform, newell_cropping_transform, \
    rotation_transform, permutation_transform, horizontal_flip_transform


def get_mpii_tensor(annotation_dir: Path, img_dir: Path, split='train'):
    """Returns a tuple of np arrays containing the necessary information for building a dataset.

    Args:
        annotation_dir (Path): Directory that contains train.h5 and valid.h5 from Newell et al.
                               (see original github repo)
        img_dir (Path): Directory that contains all the images from MPII dataset
        split (str, optional): Either 'val' or 'train'. Defaults to 'train'.

    Returns:
        numpy.array: Array that contains filenames, body center, body scale, keypoints and
                     keypoint visibility
    """
    split_to_file = {
        'train': 'train.h5',
        'val': 'valid.h5'
    }

    db = h5py.File(annotation_dir / split_to_file[split], mode='r')

    # Convert short name of images to complete paths
    filenames = []
    for short_name in db['imgname'][:]:
        filenames.append(str(img_dir / short_name.decode()))

    return (
        np.array(filenames, dtype=str),
        # the h5py databases are converted to np.array through slicing
        db['center'][:].astype(np.float32),
        db['scale'][:].astype(np.float32),
        db['normalize'][:].astype(np.float32),  # Head size for PCKh metric
        db['part'][:].astype(np.float32),
        db['visible'][:].astype(np.float32)
    )


class ReadImage:
    def __call__(self, *data):
        return (tf.image.decode_image(tf.io.read_file(data[0]), dtype=tf.dtypes.float32),) \
            + data[1:]


class StandardizeImage:
    def __call__(self, *data):
        return (tf.image.per_image_standardization(data[0]),) + data[1:]


class HeatmapRetreiver:
    def __init__(self, n_keypoints, hmap_sigma, hmap_res):
        self.heatmap_generator = HeatmapGenerator(hmap_sigma, n_keypoints, resolution=hmap_res)
        self.hmap_res = hmap_res

    def __call__(self, keypoints, visible, body_center, body_scale, angle, flipped):
        # transformation matrix
        m_crop = newell_cropping_transform(
            center=body_center,
            body_scale=body_scale,
            crop_resolution=self.hmap_res)

        if flipped:
            m_flip = horizontal_flip_transform(self.hmap_res).astype(tf.dtypes.float32)
        else:
            m_flip = tf.eye(3)

        m_rot = rotation_transform(
            angle=-angle,
            resolution=self.hmap_res)

        m_transform = m_rot @ m_flip @ m_crop

        # heatmaps
        kps = apply_transform(keypoints, m_transform)
        return self.heatmap_generator(kps, visible), m_transform


class NewellPreprocessing:
    def __init__(self, is_train=True, rotate=(-30, 30), scale=(0.75, 1.25), flip=True, hue=0.5,
                 brightness=0.2, saturation=0.5, contrast=0.5, heatmap_sigma=1,
                 n_keypoints=16, img_res=256, heatmap_res=64, seed=None,
                 flipped_indices=[5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]):
        self.is_train = is_train
        self.rotate = rotate  # rotation bounds
        self.scale = scale  # scaling bounds
        self.flip = flip
        self.hue = hue
        self.brightness = brightness
        self.saturation = saturation
        self.contrast = contrast
        self.img_res = img_res  # output image resolution
        self.hmap_res = float(heatmap_res)  # output heatmap resolution
        self.flipped_permutation = \
            permutation_transform(tf.constant(flipped_indices)).astype(tf.dtypes.float32)
        self.flip_transform = horizontal_flip_transform(img_res)
        self.heatmap_retreiver = HeatmapRetreiver(n_keypoints, heatmap_sigma, heatmap_res)
        self.rng = tf.random.Generator.from_non_deterministic_state() if seed is None \
            else tf.random.Generator.from_seed(seed)

    def __call__(self, *data):
        img, body_center, body_scale, head_size, keypoints, visible = data

        # data augmentation
        if self.is_train:
            angle = (self.rotate[1] - self.rotate[0]) * self.rng.uniform([]) + self.rotate[0]
            scale = (self.scale[1] - self.scale[0]) * self.rng.uniform([]) + self.scale[0]
            do_flip = self.rng.uniform([]) < 0.5
        else:
            angle = 0
            scale = 1
            do_flip = False

        body_size = 200 * body_scale * scale
        tl = tf.cast(body_center - body_size / 2, tf.int32)
        br = tf.cast(body_center + body_size / 2, tf.int32)
        height = tf.shape(img)[0]
        width = tf.shape(img)[1]

        bboxes = tf.stack(
            (tl[1]/(height-1), tl[0]/(width-1),
             br[1]/(height-1), br[0]/(width-1)), axis=0)

        crop = tf.squeeze(tf.image.crop_and_resize(
            tf.expand_dims(img, axis=0),
            tf.cast(tf.expand_dims(bboxes, axis=0), dtype=tf.float32),
            tf.constant([0]),
            tf.constant([self.img_res, self.img_res])
        ))

        # crop = tf.image.per_image_standardization(crop)

        # Flipping
        if do_flip:
            crop = tf.image.flip_left_right(crop)
            keypoints = self.flipped_permutation @ keypoints

        # Color jittering
        if self.is_train:
            seeds = self.rng.make_seeds(4)
            crop = tf.image.stateless_random_brightness(
                crop, max_delta=self.brightness, seed=seeds[:, 0])
            crop = tf.image.stateless_random_contrast(
                crop, lower=max(0, 1-self.contrast), upper=1+self.contrast, seed=seeds[:, 1])
            crop = tf.image.stateless_random_saturation(
                crop, lower=max(0, 1-self.saturation), upper=1+self.saturation, seed=seeds[:, 2])
            crop = tf.image.stateless_random_hue(
                crop, max_delta=self.hue, seed=seeds[:, 3]
            )

        rot_img = tfa.image.rotate(crop, angle/180*np.pi)  # numpy < 1.20 only
        heatmaps, transform = self.heatmap_retreiver(
            keypoints, visible, body_center, body_scale*scale, angle, flipped=do_flip)

        # Heatmaps come in shape H,W,C
        return rot_img, heatmaps, keypoints, transform, head_size, visible


class VisualizationPreprocessing:
    def __init__(self, img_res=256, hmap_res=64, n_keypoints=16, hmap_sigma=1):
        self.img_res = img_res
        self.hmap_res = float(hmap_res)
        self.heatmap_retreiver = HeatmapRetreiver(n_keypoints, hmap_sigma, hmap_res)

    def __call__(self, *data):
        img, body_center, body_scale, head_size, keypoints, visible = data

        body_size = 200 * body_scale
        tl = tf.cast(body_center - body_size / 2, tf.int32)
        br = tf.cast(body_center + body_size / 2, tf.int32)
        height = tf.shape(img)[0]
        width = tf.shape(img)[1]

        bboxes = tf.stack(
            (tl[1]/(height-1), tl[0]/(width-1),
             br[1]/(height-1), br[0]/(width-1)), axis=0)

        crop = tf.squeeze(tf.image.crop_and_resize(
            tf.expand_dims(img, axis=0),
            tf.cast(tf.expand_dims(bboxes, axis=0), dtype=tf.float32),
            tf.constant([0]),
            tf.constant([self.img_res, self.img_res])
        ))

        crop_standardized = tf.image.per_image_standardization(crop)
        crop_transform = newell_cropping_transform(body_center, body_scale, self.img_res)

        heatmaps, hmap_transform = self.heatmap_retreiver(
            keypoints, visible, body_center, body_scale, 0, False)

        return crop, crop_standardized, heatmaps, crop_transform, hmap_transform, \
            keypoints, head_size, visible


# This class repeats along the batch dimension, thus, enlarging the number of batched data
# Hourglass models apply a loss multiple times after each stage/hourglass
class TileLabelBatches:
    def __init__(self, num_repeats):
        self.num_repeats = num_repeats

    def __call__(self, *data):
        return (data[0], tf.tile(data[1], (self.num_repeats, 1, 1, 1)),) + data[2:]
