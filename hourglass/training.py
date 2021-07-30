import argparse
import datetime
import json
import tensorflow as tf
import tensorflow.experimental.numpy as tnp  # type: ignore

from pathlib import Path
from tensorflow import keras
from time import perf_counter, time

from hourglass.data import preprocessing
from hourglass.data.metrics import PCKhMetric
from hourglass.data.visualization import KeypointVisualizer, hmaps_to_grayscale_img
from hourglass.models import BulatSmall, BulatTiny, HourglassModel, StackedHourglass,\
    StackedHourglassSmall
from hourglass.utils.config import YAMLConfig
from hourglass.utils.logging import TFSummaryWriter, WandBWriter, SummaryWriter


class ConstantSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr):
        self.lr = lr

    def __call__(self, step):
        return self.lr


class MultiStepSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, steps_per_epoch, init_lr=2.5e-4, gamma=0.2, milestones=[75, 100, 150]):
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = init_lr
        self.gamma = gamma
        self.milestones = tnp.array(milestones)

    def __call__(self, step):
        epoch = step // self.steps_per_epoch
        exponent = tnp.sum(tnp.where(self.milestones <= epoch, 1, 0))
        return self.init_lr * self.gamma ** exponent


class CheckpointManager:
    def __init__(self, cp_dir: Path, keep_best=True, max_keep=10):
        self.cp_dir = cp_dir
        self.cp_dir.mkdir(exist_ok=False)

        self.checkpoints = []
        self.checkpoints_file = self.cp_dir / 'checkpoints.json'
        self.max_keep = max_keep

        if max_keep < 1:
            raise ValueError(f"Argument 'max_keep' must be at least 1, but is {max_keep}")

    def load(self):
        with self.checkpoints_file.open('r') as file:
            self.checkpoints = json.load(file)

    def _write(self, model, path):
        model.save_weights(path, save_format='h5')
        print(f"Checkpoint saved to {path}.")

    def save(self, model: keras.Model, epoch: int, metric: float, args: dict = {}):
        """Save a checkpoint and handle things like keeping best and save max 10

        Args:
            filename (str): Name of saved checkpoint without ending
            metric (float): A metric in order to compare ckpts and find best. Higher means better,
                            zero is worst possible
            args (dict): arguments to save with checkpoints such as epoch, metric, ...
        """

        new_filename = f"checkpoint-{epoch:04}-{metric:.6f}".replace('.', '_')
        new_ckpt = {'filename': new_filename, 'metric': metric, 'epoch': epoch, 'args': args}
        self.checkpoints.append(new_ckpt)

        # Check if one ckpt has to be removed
        if len(self.checkpoints) > self.max_keep:
            best_ckpt = max(self.checkpoints, key=lambda ckpt: ckpt['metric'])

            sorted_ckpts = sorted(self.checkpoints, key=lambda ckpt: ckpt['epoch'])

            for ckpt in sorted_ckpts:
                if ckpt is not best_ckpt:
                    self.checkpoints.remove(ckpt)

                    # Save new ckpt only if it wasnt removed instantly
                    if ckpt is not new_ckpt:
                        (self.cp_dir / ckpt['filename']).unlink()  # delete old
                        self._write(model, self.cp_dir / new_filename)

                    break
        else:
            self._write(model, self.cp_dir / new_filename)

        # Update checkpoints json
        with self.checkpoints_file.open('w') as file:
            json.dump(self.checkpoints, file, indent=4)


def make_dataset(split, annotation_dir: Path, img_dir: Path, batch_size, n_hourglasses,
                 do_cache=False, seed=None, overfit_batches=None, rotate=[-30, 30],
                 scale=[0.75, 1.25], hue=0.5, contrast=0.5, brightness=0.2, saturation=0.5,
                 heatmap_sigma=1):
    ds = tf.data.Dataset.from_tensor_slices(
        preprocessing.get_mpii_tensor(annotation_dir, img_dir, split=split))

    if overfit_batches:
        ds = ds.take(batch_size*overfit_batches)

    ds = ds.map(preprocessing.ReadImage(), num_parallel_calls=tf.data.AUTOTUNE)

    if do_cache:
        ds = ds.cache()

    ds = ds.map(preprocessing.NewellPreprocessing(
                is_train=False if overfit_batches else (split == 'train'), heatmap_res=64,
                img_res=256, seed=seed, rotate=rotate, scale=scale, hue=hue, contrast=contrast,
                brightness=brightness, saturation=saturation, heatmap_sigma=heatmap_sigma),
                num_parallel_calls=tf.data.AUTOTUNE)

    if not overfit_batches and split == 'train':
        ds = ds.shuffle(1024, seed=seed)

    ds = ds.batch(batch_size=batch_size)
    ds = ds.map(preprocessing.TileLabelBatches(n_hourglasses), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def make_visualization_dataset(annotation_dir: Path, img_dir: Path, batch_size, split='val',
                               heatmap_sigma=1):
    ds = tf.data.Dataset.from_tensor_slices(
        preprocessing.get_mpii_tensor(annotation_dir, img_dir, split=split))
    ds = ds.map(preprocessing.ReadImage(), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(preprocessing.VisualizationPreprocessing(
                img_res=256, hmap_res=64, hmap_sigma=heatmap_sigma),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    return ds


mpii_parts = ['rank', 'rkne', 'rhip', 'lhip', 'lkne', 'lank', 'pelv', 'thrx', 'neck', 'head',
              'rwri', 'relb', 'rsho', 'lsho', 'lelb', 'lwri']


def train_newell(model: keras.Model, batch_size, epochs, summary_writer: SummaryWriter,
                 annotation_dir: Path, img_dir: Path, lr_schedule, overfit_batches=False,
                 output_dir=Path('runs'), seed=None, cache_dataset=False, rotate=[-30, 30],
                 scale=[0.75, 1.25], hue=0.5, contrast=0.5, brightness=0.2, saturation=0.5,
                 heatmap_sigma=1):

    def compute_loss(y_true, y_pred):
        return tnp.mean((y_true - y_pred) ** 2)

    n_stacks = model.n_stacks

    train_ds = make_dataset(split='train', annotation_dir=annotation_dir, img_dir=img_dir,
                            batch_size=batch_size, n_hourglasses=n_stacks, do_cache=cache_dataset,
                            seed=seed, overfit_batches=overfit_batches, rotate=rotate, scale=scale,
                            hue=hue, contrast=contrast, brightness=brightness,
                            saturation=saturation, heatmap_sigma=heatmap_sigma)

    val_ds = make_dataset(split='val', annotation_dir=annotation_dir, img_dir=img_dir,
                          batch_size=batch_size, n_hourglasses=n_stacks, do_cache=False,
                          seed=seed, overfit_batches=overfit_batches, rotate=rotate, scale=scale,
                          heatmap_sigma=heatmap_sigma)

    vis_ds = make_visualization_dataset(split='val', annotation_dir=annotation_dir,
                                        img_dir=img_dir, batch_size=10,
                                        heatmap_sigma=heatmap_sigma).take(1)

    keypoint_vis = KeypointVisualizer()

    def update_metric(heatmap_predictions, keypoint_data, metric: PCKhMetric):
        # Use heatmaps only from last hourglass output
        current_batch_size = heatmap_predictions.shape[0] // n_stacks
        eval_heatmaps = heatmap_predictions[(n_stacks-1)*current_batch_size:]
        metric.add_results(eval_heatmaps, *keypoint_data)

    # Ugly definition outside of loop to prevent step functions from retracing (tf.function)
    pckh_train = PCKhMetric(threshold=0.5, n_keypoints=16)
    pckh_val = PCKhMetric(threshold=0.5, n_keypoints=16)

    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    @tf.function
    def train_step(imgs, heatmaps, keypoint_data):
        with tf.GradientTape() as tape:
            heatmap_predictions = model(imgs, training=True)
            loss_value = compute_loss(heatmaps, heatmap_predictions)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        update_metric(heatmap_predictions, keypoint_data, pckh_train)

        return loss_value

    @tf.function
    def val_step(imgs, heatmaps, keypoint_data):
        heatmap_predictions = model(imgs, training=False)
        loss_value = compute_loss(heatmaps, heatmap_predictions)

        update_metric(heatmap_predictions, keypoint_data, pckh_val)

        return loss_value

    def loop(dataset, num_steps, step_fn, epoch, is_train=False):
        prefix_str = 'train' if is_train else 'val'
        pckh_metric = pckh_train if is_train else pckh_val
        before_step_time = perf_counter()

        # keypoint_data: gt_kps, transforms, head_sizes, visibilities
        for step, (imgs, heatmaps, *keypoint_data) in enumerate(dataset):
            loss_value = step_fn(imgs, heatmaps, keypoint_data)

            after_step_time = perf_counter()

            print(
                f"\r[{prefix_str}] Epoch {epoch+1}/{epochs}, "
                f"Step {step+1}/{len(dataset)}, "
                f"Loss {float(loss_value):2.6f}, "
                f"Duration {(after_step_time - before_step_time)*1000:.3f} ms",
                "           ",
                end='', flush=True)

            before_step_time = perf_counter()

        summary_writer.log_scalars(epoch, {
            f'basics/{prefix_str}_loss': float(loss_value)})

        if is_train:
            summary_writer.log_scalars(epoch, {
                'basics/learning_rate': optimizer._decayed_lr('float32')})

            for var in model.variables:
                if 'perchannel_weights' in var.name:
                    summary_writer.log_histogram(epoch, f"perchannel/{var.name.replace('/', '_')}",
                                                 var)

        pckh_results = pckh_metric.evaluate()
        pckh_all = pckh_metric.evaluate_all()
        summary_writer.log_scalars(epoch, {f'{prefix_str}_pckh/all': pckh_all})
        summary_writer.log_scalars(epoch, {f'{prefix_str}_pckh/mpii':
                                   pckh_metric.evaluate_exclude([
                                       mpii_parts.index('pelv'), mpii_parts.index('thrx')])})
        summary_writer.log_scalars(epoch, {f'{prefix_str}_pckh/ankle':
                                   pckh_metric.evaluate_include([
                                       mpii_parts.index('rank'), mpii_parts.index('lank')])})
        summary_writer.log_scalars(epoch, {f'{prefix_str}_pckh/knee':
                                   pckh_metric.evaluate_include([
                                       mpii_parts.index('rkne'), mpii_parts.index('lkne')])})
        summary_writer.log_scalars(epoch, {f'{prefix_str}_pckh/hip':
                                   pckh_metric.evaluate_include([
                                       mpii_parts.index('rhip'), mpii_parts.index('lhip')])})
        summary_writer.log_scalars(epoch, {f'{prefix_str}_pckh/wrist':
                                   pckh_metric.evaluate_include([
                                       mpii_parts.index('rwri'), mpii_parts.index('lwri')])})
        summary_writer.log_scalars(epoch, {f'{prefix_str}_pckh/shoulder':
                                   pckh_metric.evaluate_include([
                                       mpii_parts.index('lsho'), mpii_parts.index('rsho')])})

        for i in range(len(mpii_parts)):
            summary_writer.log_scalars(epoch, {
                f'{prefix_str}_pckh_parts/{mpii_parts[i]}': pckh_results[i]})

        return pckh_all

    checkpoint_manager = CheckpointManager(output_dir / 'checkpoints', keep_best=True, max_keep=5)

    for epoch in range(epochs):
        loop(train_ds, len(train_ds), train_step, epoch, is_train=True)

        if not overfit_batches:
            metric = loop(val_ds, len(val_ds), val_step, epoch, is_train=False)

            if (epoch+1) % 5 == 0:
                checkpoint_manager.save(model=model, epoch=epoch, metric=float(metric))

        if (epoch) % 5 == 0:
            # Visualization
            for (imgs, imgs_std, hmaps, img_tf, hmap_tf, *kp_data) in vis_ds:
                imgs_gt = keypoint_vis.from_kps(imgs, kp_data[0], img_tf, kp_data[2])

                hmap_pred = model(imgs_std, training=False)
                current_batch_size = hmap_pred.shape[0] // n_stacks
                hmap_pred = hmap_pred[(n_stacks-1)*current_batch_size:]

                imgs_pred = keypoint_vis.from_hmaps(imgs, hmap_pred, img_tf, hmap_tf, kp_data[2])

                summary_writer.log_images(epoch, 'images/ground_truth', imgs_gt)
                summary_writer.log_images(epoch, 'images/predictions', imgs_pred)

                # Log heatmaps of first image only, else too many
                summary_writer.log_images(epoch, 'heatmaps/ground_truth',
                                          hmaps_to_grayscale_img(hmaps[0]))
                summary_writer.log_images(epoch, 'heatmaps/predictions',
                                          hmaps_to_grayscale_img(hmap_pred[0]))

        pckh_train.reset()
        pckh_val.reset()


if __name__ == '__main__':
    # Stupid tensorflow bug
    from packaging import version
    if version.parse(tf.__version__) >= version.parse('2.5.0'):
        tf.experimental.numpy.experimental_enable_numpy_behavior()

    parser = argparse.ArgumentParser(description='Train hourglass networks.')
    parser.add_argument('-c', '--config', type=Path, help='Path to config.yaml file')
    parser.add_argument('-b', '--batch_size', type=int, help='Training batch size', default=None)
    parser.add_argument('-e', '--epochs', type=int, help='Training number of epochs', default=None)
    parser.add_argument('-l', '--learning_rate', type=float, help='Learning rate', default=None)
    parser.add_argument('-o', '--overfit_batches', type=int, const=1, default=0, nargs='?',
                        help='Crop the dataset to the batch size and force model to (hopefully)'
                             ' overfit.')
    parser.add_argument('-v', '--visible_device', type=int, default=0)

    # Get argsparse dictionary and remove all values that are None
    # Then update the config dict coming from the yaml file with argsparse dict
    args_dict = {key: val for key, val in vars(parser.parse_args()).items() if val is not None}
    config = YAMLConfig(args_dict['config'], args_dict)

    # This circumvents some bugs
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[config.visible_device], 'GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    mp_dir = Path.home() / config.mpii_annot_dir
    img_dir = Path.home() / config.mpii_image_dir

    log_dir = Path(config.log_dir)
    log_dir.mkdir(exist_ok=True)

    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_dir = log_dir / current_time_str
    output_dir.mkdir(exist_ok=False)

    if config.logger == 'wandb':
        summary_writer = WandBWriter(config.wandb_project_name)
    else:
        summary_writer = TFSummaryWriter(output_dir)

    summary_writer.log_config(config._dict)
    print(config)

    model = BulatTiny()

    train_newell(model, batch_size=config.batch_size, epochs=config.epochs, annotation_dir=mp_dir,
                 img_dir=img_dir, seed=config.seed, overfit_batches=config.overfit_batches,
                 lr_schedule=ConstantSchedule(config.learning_rate), summary_writer=summary_writer,
                 cache_dataset=config.cache_dataset, output_dir=output_dir, rotate=config.rotate,
                 scale=config.scale, hue=config.hue, brightness=config.brightness,
                 contrast=config.contrast, saturation=config.saturation,
                 heatmap_sigma=config.heatmap_sigma)
