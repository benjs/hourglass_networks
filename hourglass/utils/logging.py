import tensorflow as tf
import wandb
import yaml

from pathlib import Path


class SummaryWriter:
    """Simple abstract class to enable switching between loggers
    """
    def log_config(self, config: dict):
        raise NotImplementedError()

    def log_scalars(self, step, key_values: dict):
        raise NotImplementedError()

    def log_images(self, step: int, name: str, image_data, caption=None):
        raise NotImplementedError()

    def log_histogram(self, step: int, name: str, histogram_data):
        raise NotImplementedError()


class WandBWriter(SummaryWriter):
    """Logger for Weights and Biases (wandb.com)
    """

    def __init__(self, project_name: str):
        super().__init__()

        wandb.login()
        wandb.init(project=project_name)

    def log_config(self, config: dict):
        wandb.config.update(config)

    def log_scalars(self, step: int, key_values: dict):
        key_values.update({'step': step})
        wandb.log(key_values)

    def log_images(self, step: int, name: str, image_data, caption=None):
        log_dict = {'step': step}
        for i in range(image_data.shape[0]):
            log_dict.update({f"{name}_{i}": wandb.Image(image_data[i], caption=caption)})

        wandb.log(log_dict)

    def log_histogram(self, step: int, name: str, histogram_data):
        wandb.log({
            name: wandb.Histogram(histogram_data),
            'step': step
        })


class TFSummaryWriter(SummaryWriter):
    """Tensorboard logger
    """
    def __init__(self, save_dir: Path, max_image_outputs=10):
        self.writer = tf.summary.create_file_writer(str(save_dir))
        self.max_image_outputs = max_image_outputs

    def log_config(self, config: dict):
        with self.writer.as_default():
            tf.summary.text('config', yaml.dump(config))

    def log_scalars(self, step, key_values: dict):
        with self.writer.as_default():
            for key in key_values:
                tf.summary.scalar(key, key_values[key], step=step)

    def log_images(self, step: int, name: str, image_data, caption=None):
        with self.writer.as_default():
            tf.summary.image(name, image_data, step=step, max_outputs=self.max_image_outputs,
                             description=caption)

    def log_histogram(self, step: int, name: str, histogram_data):
        with self.writer.as_default():
            tf.summary.histogram(name, histogram_data, step=step)
