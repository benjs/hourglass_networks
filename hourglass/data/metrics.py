import tensorflow as tf
import tensorflow.experimental.numpy as tnp  # type: ignore

from hourglass.data.heatmap import HeatmapEvaluator
from hourglass.data.transforms import apply_transform, scaling_transform


class PCKhMetric:
    def __init__(self, threshold=0.5, n_keypoints=16):
        self.evaluator = HeatmapEvaluator()
        self.threshold = threshold

        self.total = tf.Variable(tf.zeros(n_keypoints), trainable=False)
        self.correct = tf.Variable(tf.zeros(n_keypoints), trainable=False)

    def _get_distances(self, ground_truth, predictions):
        if ground_truth.shape != predictions.shape:
            raise ValueError('Ground truth and predictions must have same shape.'
                             f' {ground_truth.shape} != {predictions.shape}')

        N, C, _ = ground_truth.shape
        ground_truth = tnp.reshape(ground_truth, (N*C, 2))
        predictions = tnp.reshape(predictions, (N*C, 2))

        distances = tnp.sum((ground_truth - predictions)**2, axis=1)**0.5

        return tnp.reshape(distances, (N, C))

    def add_results(self, heatmaps, gt_keypoints, transforms, head_sizes, visibilities):
        """Calculate the percentage of correct keypoints normalized by head bone link (PCKh) metric

        Args:
            heatmaps (tf.Tensor): Bx64x64xC heatmaps
            gt_keypoints (tf.Tensor): BxCx2 keypoints
            transforms (tf.Tensor): Bx3x3 transformation matrices
            head_sizes (tf.Tensor): B head sizes
            visibilities (tf.Tensor): BxC binary visibilities
        """
        N, C, _ = gt_keypoints.shape

        if heatmaps.shape[0] != gt_keypoints.shape[0]:
            raise ValueError('Heatmaps and ground truth keypoints have different batch dimensions.'
                             f' Dimensions are {heatmaps.shape} and {gt_keypoints.shape}')

        predictions_img = apply_transform(self.evaluator(heatmaps), tf.linalg.inv(transforms))

        distances = self._get_distances(gt_keypoints, predictions_img)
        normalized_dists = distances / tnp.repeat(head_sizes[:, tnp.newaxis], C, axis=1)

        # invisible joints are not counted
        total_per_joint = tnp.sum(visibilities, axis=0)
        batched_correct_per_joint = tnp.where(normalized_dists < self.threshold,
                                              visibilities, tnp.zeros_like(visibilities))

        correct_per_joint = tnp.sum(batched_correct_per_joint, axis=0)

        self.total.assign_add(total_per_joint)
        self.correct.assign_add(correct_per_joint)

    def evaluate(self):
        return self.correct / self.total

    def evaluate_all(self):
        return tnp.sum(self.correct) / tnp.sum(self.total)

    def _retreive_include_mask(self, included_joints: list = []):
        mask = tnp.zeros_like(self.total)
        for joint in included_joints:
            mask = mask + tf.one_hot(joint, mask.shape[0])
        return mask

    def evaluate_exclude(self, excluded_joints: list = []):
        mask = tnp.ones_like(self.total) - self._retreive_include_mask(excluded_joints)
        return tnp.dot(self.correct, mask) / tnp.dot(self.total, mask)

    def evaluate_include(self, included_joints: list = []):
        mask = self._retreive_include_mask(included_joints)
        return tnp.dot(self.correct, mask) / tnp.dot(self.total, mask)

    def reset(self):
        self.total.assign_sub(self.total)
        self.correct.assign_sub(self.correct)
