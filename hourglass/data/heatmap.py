import tensorflow as tf
import tensorflow.experimental.numpy as tnp  # type: ignore

from hourglass.data.transforms import scaling_transform, apply_transform


def _unravel_index_2D(index, shape):
    # tnp does not contain np.unravel_index. This is a 2D version of the latter
    return tnp.array([index//shape[1] % shape[0], index % shape[1]])


class HeatmapGenerator:
    """Generates heatmaps given keypoints."""

    def __init__(self, sigma=1, n_keypoints=16, resolution=64):
        self.sigma = sigma
        self.resolution = resolution

        # Position encoding matrices
        # for example "vs":
        #   0, 1, 2, 3, ...
        #   0, 1, 2, 3, ...
        #   ...
        self.pos_vs = tnp.tile(
            tnp.arange(resolution, dtype=tnp.float32),
            (n_keypoints, resolution, 1))

        self.pos_us = tnp.transpose(self.pos_vs, [0, 2, 1])  # self.pos_vs.transpose(0, 2, 1)

    def __call__(self, keypoints, visible):
        """Generate a batch of heatmaps

        Args:
            keypoints (np.array): List of keypoints. Shape: (C, 2) or (B, C, 2)
            visible (np.array): List of visibility binaries. Shape: (C) or (B, C)
        """

        if len(keypoints.shape) == 3:
            N, C, _ = keypoints.shape
            keypoints = tnp.reshape(keypoints, (N*C, 2))
            visible = tnp.reshape(visible, (N*C))
        elif len(keypoints.shape) == 2:
            N = None
            C, _ = keypoints.shape
        else:
            raise ValueError(f"Keypoints must have shape BxCx2 or Cx2"
                             f"but have shape {keypoints.shape}")

        # keypoints in mpii dataset come in as [column, row]
        # tf/numpy in contrast view tensors as [row, column] -> switch indices
        # additionally add two dimensions
        kp_us = keypoints[:, 1][:, tnp.newaxis, tnp.newaxis]
        kp_vs = keypoints[:, 0][:, tnp.newaxis, tnp.newaxis]

        if N is not None:
            pos_us = tnp.tile(self.pos_us, (N, 1, 1))
            pos_vs = tnp.tile(self.pos_vs, (N, 1, 1))
        else:
            pos_us = self.pos_us
            pos_vs = self.pos_vs

        us = pos_us - kp_us
        vs = pos_vs - kp_vs

        heatmaps = \
            tnp.exp(-(us*us + vs*vs)/(2 * self.sigma**2)) * visible[:, tnp.newaxis, tnp.newaxis]

        # transpose to channels last and restore batch dimension
        if N is not None:
            heatmaps = tnp.reshape(heatmaps, (N, C, self.resolution, self.resolution))
            heatmaps = tnp.transpose(heatmaps, [0, 2, 3, 1])  # channels last
        else:
            heatmaps = tnp.transpose(heatmaps, [1, 2, 0])

        return heatmaps


class HeatmapEvaluator:
    """Retreives keypoints given heatmaps."""

    def __call__(self, heatmaps: tnp.array):
        N, H, W, C = heatmaps.shape

        # Transpose to channels first
        heatmaps = tnp.transpose(heatmaps, (0, 3, 1, 2))

        # Reshape each heatmap to a single dimension e.g. 3x3 to 9
        # Combine batches of heatmaps to one larger list of heatmaps
        # Then take argmax and retreive 2D keypoint through unraveling
        keypoint_preds = _unravel_index_2D(
            tnp.argmax(tnp.reshape(heatmaps, (N*C, H*W)), axis=1), (H, W))

        # Switch row and column vector because of mpii notation
        keypoint_preds = keypoint_preds[[1, 0], :]

        # Transpose from 2xN*C to N*Cx2 and reshape into original batches to NxCx2
        return keypoint_preds.transpose((1, 0)).reshape(N, C, 2)
