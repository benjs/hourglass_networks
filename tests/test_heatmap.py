import tensorflow as tf
import tensorflow.experimental.numpy as tnp  # type: ignore
import unittest

from hourglass.data.heatmap import HeatmapEvaluator, HeatmapGenerator, _unravel_index_2D


# Stupid tensorflow bug
from packaging import version
if version.parse(tf.__version__) >= version.parse('2.5.0'):
    tf.experimental.numpy.experimental_enable_numpy_behavior()


def _batched_argmax(A):
    # tnp.column_stack(tnp.unravel_index(A.reshape(A.shape[0], -1).argmax(axis=1), A[0].shape))
    # tnp does not support column_stack(), hence use loop and save some dev time
    # https://stackoverflow.com/questions/30589211/numpy-argmax-over-multiple-axes-without-loop
    argmax_vals = []

    for i in range(A.shape[0]):
        argmax_vals.append(_unravel_index_2D(tnp.argmax(A[i], axis=None), tf.shape(A[i])))

    return tnp.array(argmax_vals)


class TestHeatmapGenerator(unittest.TestCase):
    def test_unravel_index_2D(self):
        indices = tnp.arange(6)
        shape = tnp.array([2, 3])

        correct = tnp.array([
            [0, 0], [0, 1], [0, 2],
            [1, 0], [1, 1], [1, 2]
        ])

        for i in range(indices.shape[0]):
            self.assertTrue(tnp.allclose(_unravel_index_2D(indices[i], shape), correct[i]))

    def test_batched_argmax(self):
        input = tnp.array([
            [[0, 1], [0, 0]],
            [[0, 0], [0, 1]]
        ], dtype=tnp.float32)

        correct = tnp.array([[0, 1], [1, 1]])
        output = _batched_argmax(input)
        self.assertTrue(tnp.allclose(correct, output))

    def test_location_and_value(self):
        generator = HeatmapGenerator(sigma=1, n_keypoints=2, resolution=5)

        kps = tnp.array([[1, 2], [3, 4]])
        visible = tnp.array([1, 1])
        heatmaps = generator(keypoints=kps, visible=visible)

        # transpose from channels last to channels first
        heatmaps = tnp.transpose(heatmaps, [2, 0, 1])

        correct_max = tnp.array([1, 1])
        correct_argmax = tnp.array([[2, 1], [4, 3]])  # switched bc mpii dataset kps are [col, row]

        max = tnp.max(heatmaps, axis=(1, 2))
        argmax = _batched_argmax(heatmaps)

        self.assertTrue(tnp.allclose(correct_max, max))
        self.assertTrue(tnp.allclose(correct_argmax, argmax))

    def test_visible(self):
        generator = HeatmapGenerator(sigma=1, n_keypoints=2, resolution=5)

        kps = tnp.array([[1, 2], [1, 2]])
        visible = tnp.array([1, 0])
        heatmaps = generator(keypoints=kps, visible=visible)

        # transpose from channels last to channels first
        heatmaps = tnp.transpose(heatmaps, [2, 0, 1])

        correct_max = tnp.array([1, 0])
        max = tnp.max(heatmaps, axis=(1, 2))
        self.assertTrue(tnp.allclose(correct_max, max))

    def test_batched_heatmap_shape(self):
        generator = HeatmapGenerator(sigma=1, n_keypoints=2, resolution=5)

        kps = tnp.array([
            [[1, 2], [3, 4]],
            [[4, 3], [2, 1]]
        ])

        visible = tnp.array([[1, 1], [1, 1]])
        heatmaps = generator(keypoints=kps, visible=visible)

        self.assertEqual(heatmaps.shape, tf.TensorShape((2, 5, 5, 2)))

    def test_batched_heatmaps_max(self):
        generator = HeatmapGenerator(sigma=1, n_keypoints=2, resolution=5)

        kps = tnp.array([
            [[1, 2], [3, 4]],
            [[4, 3], [2, 1]]
        ])

        visible = tnp.array([[1, 1], [1, 0]])
        heatmaps = generator(keypoints=kps, visible=visible)

        max = tnp.max(heatmaps, axis=(1, 2))
        correct_max = tnp.array([[1, 1], [1, 0]])
        self.assertTrue(tnp.allclose(max, correct_max))


class TestHeatmapEvaluator(unittest.TestCase):

    def test_location(self):
        hmap_generator = HeatmapGenerator(sigma=1, n_keypoints=4, resolution=64)
        hmap_evaluator = HeatmapEvaluator()

        # [column, row] (from mpii)
        input_keypoints = tnp.array([
            [20, 30],
            [25, 25],
            [42, 21],
            [10, 3]
        ])

        # [row, column] (numpy notation)
        correct = input_keypoints

        heatmaps = hmap_generator(input_keypoints, tnp.ones(4))[tnp.newaxis, :, :, :]
        output = hmap_evaluator(heatmaps)
        self.assertTrue(tnp.allclose(correct, output))

    def test_location2(self):
        hmap_evaluator = HeatmapEvaluator()

        # batched heatmaps
        input_heatmaps = tnp.array([
            [[[0, 0, 0],
              [0, 0, 1],
              [0, 0, 0]]],
            [[[0, 0, 0],
              [0, 1, 0],
              [0, 0, 0]]],
        ]).transpose([0, 2, 3, 1])  # to channels last

        correct = tnp.array([
            [[2, 1]],
            [[1, 1]]
        ])

        output = hmap_evaluator(input_heatmaps)
        self.assertTrue(tnp.allclose(correct, output))

    def test_shape(self):
        hmap_generator = HeatmapGenerator(sigma=1, n_keypoints=4, resolution=64)
        hmap_evaluator = HeatmapEvaluator()

        # [column, row] (from mpii)
        input_keypoints1 = tnp.array([
            [20, 30],
            [25, 25],
            [42, 21],
            [10, 3]
        ])

        input_keypoints2 = tnp.array([
            [20, 8],
            [25, 12],
            [42, 14],
            [10, 30]
        ])

        heatmaps = tnp.array([
            hmap_generator(input_keypoints1, tnp.ones(4)),
            hmap_generator(input_keypoints2, tnp.ones(4))
        ])

        output = hmap_evaluator(heatmaps)
        self.assertEqual(tf.TensorShape([2, 4, 2]), output.shape)

        # Test outputs because why not
        self.assertTrue(tnp.allclose(tnp.array([input_keypoints1, input_keypoints2]), output))


if __name__ == '__main__':
    unittest.main()
