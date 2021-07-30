import tensorflow as tf
import tensorflow.experimental.numpy as tnp  # type: ignore
import unittest

from hourglass.data.transforms import apply_transform, scaling_transform, translation_transform, \
    rotation_transform, newell_cropping_transform, horizontal_flip_transform, permutation_transform


# Stupid tensorflow bug
from packaging import version
if version.parse(tf.__version__) >= version.parse('2.5.0'):
    tf.experimental.numpy.experimental_enable_numpy_behavior()


class TestTransforms(unittest.TestCase):
    def test_apply_2D(self):
        t1 = tnp.array([
            [0, -2, 0],
            [1,  0, 0],
            [0,  0, 3]
        ], dtype=float)

        t2 = tnp.array([
            [0.5, 0,    0],
            [0,   0.25, 0],
            [0,   0,   0.1]
        ])

        input = tnp.array([[3, 7]])
        correct = tnp.array([[-7, 0.75]])
        output = apply_transform(input, t2 @ t1)
        self.assertTrue(tnp.allclose(correct, output))

    def test_apply_2D_2(self):
        t = tnp.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ], dtype=float)

        input = tnp.array([
            [1000, 500],
            [320, 150],
            [600, 400],
            [1920, 1280]
        ])

        correct = tnp.array([
            [500, 1000],
            [150, 320],
            [400, 600],
            [1280, 1920]
        ])

        output = apply_transform(input, t)
        self.assertTrue(tnp.allclose(correct, output))

    def test_apply_3D(self):
        # two batches
        trafos = tnp.array([
            [[0, 1, 0],
             [1, 0, 0],
             [0, 0, 1]],
            [[-1, 0, 0],
             [0, 2, 0],
             [0, 0, 1]],
        ], dtype=float)

        inputs = tnp.array([
            [[1000, 500],
             [320, 150]],
            [[600, 400],
             [1920, 1280]]
        ])

        correct = tnp.array([
            [[500, 1000],
             [150, 320]],
            [[-600, 800],
             [-1920, 2560]]
        ])

        output = apply_transform(inputs, trafos)
        self.assertTrue(tnp.allclose(correct, output))

    def test_apply_3D_single_trafo(self):
        # two batches, single trafo
        t = tnp.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ], dtype=float)

        inputs = tnp.array([
            [[1000, 500],
             [320, 150]],
            [[600, 400],
             [1920, 1280]]
        ])

        correct = tnp.array([
            [[500, 1000],
             [150, 320]],
            [[400, 600],
             [1280, 1920]]
        ])

        output = apply_transform(inputs, t)
        self.assertTrue(tnp.allclose(correct, output))
        self.assertEqual(correct.shape, output.shape)

    def test_apply_wrong_transform_shape(self):
        transform = tnp.ones([3])

        inputs = tnp.array([
            [[1000, 500],
             [320, 150]],
            [[600, 400],
             [1920, 1280]]
        ])
        with self.assertRaises(ValueError):
            apply_transform(inputs, transform)

    def test_apply_wrong_input_shape(self):
        transform = tnp.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ], dtype=float)

        inputs = tnp.array([100, 100])
        with self.assertRaises(ValueError):
            apply_transform(inputs, transform)

    def test_scaling(self):
        scaling = 0.5
        input = tnp.array([[1., 1.]])
        correct = tnp.array([[0.5, 0.5]])

        m_scaling = scaling_transform(scaling)
        output = apply_transform(input, m_scaling)
        self.assertTrue(tnp.allclose(correct, output))

    def test_translation(self):
        translation = -2.
        input = tnp.array([[1., 1.]])
        correct = tnp.array([[-1., -1.]])

        m_translation = translation_transform(translation, translation)
        output = apply_transform(input, m_translation)
        self.assertTrue(tnp.allclose(correct, output))

    def test_rotation(self):
        angle = -90
        resolution = 4
        input = tnp.array([[3, 3]])
        correct = tnp.array([[3., 1.]])

        m_rotation = rotation_transform(angle, resolution)
        output = apply_transform(input, m_rotation)
        self.assertTrue(tnp.allclose(correct, output))

    def test_newell_cropping(self):
        keypoint = tnp.array([[5, 5]])
        center = tnp.array([4, 4])
        body_scale = 4 / 200
        resolution = 2
        correct = tnp.array([[1.5, 1.5]])

        m_transform = newell_cropping_transform(center, body_scale, resolution)
        output = apply_transform(keypoint, m_transform)
        self.assertTrue(tnp.allclose(correct, output))

    def test_flipping(self):
        keypoints = tnp.array([
            [1, 0],
            [6, 0],
            [2, 3],
            [5, 3],
            [3, 6],
            [4, 6]
        ])

        correct = tnp.array([
            [6, 0],
            [1, 0],
            [5, 3],
            [2, 3],
            [4, 6],
            [3, 6]
        ])

        m_transform = horizontal_flip_transform(8)
        output = apply_transform(keypoints, m_transform)
        self.assertTrue(tnp.allclose(correct, output))

    def test_permutation(self):
        keypoints = tnp.array([
            [1, 0],
            [6, 0],
            [2, 3],
            [5, 3],
            [3, 6],
            [4, 6]
        ])

        permute_t = permutation_transform([1, 0, 5, 4, 2, 3])

        correct = tnp.array([
            [6, 0],
            [1, 0],
            [4, 6],
            [3, 6],
            [2, 3],
            [5, 3],
        ])

        self.assertTrue(tnp.allclose(correct, permute_t @ keypoints))


if __name__ == '__main__':
    unittest.main()
