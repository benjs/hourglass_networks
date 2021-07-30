import tensorflow as tf
import tensorflow.experimental.numpy as tnp  # type: ignore
import unittest

from hourglass.data.heatmap import HeatmapGenerator
from hourglass.data.metrics import PCKhMetric
from hourglass.data.transforms import apply_transform


# Stupid tensorflow bug
from packaging import version
if version.parse(tf.__version__) >= version.parse('2.5.0'):
    tf.experimental.numpy.experimental_enable_numpy_behavior()


class TestPCKhMetric(unittest.TestCase):
    def test_evaluate(self):
        # imgs: 200x200
        # head_size: 100
        # threshold: 0.5

        gt_keypoints = tnp.array([
            [[100, 120],
             [50,   30],
             [190, 130],
             [155,  10]],
            [[30, 170],
             [60, 111],
             [167, 10],
             [95, 100]]
        ], dtype=tnp.float32)

        # same coord frame as gt_keypoints
        kp_predictions = tnp.array([
            [[10, 120],   # out, vis,   invis
             [50,  30],   # in,  vis,   invis
             [0,  130],   # out, invis, vis
             [155, 10]],  # in,  invis, vis

            [[30, 170],   # in,  vis,   invis
             [60, 111],   # in,  invis, vis
             [67,  10],   # out, vis,   invis
             [95,  10]]   # out, invis, vis
        ], dtype=tnp.float32)

        # after kps with both visibilites added,
        # results should be:
        #
        #     total | correct
        # --------------------
        # 1 |   2       1
        # 2 |   2       2
        # 3 |   2       0
        # 4 |   2       1

        visibilities1 = tnp.array([
            [1, 1, 0, 0],
            [1, 0, 1, 0]
        ], dtype=tnp.float32)

        visibilities2 = tnp.array([
            [0, 0, 1, 1],
            [0, 1, 0, 1]
        ], dtype=tnp.float32)

        head_sizes = tnp.array([100, 100], dtype=tnp.float32)

        transforms = tnp.array([
            [[0, 1, 0],
             [1, 0, 0],
             [0, 0, 1]],
            [[2, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
        ], dtype=tnp.float32)

        # Transformed through data augmentation
        kp_predictions = apply_transform(kp_predictions, transforms)

        hmap_generator = HeatmapGenerator(sigma=1, n_keypoints=4, resolution=200)
        heatmaps1 = hmap_generator(kp_predictions, visibilities1)
        heatmaps2 = hmap_generator(kp_predictions, visibilities2)

        pckh05 = PCKhMetric(0.5, 4)
        pckh05.add_results(heatmaps1, gt_keypoints, transforms, head_sizes, visibilities1)
        pckh05.add_results(heatmaps2, gt_keypoints, transforms, head_sizes, visibilities2)
        metrics = pckh05.evaluate()

        correct_pckh = tnp.array([0.5, 1, 0, 0.5])
        correct_total = tnp.array([2, 2, 2, 2])
        correct_correct = tnp.array([1, 2, 0, 1])
        correct_all = 4/8

        self.assertTrue(tnp.allclose(correct_pckh, metrics))
        self.assertTrue(tnp.allclose(correct_total, pckh05.total))
        self.assertTrue(tnp.allclose(correct_correct, pckh05.correct))
        self.assertTrue(tnp.allclose(pckh05.evaluate_all(), correct_all))

        pckh05.reset()

        self.assertTrue(tnp.allclose(pckh05.total, tnp.zeros_like(correct_total)))
        self.assertTrue(tnp.allclose(pckh05.correct, tnp.zeros_like(correct_correct)))

    def test_evaluate_exclude(self):
        pckh05 = PCKhMetric(0.5, 4)
        pckh05.correct = tnp.array([1, 2, 3, 4])
        pckh05.total = tnp.array([5, 8, 3, 5])

        self.assertTrue(tnp.allclose(pckh05.evaluate_exclude([1, 2]), 0.5))

    def test_evaluate_include(self):
        pckh05 = PCKhMetric(0.5, 4)
        pckh05.correct = tnp.array([1, 2, 3, 4])
        pckh05.total = tnp.array([5, 8, 3, 5])

        self.assertTrue(tnp.allclose(pckh05.evaluate_include([1, 2]), 5/11))


if __name__ == '__main__':
    unittest.main()
