import shutil
import unittest
import tensorflow.keras as keras
from tensorflow.python.ops.gen_array_ops import shape

from hourglass.training import MultiStepSchedule, CheckpointManager
from pathlib import Path


class TestLRSchedules(unittest.TestCase):
    def test_multistep(self):
        ms_sched = MultiStepSchedule(steps_per_epoch=1, init_lr=1., gamma=1/2,
                                     milestones=[10, 20, 30])

        # 40 steps
        for step in range(50):
            if step < 10:
                self.assertEqual(ms_sched(step), 1)
            elif step < 20:
                self.assertEqual(ms_sched(step), 1/2)
            elif step < 30:
                self.assertEqual(ms_sched(step), 1/4)
            elif step >= 30:
                self.assertEqual(ms_sched(step), 1/8)


class TestCheckpointManager(unittest.TestCase):
    def setUp(self):
        self.testing_dir = Path('test_ckptmngr')

        # Create dummy model
        inputs = keras.Input(shape=(3,))
        x = keras.layers.Dense(4)(inputs)
        outputs = keras.layers.Dense(5)(x)
        self.dummy_model = keras.Model(inputs=inputs, outputs=outputs)

    def tearDown(self):
        shutil.rmtree(self.testing_dir, ignore_errors=True)

    def test_keep_best(self):
        ckpt_manager = CheckpointManager(self.testing_dir, keep_best=True, max_keep=2)

        ckpt_manager.save(self.dummy_model, epoch=10, metric=1)
        ckpt_manager.save(self.dummy_model, epoch=20, metric=0.5)
        fn_should_not_exist = ckpt_manager.checkpoints[1]['filename']
        ckpt_manager.save(self.dummy_model, epoch=30, metric=0.7)

        self.assertEqual(ckpt_manager.checkpoints[0]['epoch'], 10)
        self.assertEqual(ckpt_manager.checkpoints[1]['epoch'], 30)

        self.assertTrue((self.testing_dir / ckpt_manager.checkpoints[0]['filename']).exists())
        self.assertTrue((self.testing_dir / ckpt_manager.checkpoints[1]['filename']).exists())
        self.assertFalse((self.testing_dir / fn_should_not_exist).exists())
        self.assertTrue((ckpt_manager.checkpoints_file).exists())

    def test_keep_max_1(self):
        ckpt_manager = CheckpointManager(self.testing_dir, keep_best=True, max_keep=1)

        ckpt_manager.save(self.dummy_model, epoch=10, metric=1)
        ckpt_manager.save(self.dummy_model, epoch=20, metric=0.5)
        ckpt_manager.save(self.dummy_model, epoch=30, metric=0.7)

        self.assertEqual(len(ckpt_manager.checkpoints), 1)
        self.assertEqual(ckpt_manager.checkpoints[0]['epoch'], 10)

        self.assertTrue((self.testing_dir / ckpt_manager.checkpoints[0]['filename']).exists())
        self.assertTrue((ckpt_manager.checkpoints_file).exists())


if __name__ == '__main__':
    unittest.main()
