from inspect import Attribute
import shutil
import unittest
import yaml
from hourglass.utils.config import YAMLConfig
from pathlib import Path


class TestYAMLConfig(unittest.TestCase):

    def setUp(self):
        self.testing_dir = Path('test_yamlconfig')
        self.testing_dir.mkdir(exist_ok=False)

    def tearDown(self):
        shutil.rmtree(self.testing_dir, ignore_errors=True)

    def test_basic_types(self):
        test_int = 10
        test_float = 4.5678
        test_str = "string"

        config_str = f"""
test_int: {test_int}
test_str: "{test_str}"
test_float: {test_float}
test_list: [-30, 30]
test_dict: {{a: 10}}
        """

        cfg_path = self.testing_dir / "testconfig.yaml"
        with cfg_path.open('w') as file:
            file.write(config_str)

        config = YAMLConfig(cfg_path)
        self.assertEqual(config.test_int, test_int)
        self.assertEqual(config.test_str, test_str)
        self.assertEqual(config.test_float, test_float)
        self.assertEqual(config.test_list, [-30, 30])
        self.assertEqual(config.test_dict, {"a": 10})

    def test_overwrite(self):
        config1_str = """
test: 1
test2: 3
        """

        config2_str = """
overwrite_config: config1.yaml
test: 2
        """

        config1_path = self.testing_dir / 'config1.yaml'
        with config1_path.open('w') as file:
            file.write(config1_str)

        config2_path = self.testing_dir / 'config2.yaml'
        with config2_path.open('w') as file:
            file.write(config2_str)

        config1 = YAMLConfig(config1_path)
        self.assertEqual(config1.test, 1)
        self.assertEqual(config1.test2, 3)

        config2 = YAMLConfig(config2_path)
        self.assertEqual(config2.test, 2)
        self.assertEqual(config2.test2, 3)

    def test_no_recursion(self):
        config1_str = """
overwrite_config: config1.yaml
test: 1
test2: 3
        """

        config1_path = self.testing_dir / 'config1.yaml'
        with config1_path.open('w') as file:
            file.write(config1_str)

        YAMLConfig(config1_path)

    def test_param_not_exists(self):
        config1_str = """
a: 1
        """

        config1_path = self.testing_dir / 'config1.yaml'
        with config1_path.open('w') as file:
            file.write(config1_str)

        config = YAMLConfig(config1_path)
        with self.assertRaises(AttributeError):
            config.b

        self.assertEqual(config.a, 1)
