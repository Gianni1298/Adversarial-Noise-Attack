import unittest
import importlib

from torchvision.io import read_image

from src import generator


class TestAdversarialImageGenerator(unittest.TestCase):
    def setUp(self):
        # Reload the module before each test
        importlib.reload(generator)

    def test_modelNotSupported(self):
        with self.assertRaises(ValueError):
            generator.AdversarialImageGenerator("resnet51")

    def test_initialization(self):
        test_generator = generator.AdversarialImageGenerator("resnet50")
        self.assertIsNotNone(test_generator.model)


    def test_prediction(self):
        test_generator = generator.AdversarialImageGenerator("resnet50")
        image_path = '../data/sample_images/burrito.jpg'
        image = read_image(image_path)
        input_image = test_generator.preprocess(image).unsqueeze(0)
        class_id, name, score = test_generator._AdversarialImageGenerator__prediction(input_image)
        assert name == 'burrito'

    def test_validate_target_class(self):
        test_generator = generator.AdversarialImageGenerator("resnet50")
        with self.assertRaises(ValueError):
            test_generator._AdversarialImageGenerator__validate_target_class("non existent class")
        assert test_generator._AdversarialImageGenerator__validate_target_class("black swan") == 100

    def test_generate_adversarial_image(self):
        test_generator = generator.AdversarialImageGenerator("resnet50")
        image_path = '../data/sample_images/bird.jpg'
        class_id =  test_generator.generate_adversarial_image(image_path, "goldfish")
        assert class_id == 1


if __name__ == '__main__':
    unittest.main()
