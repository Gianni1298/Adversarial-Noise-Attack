import unittest
import importlib

from matplotlib import pyplot as plt

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

    def test_load_and_preprocess(self):
        test_generator = generator.AdversarialImageGenerator("resnet50")
        image_path = '../data/sample_images/bird.jpg'
        image = test_generator._AdversarialImageGenerator__load_and_preprocess(image_path)
        print(image)

    def test_plot_image(self):
        test_generator = generator.AdversarialImageGenerator("resnet50")
        image_path = '../data/sample_images/bird.jpg'
        image = test_generator._AdversarialImageGenerator__load_and_preprocess(image_path)
        fig = test_generator._AdversarialImageGenerator__plot_image(image)

        plt.show()

    def test_prediction(self):
        test_generator = generator.AdversarialImageGenerator("resnet50")
        image_path = '../data/sample_images/burrito.jpg'
        image = test_generator._AdversarialImageGenerator__load_and_preprocess(image_path)
        name, score = test_generator._AdversarialImageGenerator__prediction(image)
        assert name == 'burrito'

    def test_validate_target_class(self):
        test_generator = generator.AdversarialImageGenerator("resnet50")
        with self.assertRaises(ValueError):
            test_generator._AdversarialImageGenerator__validate_target_class(34568)
        assert test_generator._AdversarialImageGenerator__validate_target_class(100) == 'black swan'

    def test_generate_adversarial_image(self):
        # Write tests for the generate_adversarial_image method
        pass

    def test_plot_images(self):
        # Write tests for the plot_images method
        pass

    def test_save_adversarial_image(self):
        # Write tests for the save_adversarial_image method
        pass


if __name__ == '__main__':
    unittest.main()
