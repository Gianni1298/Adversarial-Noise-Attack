import unittest
from src.generator import AdversarialImageGenerator


class TestAdversarialImageGenerator(unittest.TestCase):

    def test_modelNotSupported(self):
        with self.assertRaises(ValueError):
            AdversarialImageGenerator("resnet51")

    def test_initialization(self):
        generator = AdversarialImageGenerator("resnet50")
        self.assertIsNotNone(generator.model)

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
