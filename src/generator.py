from torchvision.io.image import read_image
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
from torchvision.utils import save_image

AVAILABLE_MODELS = {
    "resnet50": {'model': resnet50, 'weights': ResNet50_Weights},
}


class AdversarialImageGenerator:
    def __init__(self, model_name="resnet50"):
        self.model, self.weights = self.__init_model(model_name)
        # Step 2: Initialize the inference transforms
        self.preprocess = self.weights.transforms()

    def __init_model(self, model):
        """
        Initialize the model
        :param model: The model name
        :return: The model and the weights
        """
        if model in AVAILABLE_MODELS:
            weights = AVAILABLE_MODELS[model]['weights'].DEFAULT
            model = AVAILABLE_MODELS[model]['model'](weights=weights)
            model.eval()
        else:
            raise ValueError("Model not supported")
        return model, weights


    def generate_adversarial_image(self, image_path, target_class):
        """
        Main function to generate adversarial image given an image path and target class
        :param image_path:
        :param target_class:
        :return:
        """

        # Step 1: Load the image and preprocess
        input = self.__load_and_preprocess(image_path)

        # Get the preprocessed image in a format that can be plotted
        preprocessed_fig = self.__plot_image(input)


        # Step 2: Generate the adversarial image

    def __load_and_preprocess(self, image_path):
        """
        Load and preprocess the image
        :param image_path:
        :return:
        """
        img = read_image(image_path)
        return self.preprocess(img).unsqueeze(0)

    def __plot_image(self, img_tensor):
        """
        Plot the given image tensor
        :param img: Image tensor
        """

        # Save the preprocessed image - not necessary to create new files
        # save_image(img_tensor, "preprocessed_image.jpg")

        # Convert the image tensor to a numpy array
        img = img_tensor.squeeze().permute(1, 2, 0).numpy()

        # Create a new figure
        fig, ax = plt.subplots()

        # Plot the image on the figure
        ax.imshow(img)
        ax.axis('off')

        return fig


