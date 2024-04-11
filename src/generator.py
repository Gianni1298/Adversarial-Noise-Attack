from torchvision.io.image import read_image
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import logging as log

log.basicConfig(level=log.INFO)

AVAILABLE_MODELS = {
    "resnet50": {'model': resnet50, 'weights': ResNet50_Weights},
}


class AdversarialImageGenerator:
    def __init__(self, model_name="resnet50"):
        log.info(f"Initializing AdversarialImageGenerator with model: {model_name}")
        self.model, self.weights = self.__init_model(model_name)
        # Step 2: Initialize the inference transforms
        self.preprocess = self.weights.transforms()

    def __init_model(self, model):
        """
        Initialize the model
        :param model: The model name
        :return: The model and the weights
        """
        log.info(f"Initializing model: {model}")
        if model in AVAILABLE_MODELS:
            weights = AVAILABLE_MODELS[model]['weights'].DEFAULT
            model = AVAILABLE_MODELS[model]['model'](weights=weights)
            model.eval()
        else:
            log.error(f"Model not supported: {model}")
            raise ValueError("Model not supported")
        return model, weights


    def generate_adversarial_image(self, image_path):
        """
        Main function to generate adversarial image given an image path and target class
        :param image_path:
        :param target_class:
        :return:
        """

        # Step 1: Load the image and preprocess
        input = self.__load_and_preprocess(image_path)
        log.info("Image loaded and preprocessed")

        # Get the preprocessed image in a format that can be plotted
        preprocessed_fig = self.__plot_image(input)

        # Get original prediction
        category_name, score = self.__prediction(input)
        log.info(f"Original prediction: {category_name} with score: {score}")

        # Step 2: Select the target class
        target_index, target_category = self.__select_target_class()
        log.info(f"Selected target class: [{target_index}] {target_category}")

        # Step 2: Generate the adversarial image
        adversarial_image = self.__generate_adversarial_image(input, target_class)

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


    def __prediction(self, img_tensor):
        """
        Get the prediction for the given image tensor
        :param img_tensor: Image tensor
        :return: The prediction
        """
        prediction = self.model(img_tensor).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = self.weights.meta["categories"][class_id]
        print(f"class_id: {class_id}")
        print(f"{category_name}: {100 * score:.1f}%")
        return category_name, score

    def __select_target_class(self):
        categories = self.weights.meta["categories"]
        num_columns = 6
        num_rows = len(categories) // num_columns + (1 if len(categories) % num_columns > 0 else 0)

        print("Available categories:")
        for i in range(num_rows):
            row = ""
            for j in range(num_columns):
                index = i * num_columns + j
                if index < len(categories):
                    row += f"{index:3d}. {categories[index]:<20}"
            print(row)

        while True:
            try:
                selected_index = int(input("Enter the index of the target class: "))
                if 0 <= selected_index < len(categories):
                    return selected_index, categories[selected_index]
                else:
                    print("Invalid index. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a valid integer.")


    def __generate_adversarial_image(self, input, target_class):
        """
        Generate the adversarial image
        :param input: The input image tensor
        :param target_class: The target class
        :return: The adversarial image tensor
        """
        return input
