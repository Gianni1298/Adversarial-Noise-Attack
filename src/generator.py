import logging as log

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.io.image import read_image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.utils import save_image

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
        self.categories = self.weights.meta["categories"]

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


    def generate_adversarial_image(self, image_path, target_class):
        """
        Main function to generate adversarial image given an image path and target class
        :param image_path:
        :param target_class:
        :return:
        """

        # Step 1: Load the image and preprocess
        image = read_image(image_path)
        input_image = self.preprocess(image).unsqueeze(0)
        log.info("Image loaded and preprocessed")

        # Get original prediction
        original_class_id, category_name, _ = self.__prediction(input_image)
        log.info(f"Original prediction: {category_name}, index_class: {original_class_id}")

        # Step 2: Validate the target class
        target_category = self.__validate_target_class(target_class)
        log.info(f"Selected target class: [{target_class}] {target_category}")

        # Initialise a perturbation tensor
        delta = torch.zeros_like(input_image, requires_grad=True)

        # Generate the perturbation tensor
        delta_updated = self.__update_delta(input_image, delta, original_class_id, target_class)

        # Create the adversarial image
        adverImage = input_image + delta
        final_class_id, final_category_name, _ = self.__prediction(adverImage)
        log.info(f"Final prediction: {final_category_name}, index_class: {final_class_id}")
        save_image(adverImage, "adversarial.jpg")


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

    def show_available_categories(self):
        num_columns = 6
        num_rows = len(self.categories) // num_columns + (1 if len(self.categories) % num_columns > 0 else 0)

        print("Available categories:")
        for i in range(num_rows):
            row = ""
            for j in range(num_columns):
                index = i * num_columns + j
                if index < len(self.categories):
                    row += f"{index:3d}. {self.categories[index]:<20}"
            print(row)

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
        return class_id, category_name, score

    def __validate_target_class(self, selected_index):
        if 0 <= selected_index < len(self.categories):
            return self.categories[selected_index]
        else:
            raise ValueError("Invalid target class. Please enter a valid target class")


    def __update_delta(self, image_input, delta, original_class, target_class):
        """
        Update the perturbation tensor
        :param image_input:
        :param delta:
        :param original_class:
        :param target_class:
        :param num_of_steps:
        :return:
        """
        # Initialise the loss function
        criterion = nn.CrossEntropyLoss()

        # define the epsilon, learning rate and step number values
        epsilon = 0.01
        learning_rate = 0.01
        num_of_steps = 300

        for step in range(num_of_steps):
            adversary = image_input + delta

            predictions = self.model(adversary)

            original_loss = -criterion(predictions, torch.tensor([original_class]))

            target_loss = criterion(predictions, torch.tensor([target_class]))
            total_loss = original_loss + target_loss

            # display the loss every 10 steps
            if step % 10 == 0:
                print('step: {}, loss: {}'.format(step, total_loss.item()))

            total_loss.backward()
            gradients = delta.grad
            gradients = torch.clamp(gradients, -epsilon, epsilon)

            # Update the delta tensor
            delta.data -= learning_rate * gradients

            # Clip the updated delta tensor
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)

        return delta

