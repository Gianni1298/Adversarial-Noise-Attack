# Adversarial Image Noise Library
## Introduction
The Adversarial Image Noise Library is a Python package that allows users to manipulate images by adding adversarial noise. 
The purpose of this noise is to trick an image classification model into misclassifying the altered image as a specified 
target class, regardless of the original content. This library provides an easy-to-use interface for generating adversarial 
images that can be used for testing the robustness of image classification models.

## Features
- _MUST-HAVE_: Utilizes a pre-trained image classification model from the torchvision library (NICE-TO-HAVE: customizable)
- _MUST-HAVE_: Generates adversarial noise 
- _MUST-HAVE_: Allows users to specify a target class for misclassification
- _SHOULD-HAVE_: Provides a simple and intuitive API for generating adversarial images
- _NICE-TO-HAVE_: Supports various image formats (e.g., PNG, JPEG) 

## Steps to develop:
1. ✅ Define high-level library structure and user interface 
1. Select a pre-trained image classification model from the torchvision library
2. Download some sample images to test the model