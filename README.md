# Adversarial Noise Attack
## Introduction
Adversarial Noise Attack is a Python package that allows users to manipulate images by adding adversarial noise. 
The purpose of this noise is to trick an image classification model into misclassifying the altered image as a specified 
target class, regardless of the original content. This library provides an easy-to-use interface for generating adversarial 
images that can be used for testing the robustness of image classification models.


## Installation
Requirements: Python3

```
pip install -r requirements.txt
```

## Usage
In order to generate an adversarial image, you will need to provide an input image and a target class name. The target class name
should be a valid class label from the model's output space.

```shell
python src/generator.py <input_file_name> <target_class_name>
```

For example:
```shell
python src/generator.py data/sample_images/bird.jpg airliner
```
