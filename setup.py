from setuptools import setup, find_packages

setup(
    name='AdversarialImageGenerator',
    version='1.0',
    packages=find_packages(),
    description='A package for generating adversarial images.',
    author='Gianpiero Colonna',
    url='https://github.com/Gianni1298/Adversarial-Noise-Attack',  # Use the URL to the github repo.
    install_requires=['torch', 'matplotlib', 'torchvision'],  # All your dependencies
)
