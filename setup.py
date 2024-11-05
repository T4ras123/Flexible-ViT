from setuptools import setup, find_packages

setup(
    name="vision-transformer",
    version="1.0.0",
    author="vover",
    author_email="vovatara123@gmail.com",
    description="Flexible Vision Transformer (ViT) model for your needs.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/T4ras123/Flexible-ViT",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "einops",
        "matplotlib",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)