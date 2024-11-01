from setuptools import setup, find_packages

setup(
    name="vision-transformer",
    version="0.1.0",
    description="Implementation of Vision Transformer (ViT) model",
    author="Vover",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "einops",
        "matplotlib",
        "numpy",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)