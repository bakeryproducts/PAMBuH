import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PAMBUH", 
    version="0.0.1",
    author="overfitim",
    author_email="author@example.com",
    description="yendik",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    install_requires=[
        "pillow",
        "pandas",
        "shapely",
        "rasterio",
        "requests",
        "numpy",
        "opencv-python",
        "albumentations ",
        "numpy",
        "fastprogress",
        "tqdm",
        "matplotlib",
        "torch",
        "torchvision",
        "torchsummary",
        "scipy",
        "yacs",
        "loguru",
        "ttach"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
