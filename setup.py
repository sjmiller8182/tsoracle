import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tsoracle",
    version="0.0.dev0",
    author="Stuart Miller",
    author_email="s-miller@ti.com",
    description="Simple time series analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sjmiller8182/tsoracle",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)