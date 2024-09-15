from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    version="0.5",
    name="face-alignment-mtcnn",
    packages=find_packages(),
    url="https://github.com/Martlgap/face-alignment",
    author="Martin Knoche",
    author_email="martin.knoche@tum.de",
    license="MIT",
    description="A lightweight face-alignment toolbox with MTCNN.",
    long_description=open("README.md").read(),
    install_requires=requirements,
)
