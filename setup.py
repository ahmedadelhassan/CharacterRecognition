#!/usr/bin/env python
#coding: utf-8
from setuptools import setup, find_packages

setup(
    name='CharacterRecognition',
    version='0.1.0',
    packages=find_packages(),
    url='',
    license='',
    author='Tho Vo, Ahmed Hassan, Omar Samir',
    author_email='votuongtho@gmail.com',
    description='For Machine Learning and Data Mining project',
    install_requires=[
        "scikit-image >= 0.11.3",
        "scikit-learn >= 0.16.1",
        "numpy >= 1.10.1",
        "matplotlib >= 1.4.3",
        "nltk >= 3.1"
    ],
)
