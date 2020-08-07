#!/usr/bin/env python3

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='PostCactus',
    version='3',
    author='Gabriele Bozzola',
    author_email='gabrielebozzola@arizona.edu',
    packages=['postcactus'],
    license='LICENSE.txt',
    description='Read and postprocess Einstein Toolkit simulations',
    install_requires=["scipy"],
)
