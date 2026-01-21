# setup.py

import os
from setuptools import setup, find_packages

setup(
    name='slbt',
    version='0.1.0',
    packages=find_packages(),
    package_data={
        'slbt._backend': ['*.dylib', '*.so', '*.dll'],
        'slbt._preprocessing._backend': ['*.dylib', '*.so', '*.dll'],
    },
    install_requires=[
        'numpy>=1.19.0',
        'pandas>=1.0.0',
        'matplotlib>=3.0.0',
    ],
    python_requires='>=3.7',
    author='Your Name',
    author_email='your.email@example.com',
    description='Simultaneous Latent Budget Tree - A supervised ML algorithm',
    long_description=open('README.md').read() if os.path.exists('README.md') else 'SLBT Library',
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)