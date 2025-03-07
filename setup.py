from setuptools import setup, find_packages

setup(
    name='BALinFit',
    version='0.1.0',
    description='Linear regression using MCMC with asymmetric uncertainties',
    author='Alessandro Peca',
    author_email='peca.alessandro@gmail.com',
    url='https://github.com/yourusername/BALinFit',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'emcee',
        'corner',
        'matplotlib',
        'tqdm',
        'scipy'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
    ],
)