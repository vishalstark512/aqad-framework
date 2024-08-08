from setuptools import setup, find_packages

setup(
    name="aqad",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'pyyaml',
        'jupyter',
        'numpy',
        'pandas',
        'scikit-learn',
    ],
    python_requires='>=3.9',
)