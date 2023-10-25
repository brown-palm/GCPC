from setuptools import setup, find_packages

setup(
    name='gcpc',
    version='0.1',
    packages=find_packages(),  #same as name
    python_requires=">=3.7",
    install_requires=['torch', 'pytorch-lightning', 'wandb', 'hydra-core'],
)