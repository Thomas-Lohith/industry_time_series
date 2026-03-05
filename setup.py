from setuptools import setup, find_packages

setup(
    name="mypackage",
    version="0.1.0",
    packages=find_packages(),  # Auto-discovers packages with __init__.py
    install_requires=["numpy>=1.21"],  # List dependencies
    author="Thomas Routhu",
    description="this repo is meant ofr the bridge senosr analysis",
)
