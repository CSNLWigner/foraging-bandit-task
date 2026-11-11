from setuptools import setup, find_packages

setup(
    name="popy",
    version="0.1",
    packages=find_packages(include=["popy", "popy.*"]),
    install_requires=[
        # dependencies are listed in environment.yml
    ]
)
