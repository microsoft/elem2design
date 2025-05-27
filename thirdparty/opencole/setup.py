from setuptools import setup, find_packages

setup(
    name="opencole",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "skia-python",
        "datasets",
        "langchain_core==0.2.23",
        "langchain==0.2.11",
    ],
)
