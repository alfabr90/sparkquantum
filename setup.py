try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

    def find_packages():
        return []

long_description = ""

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='sparkquantum',
    version='0.1.0',
    packages=find_packages(),
    requires=['numpy', 'matplotlib'],
    url='https://github.com/alfabr90/sparkquantum',
    license='MIT',
    author='André Albuquerque',
    author_email='alfabr90@gmail.com',
    description='Quantum algorithms simulator using Apache Spark',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    python_requires='>=3.6'
)
