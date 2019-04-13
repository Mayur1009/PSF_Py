from setuptools import setup, find_packages

setup(
    name='PSF',
    version='0.1',
    packages=find_packages(),
    url='https://github.com/neerajdhanraj',
    license='GPL',
    author='Mayur Shende, Neeraj Bokde',
    author_email='mayur.k.shende@gmail.com, neerajdhanraj@gmail.com',
    description=open('README.md', 'r'),
    install_requires=[
        'pandas',
        'numpy',
        'sklearn',
        'matplotlib',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
