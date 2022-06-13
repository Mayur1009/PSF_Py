import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name='PSF_Py',
    version='0.3',
    url='https://github.com/Mayur1009/PSF_Py',
    license='GPL >= 3',
    author='Mayur Shende, Neeraj Bokde',
    author_email='mayur.k.shende@gmail.com, neerajdhanraj@gmail.com',
    description='PSF: Pattern Sequence-based Forecasting (PSF) algorithm',
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Bug Tracker": "https://github.com/Mayur1009/PSF_Py/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=[
        'pandas',
        'numpy',
        'sklearn',
        'matplotlib',
    ]
)
